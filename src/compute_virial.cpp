/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "compute_virial.h"

#include "angle.h"
#include "atom.h"
#include "bond.h"
#include "dihedral.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "improper.h"
#include "kspace.h"
#include "modify.h"
#include "pair.h"
#include "pair_hybrid.h"
#include "update.h"
#include "compute_chunk_atom.h"
#include "memory.h"

#include <cctype>
#include <cstring>
using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeVirial::ComputeVirial(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg), vptr(nullptr), pstyle(nullptr), idchunk(nullptr)
{
  if (narg < 3 || narg > 4) error->all(FLERR,"Illegal compute virial command");
  if (igroup) error->all(FLERR,"compute virial must use group all");

  if (narg == 4)
    idchunk = utils::strdup(arg[3]);

  scalar_flag = vector_flag = 1;
  size_vector = 6;
  extscalar = 0;
  extvector = 0;
  pressflag = 1;
  timeflag = 1;

  vector = new double[size_vector];
  nvirial = 0;
  vptr = nullptr;
  firstflag = 1;
  nchunk = 0;
  vchunk = new double[6];
}

/* ---------------------------------------------------------------------- */

ComputeVirial::~ComputeVirial()
{
  delete [] vector;
  delete [] vptr;
  delete [] pstyle;
  delete [] idchunk;
  delete [] vchunk;
  memory->destroy(massproc);
  memory->destroy(masstotal);
  memory->destroy(com);
  memory->destroy(comall);
}

/* ---------------------------------------------------------------------- */

void ComputeVirial::init()
{
  if (idchunk) {
    int icompute = modify->find_compute(idchunk);
    if (icompute < 0)
      error->all(FLERR,"Chunk/atom compute does not exist for compute virial");
    cchunk = (ComputeChunkAtom *) modify->compute[icompute];
    if (strcmp(cchunk->style,"chunk/atom") != 0)
      error->all(FLERR,"Compute virial does not use chunk/atom compute");
  }

  boltz = force->boltz;
  nktv2p = force->nktv2p;
  dimension = domain->dimension;

  // detect contributions to virial
  // vptr points to all virial[6] contributions

  delete [] vptr;
  nvirial = 0;
  vptr = nullptr;

  if (force->pair) nvirial++;
  if (atom->molecular != Atom::ATOMIC) {
    if (force->bond) nvirial++;
    if (force->angle) nvirial++;
    if (force->dihedral) nvirial++;
    if (force->improper) nvirial++;
  }
  for (int i = 0; i < modify->nfix; i++)
    if (modify->fix[i]->thermo_virial) nvirial++;
  if (idchunk) nvirial++;

  if (nvirial) {
    vptr = new double*[nvirial];
    nvirial = 0;
    if (force->pair) vptr[nvirial++] = force->pair->virial;
    if (force->bond) vptr[nvirial++] = force->bond->virial;
    if (force->angle) vptr[nvirial++] = force->angle->virial;
    if (force->dihedral)
      vptr[nvirial++] = force->dihedral->virial;
    if (force->improper)
      vptr[nvirial++] = force->improper->virial;
    for (int i = 0; i < modify->nfix; i++)
      if (modify->fix[i]->virial_global_flag && modify->fix[i]->thermo_virial)
        vptr[nvirial++] = modify->fix[i]->virial;
    if (idchunk) vptr[nvirial++] = vchunk;
  }

  // flag Kspace contribution separately, since not summed across procs

  if (force->kspace)
    kspace_virial = force->kspace->virial;
  else
    kspace_virial = nullptr;
}

/* ----------------------------------------------------------------------
   compute total pressure, averaged over Pxx, Pyy, Pzz
------------------------------------------------------------------------- */

double ComputeVirial::compute_scalar()
{
  invoked_scalar = update->ntimestep;
  if (update->vflag_global != invoked_scalar)
    error->all(FLERR,"Virial was not tallied on needed timestep");

  if (idchunk) compute_chunk_virial();

  if (dimension == 3) {
    inv_volume = 1.0 / (domain->xprd * domain->yprd * domain->zprd);
    virial_compute(3,3);
    scalar = (virial[0] + virial[1] + virial[2]) / 3.0 * inv_volume * nktv2p;
  } else {
    inv_volume = 1.0 / (domain->xprd * domain->yprd);
    virial_compute(2,2);
    scalar = (virial[0] + virial[1]) / 2.0 * inv_volume * nktv2p;
  }

  return scalar;
}

/* ----------------------------------------------------------------------
   compute virial tensor
   assume KE tensor has already been computed
------------------------------------------------------------------------- */

void ComputeVirial::compute_vector()
{
  invoked_vector = update->ntimestep;
  if (update->vflag_global != invoked_vector)
    error->all(FLERR,"Virial was not tallied on needed timestep");

  if (force->kspace && kspace_virial && force->kspace->scalar_pressure_flag)
    error->all(FLERR,"Must use 'kspace_modify pressure/scalar no' for "
               "tensor components with kspace_style msm");

  if (idchunk) compute_chunk_virial();

  if (dimension == 3) {
    inv_volume = 1.0 / (domain->xprd * domain->yprd * domain->zprd);
    virial_compute(6,3);
    for (int i = 0; i < 6; i++)
      vector[i] = virial[i] * inv_volume * nktv2p;
  } else {
    inv_volume = 1.0 / (domain->xprd * domain->yprd);
    virial_compute(4,2);
    vector[0] = virial[0] * inv_volume * nktv2p;
    vector[1] = virial[1] * inv_volume * nktv2p;
    vector[3] = virial[3] * inv_volume * nktv2p;
    vector[2] = vector[4] = vector[5] = 0.0;
  }
}

/* ---------------------------------------------------------------------- */

void ComputeVirial::virial_compute(int n, int ndiag)
{
  int i,j;
  double v[6],*vcomponent;

  for (i = 0; i < n; i++) v[i] = 0.0;

  // sum contributions to virial from forces and fixes

  for (j = 0; j < nvirial; j++) {
    vcomponent = vptr[j];
    for (i = 0; i < n; i++) v[i] += vcomponent[i];
  }

  // sum virial across procs

  MPI_Allreduce(v,virial,n,MPI_DOUBLE,MPI_SUM,world);

  // KSpace virial contribution is already summed across procs

  if (kspace_virial)
    for (i = 0; i < n; i++) virial[i] += kspace_virial[i];


  // LJ long-range tail correction, only if pair contributions are included

  if (force->pair && force->pair->tail_flag)
    for (i = 0; i < ndiag; i++) virial[i] += force->pair->ptail * inv_volume;

}

/* ---------------------------------------------------------------------- */

void ComputeVirial::reset_extra_compute_fix(const char *id_new)
{
}

/* ---------------------------------------------------------------------- */

void ComputeVirial::compute_chunk_virial()
{
  int index;
  double massone;
  double unwrap[3];

  // compute chunk/atom assigns atoms to chunk IDs
  // extract ichunk index vector from compute
  // ichunk = 1 to Nchunk for included atoms, 0 for excluded atoms

  int n = cchunk->setup_chunks();
  cchunk->compute_ichunk();
  int *ichunk = cchunk->ichunk;

  // first time call, allocate per-chunk arrays
  // thereafter, require nchunk remain the same

  if (firstflag) {
    nchunk = n;
    memory->create(massproc,nchunk,"virial:massproc");
    memory->create(masstotal,nchunk,"virial:masstotal");
    memory->create(com,nchunk,3,"virial:com");
    memory->create(comall,nchunk,3,"virial:comall");
    firstflag = 0;
  }
  else if (n != nchunk)
    error->all(FLERR,"Compute virial nchunk is not static");

  // zero local per-chunk values

  for (int i = 0; i < nchunk; i++) {
    massproc[i] = 0.0;
    com[i][0] = com[i][1] = com[i][2] = 0.0;
  }

  // compute current COM for each chunk

  double **x = atom->x;
  double **f = atom->f;
  int *mask = atom->mask;
  int *type = atom->type;
  imageint *image = atom->image;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      index = ichunk[i]-1;
      if (index < 0) continue;
      if (rmass) massone = rmass[i];
      else massone = mass[type[i]];
      domain->unmap(x[i],image[i],unwrap);
      massproc[index] += massone;
      com[index][0] += unwrap[0] * massone;
      com[index][1] += unwrap[1] * massone;
      com[index][2] += unwrap[2] * massone;
    }

  MPI_Allreduce(massproc,masstotal,nchunk,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(&com[0][0],&comall[0][0],3*nchunk,MPI_DOUBLE,MPI_SUM,world);

  for (int i = 0; i < nchunk; i++)
    if (masstotal[i] > 0.0) {
      comall[i][0] /= masstotal[i];
      comall[i][1] /= masstotal[i];
      comall[i][2] /= masstotal[i];
    }

  for (int i = 0; i < 6; i++) vchunk[i] = 0.0;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      index = ichunk[i]-1;
      if (index < 0) continue;
      domain->unmap(x[i],image[i],unwrap);
      double dx = comall[index][0] - unwrap[0];
      double dy = comall[index][1] - unwrap[1];
      double dz = comall[index][2] - unwrap[2];
      double fx = f[i][0];
      double fy = f[i][1];
      double fz = f[i][2];
      vchunk[0] += dx*fx;
      vchunk[1] += dy*fy;
      vchunk[2] += dz*fz;
      vchunk[3] += dx*fy;
      vchunk[4] += dx*fz;
      vchunk[5] += dy*fz;
    }
}

/* ----------------------------------------------------------------------
   memory usage of local data
------------------------------------------------------------------------- */

double ComputeVirial::memory_usage()
{
  double bytes = (double) nchunk * 8 * sizeof(double);
  return bytes;
}
