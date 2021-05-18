// clang-format off
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

#include "compute_pressure_molecular.h"

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
#include "memory.h"
#include "modify.h"
#include "pair.h"
#include "pair_hybrid.h"
#include "update.h"

#include <cctype>
#include <cstring>
using namespace LAMMPS_NS;

enum{ONCE,NFREQ,EVERY};

/* ---------------------------------------------------------------------- */

ComputePressureMolecular::ComputePressureMolecular(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg),
  vptr(nullptr), mproc(nullptr), mtotal(nullptr),
  mrproc(nullptr), rcm(nullptr), mvproc(nullptr), vcm(nullptr)
{
  if (narg != 3) error->all(FLERR,"Illegal compute pressure/molecular command");
  if (igroup) error->all(FLERR,"compute pressure/molecular must use group all");

  scalar_flag = vector_flag = 1;
  size_vector = 6;
  extscalar = 0;
  extvector = 0;
  pressflag = 1;
  timeflag = 1;

  vector = new double[size_vector];
  nvirial = 0;
  vptr = nullptr;

  massneed = 1;
}

/* ---------------------------------------------------------------------- */

ComputePressureMolecular::~ComputePressureMolecular()
{
  delete [] vector;
  delete [] vptr;

  memory->destroy(mproc);
  memory->destroy(mtotal);
  memory->destroy(mrproc);
  memory->destroy(rcm);
  memory->destroy(mvproc);
  memory->destroy(vcm);
}

/* ---------------------------------------------------------------------- */

void ComputePressureMolecular::init()
{
  boltz = force->boltz;
  nktv2p = force->nktv2p;
  mvv2e = force->mvv2e;
  dimension = domain->dimension;

  // count molecules

  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
  tagint *molecule = atom->molecule;
  int maxmol = INT_MIN;
  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit)
      maxmol = MAX(molecule[i], maxmol);
  MPI_Allreduce(&maxmol, &nmols, 1, MPI_INT, MPI_MAX, world);

  allocate(nmols);

  dof = dimension*nmols;

  // detect contributions to virial
  // vptr points to all virial[6] contributions

  delete [] vptr;
  nvirial = 0;
  vptr = nullptr;

  if (force->pair) nvirial++;
  if (force->bond) nvirial++;
  if (force->angle) nvirial++;
  if (force->dihedral) nvirial++;
  if (force->improper) nvirial++;
  for (int i = 0; i < modify->nfix; i++)
    if (modify->fix[i]->thermo_virial)
      nvirial++;

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
  }

  // flag Kspace contribution separately, since not summed across procs

  if (force->kspace)
    kspace_virial = force->kspace->virial;
  else
    kspace_virial = nullptr;
}

/* ---------------------------------------------------------------------- */

void ComputePressureMolecular::setup()
{
  if (massneed) {
    compute_com();
    massneed = 0;
  }
}

/* ----------------------------------------------------------------------
   compute total pressure, averaged over Pxx, Pyy, Pzz
------------------------------------------------------------------------- */

double ComputePressureMolecular::compute_scalar()
{
  invoked_scalar = update->ntimestep;
  if (update->vflag_global != invoked_scalar)
    error->all(FLERR,"Virial was not tallied on needed timestep");

  compute_com();

  double two_ke = 0.0;
  if (dimension == 3) {
    for (int j = 0; j < nmols; j++)
      two_ke += mtotal[j]*(vcm[j][0]*vcm[j][0] + vcm[j][1]*vcm[j][1] + vcm[j][2]*vcm[j][2]);
    two_ke *= mvv2e;
    inv_volume = 1.0 / (domain->xprd * domain->yprd * domain->zprd);
    virial_compute(3,3);
    scalar = (two_ke + virial[0] + virial[1] + virial[2]) / 3.0 * inv_volume * nktv2p;
  }
  else {
    for (int j = 0; j < nmols; j++)
      two_ke += mtotal[j]*(vcm[j][0]*vcm[j][0] + vcm[j][1]*vcm[j][1]);
    two_ke *= mvv2e;
    inv_volume = 1.0 / (domain->xprd * domain->yprd);
    virial_compute(2,2);
    scalar = (two_ke + virial[0] + virial[1]) / 2.0 * inv_volume * nktv2p;
  }
  temp = two_ke/(dof*boltz);

  return scalar;
}

/* ----------------------------------------------------------------------
   compute pressure/molecular tensor
   assume KE tensor has already been computed
------------------------------------------------------------------------- */

void ComputePressureMolecular::compute_vector()
{
  invoked_vector = update->ntimestep;
  if (update->vflag_global != invoked_vector)
    error->all(FLERR,"Virial was not tallied on needed timestep");

  if (force->kspace && kspace_virial && force->kspace->scalar_pressure_flag)
    error->all(FLERR,"Must use 'kspace_modify pressure/scalar no' for "
               "tensor components with kspace_style msm");

  compute_com();

  for (int i = 0; i < 6; i++)
    ke_tensor[i] = 0.0;
  for (int i = 0; i < nmols; i++) {
    double massone = mtotal[i];
    ke_tensor[0] += vcm[i][0]*vcm[i][0] * massone;
    ke_tensor[1] += vcm[i][1]*vcm[i][1] * massone;
    ke_tensor[2] += vcm[i][2]*vcm[i][2] * massone;
    ke_tensor[3] += vcm[i][0]*vcm[i][1] * massone;
    ke_tensor[4] += vcm[i][0]*vcm[i][2] * massone;
    ke_tensor[5] += vcm[i][1]*vcm[i][2] * massone;
  }
  for (int i = 0; i < 6; i++)
    ke_tensor[i] *= mvv2e;

  if (dimension == 3) {
    inv_volume = 1.0 / (domain->xprd * domain->yprd * domain->zprd);
    virial_compute(6,3);
    for (int i = 0; i < 6; i++)
      vector[i] = (ke_tensor[i] + virial[i]) * inv_volume * nktv2p;
  } else {
    inv_volume = 1.0 / (domain->xprd * domain->yprd);
    virial_compute(4,2);
    vector[0] = (ke_tensor[0] + virial[0]) * inv_volume * nktv2p;
    vector[1] = (ke_tensor[1] + virial[1]) * inv_volume * nktv2p;
    vector[3] = (ke_tensor[3] + virial[3]) * inv_volume * nktv2p;
    vector[2] = vector[4] = vector[5] = 0.0;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePressureMolecular::virial_compute(int n, int ndiag)
{
  int i,j;
  double v[6],*vcomponent;

  for (i = 0; i < n; i++) v[i] = 0.0;

  // sum contributions to virial from forces and fixes

  for (j = 0; j < nvirial; j++) {
    vcomponent = vptr[j];
    for (i = 0; i < n; i++) v[i] += vcomponent[i];
  }

  double **x = atom->x;
  double **f = atom->f;
  int *mask = atom->mask;
  imageint *image = atom->image;
  tagint *molecule = atom->molecule;
  int nlocal = atom->nlocal;

  double dx[3], vcorr[6];
  for (i = 0; i < 6; i++) vcorr[i] = 0.0;
  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      int index = molecule[i]-1;
      if (index < 0) continue;
      domain->unmap(x[i],image[i],dx);
      dx[0] -= rcm[index][0];
      dx[1] -= rcm[index][1];
      dx[2] -= rcm[index][2];
      vcorr[0] -= dx[0]*f[i][0];
      vcorr[1] -= dx[1]*f[i][1];
      vcorr[2] -= dx[2]*f[i][2];
      vcorr[3] -= dx[0]*f[i][1];
      vcorr[4] -= dx[0]*f[i][2];
      vcorr[5] -= dx[1]*f[i][2];
    }
  for (i = 0; i < n; i++) v[i] += vcorr[i];

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

void ComputePressureMolecular::compute_com()
{
  int index;
  double massone;
  double unwrap[3];

  // zero local per-molecule values

  for (int i = 0; i < nmols; i++) {
    mrproc[i][0] = mrproc[i][1] = mrproc[i][2] = 0.0;
    mvproc[i][0] = mvproc[i][1] = mvproc[i][2] = 0.0;
  }
  if (massneed)
    for (int i = 0; i < nmols; i++)
      mproc[i] = 0.0;

  // compute COM for each molecule

  double **x = atom->x;
  double **v = atom->v;
  int *mask = atom->mask;
  int *type = atom->type;
  imageint *image = atom->image;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  tagint *molecule = atom->molecule;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      index = molecule[i]-1;
      massone = rmass ? rmass[i] : mass[type[i]];
      domain->unmap(x[i],image[i],unwrap);
      mrproc[index][0] += unwrap[0] * massone;
      mrproc[index][1] += unwrap[1] * massone;
      mrproc[index][2] += unwrap[2] * massone;
      mvproc[index][0] += v[i][0] * massone;
      mvproc[index][1] += v[i][1] * massone;
      mvproc[index][2] += v[i][2] * massone;
      if (massneed)
        mproc[index] += massone;
    }

  MPI_Allreduce(&mrproc[0][0], &rcm[0][0], 3*nmols, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(&mvproc[0][0], &vcm[0][0], 3*nmols, MPI_DOUBLE, MPI_SUM, world);
  if (massneed) MPI_Allreduce(mproc, mtotal, nmols, MPI_DOUBLE, MPI_SUM, world);

  for (int i = 0; i < nmols; i++) {
    rcm[i][0] /= mtotal[i];
    rcm[i][1] /= mtotal[i];
    rcm[i][2] /= mtotal[i];
    vcm[i][0] /= mtotal[i];
    vcm[i][1] /= mtotal[i];
    vcm[i][2] /= mtotal[i];
  }
}

void ComputePressureMolecular::allocate(int nmols)
{
  memory->destroy(mproc);
  memory->destroy(mtotal);
  memory->destroy(mrproc);
  memory->destroy(rcm);
  memory->destroy(mvproc);
  memory->destroy(vcm);

  memory->create(mproc,nmols,"pressure/molecular:mproc");
  memory->create(mtotal,nmols,"pressure/molecular:mtotal");
  memory->create(mrproc,nmols,3,"pressure/molecular:mrproc");
  memory->create(rcm,nmols,3,"pressure/molecular:rcm");
  memory->create(mvproc,nmols,3,"pressure/molecular:mvproc");
  memory->create(vcm,nmols,3,"pressure/molecular:vcm");
}

/* ----------------------------------------------------------------------
   memory usage of local data
------------------------------------------------------------------------- */

double ComputePressureMolecular::memory_usage()
{
  double bytes = (bigint) nmols * 2 * sizeof(double);
  bytes += (double) nmols * 4 * 3 * sizeof(double);
  return bytes;
}
