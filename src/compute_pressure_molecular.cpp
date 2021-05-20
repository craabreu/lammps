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

/* ---------------------------------------------------------------------- */

ComputePressureMolecular::ComputePressureMolecular(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg),
  vptr(nullptr), m_proc(nullptr), m_total(nullptr), mr_proc(nullptr), rcm(nullptr)
{
  if (narg != 4) error->all(FLERR,"Illegal compute pressure/molecular command");
  if (igroup) error->all(FLERR,"compute pressure/molecular must use group all");

  id_temp = strdup(arg[3]);
  keflag = strcmp(id_temp, "NULL") != 0;

  scalar_flag = vector_flag = 1;
  size_vector = 6;
  extscalar = 0;
  extvector = 0;
  pressflag = 1;
  timeflag = 1;

  vector = new double[size_vector];
  nvirial = 0;
  vptr = nullptr;

  mass_needed = 1;
  ke_tensor_flag = 0;
}

/* ---------------------------------------------------------------------- */

ComputePressureMolecular::~ComputePressureMolecular()
{
  delete [] vector;
  delete [] vptr;
  delete [] id_temp;

  if (ke_tensor_flag) delete [] ke_tensor;

  memory->destroy(m_proc);
  memory->destroy(m_total);
  memory->destroy(mr_proc);
  memory->destroy(rcm);
}

/* ---------------------------------------------------------------------- */

void ComputePressureMolecular::init()
{
  if (keflag) {
    int icompute = modify->find_compute(id_temp);
    if (icompute < 0)
      error->all(FLERR,"Temperature ID for compute pressure/molecular does not exist");
    temperature = (ComputeTempMolecular *) modify->compute[icompute];
    if (strcmp(temperature->style, "temp/molecular") != 0)
      error->all(FLERR,"Temperature compute for pressure/molecular is not temp/molecular style");
    ke_tensor = temperature->vector;
  }
  else {
    two_ke = 0.0;
    ke_tensor_flag = 1;
    ke_tensor = new double[6];
    for (int i = 0; i < 6; i++)
      ke_tensor[i] = 0.0;
  }

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
  MPI_Allreduce(&maxmol, &nmolecules, 1, MPI_INT, MPI_MAX, world);

  allocate(nmolecules);

  dof = dimension*nmolecules;

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
  if (mass_needed) {
    compute_com();
    mass_needed = 0;
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

  if (keflag) {
    if (temperature->invoked_scalar != update->ntimestep)
      two_ke = dof*boltz*temperature->compute_scalar();
    else
      two_ke = dof*boltz*temperature->scalar;
  }

  if (dimension == 3) {
    inv_volume = 1.0 / (domain->xprd * domain->yprd * domain->zprd);
    virial_compute(3, 3);
    scalar = (two_ke + virial[0] + virial[1] + virial[2]) / 3.0 * inv_volume * nktv2p;
  }
  else {
    inv_volume = 1.0 / (domain->xprd * domain->yprd);
    virial_compute(2, 2);
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

  if (keflag)
    if (temperature->invoked_vector != update->ntimestep)
      temperature->compute_vector();

  if (dimension == 3) {
    inv_volume = 1.0 / (domain->xprd * domain->yprd * domain->zprd);
    virial_compute(6,3);
    for (int i = 0; i < 6; i++)
      vector[i] = (ke_tensor[i] + virial[i]) * inv_volume * nktv2p;
  }
  else {
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
      if (index < 0)
        continue;
      domain->unmap(x[i], image[i], dx);
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
  int imol;
  double massone;
  double unwrap[3];

  // zero local per-molecule values

  for (int i = 0; i < nmolecules; i++)
    mr_proc[i][0] = mr_proc[i][1] = mr_proc[i][2] = 0.0;
  if (mass_needed)
    for (int i = 0; i < nmolecules; i++)
      m_proc[i] = 0.0;

  // compute COM variables for each molecule

  double **x = atom->x;
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
      imol = molecule[i] - 1;
      massone = rmass ? rmass[i] : mass[type[i]];
      domain->unmap(x[i], image[i], unwrap);

      mr_proc[imol][0] += massone * unwrap[0];
      mr_proc[imol][1] += massone * unwrap[1];
      mr_proc[imol][2] += massone * unwrap[2];

      if (mass_needed) m_proc[imol] += massone;
    }

  MPI_Allreduce(&mr_proc[0][0], &rcm[0][0], 3*nmolecules, MPI_DOUBLE, MPI_SUM, world);
  if (mass_needed)
    MPI_Allreduce(m_proc, m_total, nmolecules, MPI_DOUBLE, MPI_SUM, world);

  for (int i = 0; i < nmolecules; i++) {
    rcm[i][0] /= m_total[i];
    rcm[i][1] /= m_total[i];
    rcm[i][2] /= m_total[i];
  }
}

void ComputePressureMolecular::allocate(int nmolecules)
{
  memory->destroy(m_proc);
  memory->destroy(m_total);
  memory->destroy(mr_proc);
  memory->destroy(rcm);

  memory->create(m_proc, nmolecules, "pressure/molecular:m_proc");
  memory->create(m_total, nmolecules, "pressure/molecular:m_total");
  memory->create(mr_proc, nmolecules, 3, "pressure/molecular:mr_proc");
  memory->create(rcm, nmolecules, 3, "pressure/molecular:rcm");
}

/* ----------------------------------------------------------------------
   memory usage of local data
------------------------------------------------------------------------- */

double ComputePressureMolecular::memory_usage()
{
  double bytes = (bigint) nmolecules * 2 * sizeof(double);
  bytes += (double) nmolecules * 2 * 3 * sizeof(double);
  return bytes;
}
