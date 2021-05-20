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

#include "compute_pressure_regulated.h"

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

ComputePressureRegulated::ComputePressureRegulated(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg),
  vptr(nullptr), m_proc(nullptr), m_total(nullptr), mr_proc(nullptr), rcm(nullptr),
  mv_proc(nullptr), vcm(nullptr), mu_proc(nullptr), ucm(nullptr)
{
  if (narg != 5) error->all(FLERR,"Illegal compute pressure/regulated command");
  if (igroup) error->all(FLERR,"compute pressure/regulated must use group all");

  double n = utils::numeric(FLERR, arg[3], false, lmp);
  double temp = utils::numeric(FLERR, arg[4], false, lmp);

  nkT = n * force->boltz * temp / force->mvv2e;

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
}

/* ---------------------------------------------------------------------- */

ComputePressureRegulated::~ComputePressureRegulated()
{
  delete [] vector;
  delete [] vptr;

  memory->destroy(m_proc);
  memory->destroy(m_total);
  memory->destroy(mr_proc);
  memory->destroy(rcm);
  memory->destroy(mv_proc);
  memory->destroy(vcm);
  memory->destroy(mu_proc);
  memory->destroy(ucm);
}

/* ---------------------------------------------------------------------- */

void ComputePressureRegulated::init()
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

void ComputePressureRegulated::setup()
{
  if (mass_needed) {
    compute_com();
    mass_needed = 0;
  }
}

/* ----------------------------------------------------------------------
   compute total pressure, averaged over Pxx, Pyy, Pzz
------------------------------------------------------------------------- */

double ComputePressureRegulated::compute_scalar()
{
  invoked_scalar = update->ntimestep;
  if (update->vflag_global != invoked_scalar)
    error->all(FLERR,"Virial was not tallied on needed timestep");

  compute_com();

  double two_ke = 0.0;
  if (dimension == 3) {
    for (int j = 0; j < nmolecules; j++)
      two_ke += m_total[j]*(vcm[j][0]*ucm[j][0] + vcm[j][1]*ucm[j][1] + vcm[j][2]*ucm[j][2]);
    two_ke *= mvv2e;
    inv_volume = 1.0 / (domain->xprd * domain->yprd * domain->zprd);
    virial_compute(3, 3);
    scalar = (two_ke + virial[0] + virial[1] + virial[2]) / 3.0 * inv_volume * nktv2p;
  }
  else {
    for (int j = 0; j < nmolecules; j++)
      two_ke += m_total[j]*(vcm[j][0]*ucm[j][0] + vcm[j][1]*ucm[j][1]);
    two_ke *= mvv2e;
    inv_volume = 1.0 / (domain->xprd * domain->yprd);
    virial_compute(2, 2);
    scalar = (two_ke + virial[0] + virial[1]) / 2.0 * inv_volume * nktv2p;
  }
  temp = two_ke/(dof*boltz);

  return scalar;
}

/* ----------------------------------------------------------------------
   compute pressure/regulated tensor
   assume KE tensor has already been computed
------------------------------------------------------------------------- */

void ComputePressureRegulated::compute_vector()
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
  for (int j = 0; j < nmolecules; j++) {
    ke_tensor[0] += m_total[j]*vcm[j][0]*ucm[j][0];
    ke_tensor[1] += m_total[j]*vcm[j][1]*ucm[j][1];
    ke_tensor[2] += m_total[j]*vcm[j][2]*ucm[j][2];
    ke_tensor[3] += m_total[j]*vcm[j][0]*ucm[j][1];
    ke_tensor[4] += m_total[j]*vcm[j][0]*ucm[j][2];
    ke_tensor[5] += m_total[j]*vcm[j][1]*ucm[j][2];
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

void ComputePressureRegulated::virial_compute(int n, int ndiag)
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

void ComputePressureRegulated::compute_com()
{
  int index;
  double massone, mumax, umaxinv;
  double unwrap[3];

  // zero local per-molecule values

  for (int i = 0; i < nmolecules; i++) {
    mr_proc[i][0] = mr_proc[i][1] = mr_proc[i][2] = 0.0;
    mv_proc[i][0] = mv_proc[i][1] = mv_proc[i][2] = 0.0;
    mu_proc[i][0] = mu_proc[i][1] = mu_proc[i][2] = 0.0;
  }
  if (mass_needed)
    for (int i = 0; i < nmolecules; i++)
      m_proc[i] = 0.0;

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
  if (igroup == atom->firstgroup)
    nlocal = atom->nfirst;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      index = molecule[i]-1;
      massone = rmass ? rmass[i] : mass[type[i]];
      domain->unmap(x[i], image[i], unwrap);
      mumax = sqrt(massone*nkT);
      umaxinv = massone/mumax;

      mr_proc[index][0] += massone * unwrap[0];
      mr_proc[index][1] += massone * unwrap[1];
      mr_proc[index][2] += massone * unwrap[2];

      mu_proc[index][0] += mumax * tanh(v[i][0] * umaxinv);
      mu_proc[index][1] += mumax * tanh(v[i][1] * umaxinv);
      mu_proc[index][2] += mumax * tanh(v[i][2] * umaxinv);

      mv_proc[index][0] += massone * v[i][0];
      mv_proc[index][1] += massone * v[i][1];
      mv_proc[index][2] += massone * v[i][2];

      if (mass_needed)
        m_proc[index] += massone;
    }

  MPI_Allreduce(&mr_proc[0][0], &rcm[0][0], 3*nmolecules, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(&mu_proc[0][0], &ucm[0][0], 3*nmolecules, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(&mv_proc[0][0], &vcm[0][0], 3*nmolecules, MPI_DOUBLE, MPI_SUM, world);
  if (mass_needed) MPI_Allreduce(m_proc, m_total, nmolecules, MPI_DOUBLE, MPI_SUM, world);

  for (int i = 0; i < nmolecules; i++) {
    rcm[i][0] /= m_total[i];
    rcm[i][1] /= m_total[i];
    rcm[i][2] /= m_total[i];

    ucm[i][0] /= m_total[i];
    ucm[i][1] /= m_total[i];
    ucm[i][2] /= m_total[i];

    vcm[i][0] /= m_total[i];
    vcm[i][1] /= m_total[i];
    vcm[i][2] /= m_total[i];
  }
}

void ComputePressureRegulated::allocate(int nmolecules)
{
  memory->destroy(m_proc);
  memory->destroy(m_total);
  memory->destroy(mr_proc);
  memory->destroy(rcm);
  memory->destroy(mv_proc);
  memory->destroy(vcm);
  memory->destroy(mu_proc);
  memory->destroy(ucm);

  memory->create(m_proc, nmolecules, "pressure/regulated:m_proc");
  memory->create(m_total, nmolecules, "pressure/regulated:m_total");
  memory->create(mr_proc, nmolecules, 3, "pressure/regulated:mr_proc");
  memory->create(rcm, nmolecules, 3, "pressure/regulated:rcm");
  memory->create(mv_proc, nmolecules, 3, "pressure/regulated:mv_proc");
  memory->create(vcm, nmolecules, 3, "pressure/regulated:vcm");
  memory->create(mu_proc, nmolecules, 3, "pressure/regulated:mu_proc");
  memory->create(ucm, nmolecules, 3, "pressure/regulated:ucm");
}

/* ----------------------------------------------------------------------
   memory usage of local data
------------------------------------------------------------------------- */

double ComputePressureRegulated::memory_usage()
{
  double bytes = (bigint) nmolecules * 2 * sizeof(double);
  bytes += (double) nmolecules * 6 * 3 * sizeof(double);
  return bytes;
}
