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

#include "compute_temp_molecular.h"

#include "atom.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "update.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeTempMolecular::ComputeTempMolecular(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg),
  m_proc(nullptr), m_total(nullptr),
  mv_proc(nullptr), vcm(nullptr),
  mu_proc(nullptr), ucm(nullptr)
{
  if (narg < 3) error->all(FLERR,"Illegal compute temp/molecular command");
  if (igroup) error->all(FLERR,"compute temp/molecular must use group all");

  regulation_flag = 0;

  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"regulation") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal fix temp/molecular command");
      regulation_flag = 1;
      double n = utils::numeric(FLERR, arg[iarg+1], false, lmp);
      double temp = utils::numeric(FLERR, arg[iarg+2], false, lmp);
      if (n <= 0.0 || temp <= 0.0) error->all(FLERR,"Illegal fix temp/molecular command");
      nkT = n * force->boltz * temp / force->mvv2e;
      iarg += 3;
    }
    else
      error->all(FLERR,"Illegal fix temp/molecular command");
  }

  scalar_flag = vector_flag = 1;
  size_vector = 6;
  extscalar = 0;
  extvector = 1;
  timeflag = 1;

  vector = new double[size_vector];

  mass_needed = 1;
}

/* ---------------------------------------------------------------------- */

ComputeTempMolecular::~ComputeTempMolecular()
{
  delete [] vector;

  memory->destroy(m_proc);
  memory->destroy(m_total);
  memory->destroy(mv_proc);
  memory->destroy(vcm);
  if (regulation_flag) {
    memory->destroy(mu_proc);
    memory->destroy(ucm);
  }
}

/* ---------------------------------------------------------------------- */

void ComputeTempMolecular::init()
{
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

  dimension = domain->dimension;
  dof = dimension*nmolecules;
  tfactor = force->mvv2e/(dof*force->boltz);
}

/* ---------------------------------------------------------------------- */

void ComputeTempMolecular::setup()
{
  if (mass_needed) {
    compute_com();
    mass_needed = 0;
  }
}

/* ----------------------------------------------------------------------
   compute molecular temperature
------------------------------------------------------------------------- */

double ComputeTempMolecular::compute_scalar()
{
  invoked_scalar = update->ntimestep;

  compute_com();

  double two_ke = 0.0;
  if (dimension == 3)
    for (int j = 0; j < nmolecules; j++)
      two_ke += m_total[j]*(vcm[j][0]*ucm[j][0] + vcm[j][1]*ucm[j][1] + vcm[j][2]*ucm[j][2]);
  else
    for (int j = 0; j < nmolecules; j++)
      two_ke += m_total[j]*(vcm[j][0]*ucm[j][0] + vcm[j][1]*ucm[j][1]);

  scalar = tfactor*two_ke;

  return scalar;
}

/* ----------------------------------------------------------------------
   compute KE tensor
------------------------------------------------------------------------- */

void ComputeTempMolecular::compute_vector()
{
  invoked_vector = update->ntimestep;

  compute_com();

  for (int i = 0; i < 6; i++)
    vector[i] = 0.0;

  for (int j = 0; j < nmolecules; j++) {
    vector[0] += m_total[j]*vcm[j][0]*ucm[j][0];
    vector[1] += m_total[j]*vcm[j][1]*ucm[j][1];
    vector[2] += m_total[j]*vcm[j][2]*ucm[j][2];
    vector[3] += m_total[j]*vcm[j][0]*ucm[j][1];
    vector[4] += m_total[j]*vcm[j][0]*ucm[j][2];
    vector[5] += m_total[j]*vcm[j][1]*ucm[j][2];
  }

  for (int i = 0; i < 6; i++)
    vector[i] *= force->mvv2e;
}

/* ---------------------------------------------------------------------- */

void ComputeTempMolecular::compute_com()
{
  int imol;
  double massone, m_umax, umax_inv;

  // zero local per-molecule values

  for (int i = 0; i < nmolecules; i++) {
    mv_proc[i][0] = mv_proc[i][1] = mv_proc[i][2] = 0.0;
    if (regulation_flag) mu_proc[i][0] = mu_proc[i][1] = mu_proc[i][2] = 0.0;
  }
  if (mass_needed)
    for (int i = 0; i < nmolecules; i++)
      m_proc[i] = 0.0;

  // compute COM variables for each molecule

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
      imol = molecule[i] - 1;
      massone = rmass ? rmass[i] : mass[type[i]];

      mv_proc[imol][0] += massone * v[i][0];
      mv_proc[imol][1] += massone * v[i][1];
      mv_proc[imol][2] += massone * v[i][2];

      if (regulation_flag) {
        m_umax = sqrt(massone*nkT);
        umax_inv = massone/m_umax;
        mu_proc[imol][0] += m_umax * tanh(v[i][0] * umax_inv);
        mu_proc[imol][1] += m_umax * tanh(v[i][1] * umax_inv);
        mu_proc[imol][2] += m_umax * tanh(v[i][2] * umax_inv);
      }

      if (mass_needed) m_proc[imol] += massone;
    }

  MPI_Allreduce(&mv_proc[0][0], &vcm[0][0], 3*nmolecules, MPI_DOUBLE, MPI_SUM, world);
  if (regulation_flag)
    MPI_Allreduce(&mu_proc[0][0], &ucm[0][0], 3*nmolecules, MPI_DOUBLE, MPI_SUM, world);
  if (mass_needed)
    MPI_Allreduce(m_proc, m_total, nmolecules, MPI_DOUBLE, MPI_SUM, world);

  for (int i = 0; i < nmolecules; i++) {
    vcm[i][0] /= m_total[i];
    vcm[i][1] /= m_total[i];
    vcm[i][2] /= m_total[i];

    if (regulation_flag) {
      ucm[i][0] /= m_total[i];
      ucm[i][1] /= m_total[i];
      ucm[i][2] /= m_total[i];
    }
  }
}

/* ---------------------------------------------------------------------- */

void ComputeTempMolecular::allocate(int nmolecules)
{
  memory->destroy(m_proc);
  memory->destroy(m_total);
  memory->destroy(mv_proc);
  memory->destroy(vcm);
  if (regulation_flag) {
    memory->destroy(mu_proc);
    memory->destroy(ucm);
  }
  memory->create(m_proc, nmolecules, "temp/molecular:m_proc");
  memory->create(m_total, nmolecules, "temp/molecular:m_total");
  memory->create(mv_proc, nmolecules, 3, "temp/molecular:mv_proc");
  memory->create(vcm, nmolecules, 3, "temp/molecular:vcm");
  if (regulation_flag) {
    memory->create(mu_proc, nmolecules, 3, "temp/molecular:mu_proc");
    memory->create(ucm, nmolecules, 3, "temp/molecular:ucm");
  }
  else
    ucm = vcm;
}

/* ----------------------------------------------------------------------
   memory usage of local data
------------------------------------------------------------------------- */

double ComputeTempMolecular::memory_usage()
{
  double bytes = (bigint) nmolecules * 2 * sizeof(double);
  bytes += (double) nmolecules * 2 * 3 * sizeof(double);
  if (regulation_flag) bytes += (double) nmolecules * 2 * 3 * sizeof(double);
  return bytes;
}
