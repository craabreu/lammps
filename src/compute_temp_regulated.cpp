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

#include "compute_temp_regulated.h"

#include "atom.h"
#include "update.h"
#include "force.h"
#include "domain.h"
#include "group.h"
#include "error.h"

using namespace LAMMPS_NS;

enum {SEMIREGULATED, REGULATED};

/* ---------------------------------------------------------------------- */

ComputeTempRegulated::ComputeTempRegulated(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg < 5 || narg > 6) error->all(FLERR,"Illegal compute temp command");

  scalar_flag = vector_flag = 1;
  size_vector = 6;
  extscalar = 0;
  extvector = 1;
  tempflag = 1;

  double n = utils::numeric(FLERR, arg[3], false, lmp);
  double temp = utils::numeric(FLERR, arg[4], false, lmp);

  if (narg == 5)
    regulation_type = SEMIREGULATED;
  else if (strcmp(arg[5],"semi") == 0)
    regulation_type = SEMIREGULATED;
  else if (strcmp(arg[5],"full") == 0)
    regulation_type = REGULATED;
  else
    error->all(FLERR,"Illegal compute temp command");

  nkT = n * force->boltz * temp / force->mvv2e;
  np1kT = (n + 1) * force->boltz * temp / force->mvv2e;

  vector = new double[size_vector];
}

/* ---------------------------------------------------------------------- */

ComputeTempRegulated::~ComputeTempRegulated()
{
  if (!copymode)
    delete [] vector;
}

/* ---------------------------------------------------------------------- */

void ComputeTempRegulated::setup()
{
  dynamic = 0;
  if (dynamic_user || group->dynamic[igroup]) dynamic = 1;
  dof_compute();
}

/* ---------------------------------------------------------------------- */

void ComputeTempRegulated::dof_compute()
{
  adjust_dof_fix();
  natoms_temp = group->count(igroup);
  dof = domain->dimension * natoms_temp;
  if (dof > 0.0) tfactor = force->mvv2e / (dof * force->boltz);
  else tfactor = 0.0;
}

/* ---------------------------------------------------------------------- */

double ComputeTempRegulated::compute_scalar()
{
  invoked_scalar = update->ntimestep;

  double **v = atom->v;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  double t = 0.0;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      double imass = rmass ? rmass[i] : mass[type[i]];
      double mumax = sqrt(imass*nkT);
      double umaxinv = imass/mumax;
      double u0 = tanh(v[i][0]*umaxinv);
      double u1 = tanh(v[i][1]*umaxinv);
      double u2 = tanh(v[i][2]*umaxinv);
      if (regulation_type == REGULATED)
        t += u0*u0 + u1*u1 + u2*u2;
      else
        t += mumax*(v[i][0]*u0 + v[i][1]*u1 + v[i][2]*u2);
    }
  if (regulation_type == REGULATED) t *= np1kT;

  MPI_Allreduce(&t,&scalar,1,MPI_DOUBLE,MPI_SUM,world);
  if (dynamic) dof_compute();
  if (dof < 0.0 && natoms_temp > 0.0)
    error->all(FLERR,"Temperature compute degrees of freedom < 0");
  scalar *= tfactor;
  return scalar;
}

/* ---------------------------------------------------------------------- */

void ComputeTempRegulated::compute_vector()
{
  int i;

  invoked_vector = update->ntimestep;

  double **v = atom->v;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  double massone,t[6];
  for (i = 0; i < 6; i++) t[i] = 0.0;

  for (i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      massone = rmass ? rmass[i] : mass[type[i]];
      double mumax = sqrt(massone*nkT);
      double umaxinv = massone/mumax;
      double u0 = tanh(v[i][0]*umaxinv);
      double u1 = tanh(v[i][1]*umaxinv);
      double u2 = tanh(v[i][2]*umaxinv);
      if (regulation_type == REGULATED) {
        t[0] += u0*u0;
        t[1] += u1*u1;
        t[2] += u2*u2;
        t[3] += u0*u1;
        t[4] += u0*u2;
        t[5] += u1*u2;
      }
      else {
        t[0] += mumax*v[i][0]*u0;
        t[1] += mumax*v[i][1]*u1;
        t[2] += mumax*v[i][2]*u2;
        t[3] += mumax*v[i][0]*u1;
        t[4] += mumax*v[i][0]*u2;
        t[5] += mumax*v[i][1]*u2;
      }
    }

  MPI_Allreduce(t,vector,6,MPI_DOUBLE,MPI_SUM,world);

  if (regulation_type == REGULATED)
    for (i = 0; i < 6; i++) vector[i] *= np1kT*force->mvv2e;
  else
    for (i = 0; i < 6; i++) vector[i] *= force->mvv2e;
}
