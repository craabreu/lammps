/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Charlles R. A. Abreu (UFRJ-Brazil)
------------------------------------------------------------------------- */

#include <iostream>

#include <cmath>
#include <cstring>
#include "fix_nvt_subkinetic.h"
#include "fix_nve.h"
#include "atom.h"
#include "force.h"
#include "memory.h"
// #include "neighbor.h"
#include "random_mars.h"
#include "update.h"
#include "respa.h"
#include "comm.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace FixConst;

#define P_LIM 15.0
#define confine(p) (MAX(-P_LIM, MIN(p, P_LIM)))

/* ---------------------------------------------------------------------- */

FixNVTSubkinetic::FixNVTSubkinetic(LAMMPS *lmp, int narg, char **arg) :
  FixNVE(lmp, narg, arg)
{
  if (narg != 8) error->all(FLERR,"Illegal fix nvt/subkinetic command");

  double temp = force->numeric(FLERR, arg[3]);
  L = force->numeric(FLERR, arg[4]);
  double tau = force->numeric(FLERR, arg[5]);
  gamma = force->numeric(FLERR, arg[6]);
  int seed = force->numeric(FLERR, arg[7]);

  kT = force->boltz*temp/force->mvv2e;
  Q_eta = kT*tau*tau;

  // initialize Marsaglia RNG with processor-unique seed

  random = new RanMars(lmp, seed + 9973*comm->me);
  for (int i = 0; i < 1000; i++)
    random->uniform();

  // perform initial allocation of atom-based arrays
  // register with Atom class

  v_max = NULL;
  v_eta = NULL;
  grow_arrays(atom->nmax);
  atom->add_callback(0);
}

/* ---------------------------------------------------------------------- */

FixNVTSubkinetic::~FixNVTSubkinetic()
{
  delete random;
  memory->destroy(v_eta);
  memory->destroy(v_max);
}

/* ---------------------------------------------------------------------- */

void FixNVTSubkinetic::init()
{
  FixNVE::init();

  double **v = atom->v;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int nlocal = atom->nlocal;

  double sigma_eta = sqrt(kT/Q_eta);
  for (int i = 0; i < nlocal; i++) {
    double m = rmass ? rmass[i] : mass[type[i]];
    v_max[i] = sqrt(L*kT/m);
    for (int j = 0; j < 3; j++) {
      v_eta[i][j] = sigma_eta*random->gaussian();
      v[i][j] = v_max[i]*tanh(v[i][j]/v_max[i]);
    }
  }

  a = exp(-gamma*step_respa[0]);
  b = sqrt((1.0 - a*a)*kT/Q_eta);
  bmu = -kT/Q_eta;
  amu = -(L + 1.0)*bmu;
}

/* ----------------------------------------------------------------------
   allow for both per-type and per-atom mass
------------------------------------------------------------------------- */

void FixNVTSubkinetic::initial_integrate(int /*vflag*/)
{
  double dtfm;
  double dtvm;
  double half_dtv = 0.5*dtv;

  // update v and x of atoms in group

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  double mi, vmax, v1, v2, v3;
  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      mi = rmass ? rmass[i] : mass[type[i]];
      vmax = v_max[i];
      dtfm = dtf/(mi*vmax);
      dtvm = half_dtv*vmax;
      for (int j = 0; j < 3; j++) {
        v1 = tanh(confine(atanh(v[i][j]/vmax) + dtfm*f[i][j]));
        x[i][j] += dtvm*v1;
        v_eta[i][j] += amu*v1*v1 + bmu;
        v2 = v1*exp(-v_eta[i][j]*half_dtv);
        v3 = v2/sqrt(1.0 - v1*v1 + v2*v2);
        v_eta[i][j] = a*v_eta[i][j] + b*random->gaussian();
        v2 = v3*exp(-v_eta[i][j]*half_dtv);
        v1 = v2/sqrt(1.0 - v3*v3 + v2*v2);
        v_eta[i][j] += amu*v1*v1 + bmu;
        x[i][j] += dtvm*v1;
        v[i][j] = vmax*v1;
      }
    }
}

/* ---------------------------------------------------------------------- */

void FixNVTSubkinetic::final_integrate()
{
  double dtfm;

  // update v of atoms in group

  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      double mi = rmass ? rmass[i] : mass[type[i]];
      double vmax = v_max[i];
      dtfm = dtf / (mi*vmax);
      v[i][0] = vmax*tanh(confine(atanh(v[i][0]/vmax) + dtfm*f[i][0]));
      v[i][1] = vmax*tanh(confine(atanh(v[i][1]/vmax) + dtfm*f[i][1]));
      v[i][2] = vmax*tanh(confine(atanh(v[i][2]/vmax) + dtfm*f[i][2]));
    }
}

/* ---------------------------------------------------------------------- */

double FixNVTSubkinetic::memory_usage()
{
  return FixNVE::memory_usage() + (3 + 1)*atom->nmax*sizeof(double);
}

/* ---------------------------------------------------------------------- */

void FixNVTSubkinetic::grow_arrays(int nmax)
{
  memory->grow(v_eta, nmax, 3, "fix_nvt_subkinetic:v_eta");
  memory->grow(v_max, nmax, "fix_nvt_subkinetic:v_max");
}

/* ----------------------------------------------------------------------
   copy values within local atom-based arrays
------------------------------------------------------------------------- */

void FixNVTSubkinetic::copy_arrays(int i, int j, int /*delflag*/)
{
  v_max[j] = v_max[i];
  v_eta[j][0] = v_eta[i][0];
  v_eta[j][1] = v_eta[i][1];
  v_eta[j][2] = v_eta[i][2];
}

/* ----------------------------------------------------------------------
   pack values in local atom-based arrays for exchange with another proc
------------------------------------------------------------------------- */

int FixNVTSubkinetic::pack_exchange(int i, double *buf)
{
  int m = 0;
  buf[m++] = v_max[i];
  buf[m++] = v_eta[i][0];
  buf[m++] = v_eta[i][1];
  buf[m++] = v_eta[i][2];
  return m;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based arrays from exchange with another proc
------------------------------------------------------------------------- */

int FixNVTSubkinetic::unpack_exchange(int nlocal, double *buf)
{
  int m = 0;
  v_max[nlocal] = buf[m++];
  v_eta[nlocal][0] = buf[m++];
  v_eta[nlocal][1] = buf[m++];
  v_eta[nlocal][2] = buf[m++];
  return m;
}
