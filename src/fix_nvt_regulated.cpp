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

#include "fix_nvt_regulated.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "random_mars.h"
#include "respa.h"
#include "update.h"

#include <cmath>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixNVTRegulated::FixNVTRegulated(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 7)
    error->all(FLERR,"Illegal fix nvt/regulated command");

  // fix ID group nvt/regulated t_start t_stop t_damp gamma seed

  dynamic_group_allow = 1;
  time_integrate = 1;
  scalar_flag = 1;
  extscalar = 1;
  ecouple_flag = 1;

  temp = utils::numeric(FLERR, arg[3], false, lmp);
  tau = utils::numeric(FLERR, arg[4], false, lmp);
  gamma = utils::numeric(FLERR, arg[5], false, lmp);
  seed = utils::inumeric(FLERR, arg[6], false, lmp);

  if (tau <= 0.0) error->all(FLERR,"Fix nvt/regulated characteristic time must be > 0.0");
  if (gamma <= 0.0) error->all(FLERR,"Fix nvt/regulated friction coefficient must be > 0.0");
  if (seed <= 0) error->all(FLERR,"Fix nvt/regulated seed must be > 0");

  n = 1.0;

  int iarg = 7;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"regulation") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix nvt/regulated command");
      n = utils::inumeric(FLERR, arg[iarg+1], false, lmp);
      iarg += 2;
    } else error->all(FLERR,"Illegal fix nvt/regulated command");
  }

  // initialize Marsaglia RNG with processor-unique seed

  random = new RanMars(lmp, seed + 143*comm->me);

  // setup atom-based arrays
  // register with Atom class
  // no need to set peratom_flag, b/c data is for internal use only

  c = pscale = nullptr;
  p = v_eta = nullptr;
  grow_arrays(atom->nmax);
  atom->add_callback(Atom::GROW);
}

/* ---------------------------------------------------------------------- */

int FixNVTRegulated::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  mask |= INITIAL_INTEGRATE_RESPA;
  mask |= FINAL_INTEGRATE_RESPA;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixNVTRegulated::init()
{
  dtv = 0.5 * update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;

  if (utils::strmatch(update->integrate_style, "^respa"))
    step_respa = ((Respa *) update->integrate)->step;

  // Store speed limits and momentum scaling factors

  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  double nkT = n*force->boltz*temp/force->mvv2e;
  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      double mi = rmass ? rmass[i] : mass[type[i]];
      c[i] = sqrt(nkT/mi);
      pscale[i] = 1.0/(mi*c[i]);
    }

  convert_velocities();
}

/* ---------------------------------------------------------------------- */

void FixNVTRegulated::setup(int vflag)
{
  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  int out_of_range = 0;
  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit)
      for (int k = 0; k < 3; k++)
        if (abs(v[i][k]) >= c[i]) {
          out_of_range = 1;
          break;
        }

  if (out_of_range) convert_velocities();
}

/* ----------------------------------------------------------------------
   allow for both per-type and per-atom mass
------------------------------------------------------------------------- */

void FixNVTRegulated::initial_integrate(int /*vflag*/)
{
  double factor;

  // update v and x of atoms in group

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      p[i][0] += dtf * f[i][0];
      p[i][1] += dtf * f[i][1];
      p[i][2] += dtf * f[i][2];

      double dtv_ci = dtv*c[i];
      double factor = pscale[i];

      x[i][0] += dtv_ci*tanh(factor*p[i][0]);
      x[i][1] += dtv_ci*tanh(factor*p[i][1]);
      x[i][2] += dtv_ci*tanh(factor*p[i][2]);

      x[i][0] += dtv_ci*tanh(factor*p[i][0]);
      x[i][1] += dtv_ci*tanh(factor*p[i][1]);
      x[i][2] += dtv_ci*tanh(factor*p[i][2]);
    }
}

/* ---------------------------------------------------------------------- */

void FixNVTRegulated::final_integrate()
{
  double **f = atom->f;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      p[i][0] += dtf * f[i][0];
      p[i][1] += dtf * f[i][1];
      p[i][2] += dtf * f[i][2];
    }
}

/* ---------------------------------------------------------------------- */

void FixNVTRegulated::initial_integrate_respa(int vflag, int ilevel, int /*iloop*/)
{
  dtv = 0.5 * step_respa[ilevel];
  dtf = 0.5 * step_respa[ilevel] * force->ftm2v;

  // innermost level - NVE update of v and x
  // all other levels - NVE update of v

  if (ilevel == 0)
    initial_integrate(vflag);
  else
    final_integrate();
}

/* ---------------------------------------------------------------------- */

void FixNVTRegulated::final_integrate_respa(int ilevel, int /*iloop*/)
{
  dtf = 0.5 * step_respa[ilevel] * force->ftm2v;
  final_integrate();
}

/* ---------------------------------------------------------------------- */

void FixNVTRegulated::end_of_step()
{
  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      double ci = c[i];
      double factor = pscale[i];
      v[i][0] = ci*tanh(factor*p[i][0]);
      v[i][1] = ci*tanh(factor*p[i][1]);
      v[i][2] = ci*tanh(factor*p[i][2]);
    }
}

/* ---------------------------------------------------------------------- */

double FixNVTRegulated::compute_scalar()
{
  double ke_std, ke_reg, energy, total;

  double **v = atom->v;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  double nkT = n*force->boltz*temp/force->mvv2e;

  ke_std = ke_reg = 0.0;
  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      double mi = rmass ? rmass[i] : mass[type[i]];
      double factor = 1.0/sqrt(mi*nkT);
      ke_std += mi*(v[i][0]*v[i][0] + v[i][1]*v[i][1] + v[i][2]*v[i][2]);
      ke_reg += logcosh(factor*p[i][0]) + logcosh(factor*p[i][1]) + logcosh(factor*p[i][2]);
    }

  energy = force->mvv2e*(nkT*ke_reg - 0.5*ke_std);
  MPI_Allreduce(&energy, &total, 1, MPI_DOUBLE, MPI_SUM, world);
  return total;
}

/* ---------------------------------------------------------------------- */

void FixNVTRegulated::reset_dt()
{
  dtv = 0.5 * update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
}

/* ---------------------------------------------------------------------- */

void FixNVTRegulated::convert_velocities()
{
  double **v = atom->v;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      double mi = rmass ? rmass[i] : mass[type[i]];
      for (int k = 0; k < 3; k++)
        p[i][k] = mi*v[i][k];
    }

  end_of_step();
}

/* ----------------------------------------------------------------------
   memory usage of tally array
------------------------------------------------------------------------- */

double FixNVTRegulated::memory_usage()
{
  double bytes = (double)atom->nmax*8*sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
   allocate atom-based array for v_eta
------------------------------------------------------------------------- */

void FixNVTRegulated::grow_arrays(int nmax)
{
  memory->grow(c, nmax, "fix_nvt_regulated:c");
  memory->grow(pscale, nmax, "fix_nvt_regulated:pscale");
  memory->grow(p, nmax, 3, "fix_nvt_regulated:p");
  memory->grow(v_eta, nmax, 3, "fix_nvt_regulated:v_eta");
}

/* ----------------------------------------------------------------------
   copy values within local atom-based array
------------------------------------------------------------------------- */

void FixNVTRegulated::copy_arrays(int i, int j, int /*delflag*/)
{
  c[j] = c[i];
  pscale[j] = pscale[i];
  p[j][0] = p[i][0];
  p[j][1] = p[i][1];
  p[j][2] = p[i][2];
  v_eta[j][0] = v_eta[i][0];
  v_eta[j][1] = v_eta[i][1];
  v_eta[j][2] = v_eta[i][2];
}

/* ----------------------------------------------------------------------
   pack values in local atom-based array for exchange with another proc
------------------------------------------------------------------------- */

int FixNVTRegulated::pack_exchange(int i, double *buf)
{
  int n = 0;
  buf[n++] = c[i];
  buf[n++] = pscale[i];
  buf[n++] = p[i][0];
  buf[n++] = p[i][1];
  buf[n++] = p[i][2];
  buf[n++] = v_eta[i][0];
  buf[n++] = v_eta[i][1];
  buf[n++] = v_eta[i][2];
  return n;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based array from exchange with another proc
------------------------------------------------------------------------- */

int FixNVTRegulated::unpack_exchange(int nlocal, double *buf)
{
  int n = 0;
  c[nlocal] = buf[n++];
  pscale[nlocal] = buf[n++];
  p[nlocal][0] = buf[n++];
  p[nlocal][1] = buf[n++];
  p[nlocal][2] = buf[n++];
  v_eta[nlocal][0] = buf[n++];
  v_eta[nlocal][1] = buf[n++];
  v_eta[nlocal][2] = buf[n++];
  return n;
}
