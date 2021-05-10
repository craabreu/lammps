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

  dynamic_group_allow = 1;
  time_integrate = 1;

  temp = utils::numeric(FLERR, arg[3], false, lmp);
  double t_period = utils::numeric(FLERR, arg[4], false, lmp);
  gamma = utils::numeric(FLERR, arg[5], false, lmp);
  seed = utils::inumeric(FLERR, arg[6], false, lmp);

  if (t_period <= 0.0) error->all(FLERR,"Fix nvt/regulated characteristic time must be > 0.0");
  if (gamma <= 0.0) error->all(FLERR,"Fix nvt/regulated friction coefficient must be > 0.0");
  if (seed <= 0) error->all(FLERR,"Fix nvt/regulated seed must be > 0");

  n = 1.0;
  deterministic_flag = 1;  // replace later on by 0 (after implementing stochastic part)

  int iarg = 7;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "regulation") == 0) {
      if (iarg+2 > narg) error->all(FLERR, "Illegal fix nvt/regulated command");
      n = utils::inumeric(FLERR, arg[iarg+1], false, lmp);
      iarg += 2;
    }
    else if (strcmp(arg[iarg], "deterministic") == 0) {
      if (iarg+2 > narg) error->all(FLERR, "Illegal fix nvt/regulated command");
      if (strcmp(arg[iarg+1], "yes") == 0) deterministic_flag = 1;
      else if (strcmp(arg[iarg+1], "no") == 0) deterministic_flag = 0;
      else error->all(FLERR, "Illegal fix nvt/regulated command");
      iarg += 2;
    }
    else
      error->all(FLERR,"Illegal fix nvt/regulated command");
  }

  if (deterministic_flag)
    scalar_flag = extscalar = ecouple_flag = 1;

  // initialize Marsaglia RNG with processor-unique seed

  random = new RanMars(lmp, seed + 143*comm->me);

  // setup atom-based arrays
  // register with Atom class
  // no need to set peratom_flag, b/c data is for internal use only

  c = pscale = nullptr;
  ps = eta = v_eta = nullptr;
  grow_arrays(atom->nmax);
  atom->add_callback(Atom::GROW);

  tausq = n*t_period*t_period;
  omega = 1.0/sqrt(tausq);
  kT = force->boltz*temp/force->mvv2e;
  np1 = n + 1;
  vfactor = sqrt(np1/n);
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
  // Initialize thermostat velocities

  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      double mi = rmass ? rmass[i] : mass[type[i]];
      c[i] = sqrt(n*kT/mi);
      pscale[i] = 1.0/(mi*c[i]);
      for (int k = 0; k < 3; k++) {
        v_eta[i][k] = omega*random->gaussian();
        if (deterministic_flag) eta[i][k] = 0.0;
      }
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
  double dtfi, dtvci, vs0, vs1, vs2;
  double dts = 2.0*dtv;
  double dte = dtv*omega*omega;
  double a = exp(-dts*gamma);
  double b = omega*sqrt(1.0 - a*a);

  // update v and x of atoms in group

  double **x = atom->x;
  double **f = atom->f;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      dtfi = dtf*pscale[i];
      dtvci = dtv*c[i];

      ps[i][0] += dtfi*f[i][0];
      ps[i][1] += dtfi*f[i][1];
      ps[i][2] += dtfi*f[i][2];

      vs0 = tanh(ps[i][0]);
      vs1 = tanh(ps[i][1]);
      vs2 = tanh(ps[i][2]);

      x[i][0] += dtvci*vs0;
      x[i][1] += dtvci*vs1;
      x[i][2] += dtvci*vs2;

      v_eta[i][0] += dte*(np1*vs0*vs0 - 1.0);
      v_eta[i][1] += dte*(np1*vs1*vs1 - 1.0);
      v_eta[i][2] += dte*(np1*vs2*vs2 - 1.0);

      if (deterministic_flag) {
        eta[i][0] += dtv*(1.0 - vs0*vs0)*v_eta[i][0];
        eta[i][1] += dtv*(1.0 - vs1*vs1)*v_eta[i][1];
        eta[i][2] += dtv*(1.0 - vs2*vs2)*v_eta[i][2];

        ps[i][0] = asinh(sinh(ps[i][0])*exp(-dts*v_eta[i][0]));
        ps[i][1] = asinh(sinh(ps[i][1])*exp(-dts*v_eta[i][1]));
        ps[i][2] = asinh(sinh(ps[i][2])*exp(-dts*v_eta[i][2]));

        vs0 = tanh(ps[i][0]);
        vs1 = tanh(ps[i][1]);
        vs2 = tanh(ps[i][2]);

        eta[i][0] += dtv*(1.0 - vs0*vs0)*v_eta[i][0];
        eta[i][1] += dtv*(1.0 - vs1*vs1)*v_eta[i][1];
        eta[i][2] += dtv*(1.0 - vs2*vs2)*v_eta[i][2];
      }
      else {
        ps[i][0] = sinh(ps[i][0])*exp(-dtv*v_eta[i][0]);
        ps[i][1] = sinh(ps[i][1])*exp(-dtv*v_eta[i][1]);
        ps[i][2] = sinh(ps[i][2])*exp(-dtv*v_eta[i][2]);

        v_eta[i][0] = a*v_eta[i][0] + b*random->gaussian();
        v_eta[i][1] = a*v_eta[i][1] + b*random->gaussian();
        v_eta[i][2] = a*v_eta[i][2] + b*random->gaussian();

        ps[i][0] = asinh(ps[i][0]*exp(-dtv*v_eta[i][0]));
        ps[i][1] = asinh(ps[i][1]*exp(-dtv*v_eta[i][1]));
        ps[i][2] = asinh(ps[i][2]*exp(-dtv*v_eta[i][2]));

        vs0 = tanh(ps[i][0]);
        vs1 = tanh(ps[i][1]);
        vs2 = tanh(ps[i][2]);
      }

      v_eta[i][0] += dte*(np1*vs0*vs0 - 1.0);
      v_eta[i][1] += dte*(np1*vs1*vs1 - 1.0);
      v_eta[i][2] += dte*(np1*vs2*vs2 - 1.0);

      x[i][0] += dtvci*vs0;
      x[i][1] += dtvci*vs1;
      x[i][2] += dtvci*vs2;
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
      double dtfi = dtf*pscale[i];
      ps[i][0] += dtfi * f[i][0];
      ps[i][1] += dtfi * f[i][1];
      ps[i][2] += dtfi * f[i][2];
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
      double vfacci = vfactor*c[i];
      v[i][0] = vfacci*tanh(ps[i][0]);
      v[i][1] = vfacci*tanh(ps[i][1]);
      v[i][2] = vfacci*tanh(ps[i][2]);
    }
}

/* ---------------------------------------------------------------------- */

double FixNVTRegulated::compute_scalar()
{
  double ke_std, ke_reg, ke_eta, pe_eta, energy, total;

  double **v = atom->v;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  ke_std = ke_reg = ke_eta = pe_eta = 0.0;
  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      double mi = rmass ? rmass[i] : mass[type[i]];
      ke_std += mi*(v[i][0]*v[i][0] + v[i][1]*v[i][1] + v[i][2]*v[i][2]);
      ke_reg += logcosh(ps[i][0]) + logcosh(ps[i][1]) + logcosh(ps[i][2]);
      ke_eta += v_eta[i][0]*v_eta[i][0] + v_eta[i][1]*v_eta[i][1] + v_eta[i][2]*v_eta[i][2];
      if (deterministic_flag) pe_eta += eta[i][0] + eta[i][1] + eta[i][2];
    }

  energy = force->mvv2e*(n*kT*ke_reg - 0.5*ke_std + kT*(pe_eta + 0.5*tausq*ke_eta));
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
        ps[i][k] = pscale[i]*mi*v[i][k];
    }

  end_of_step();
}

/* ----------------------------------------------------------------------
   memory usage of tally array
------------------------------------------------------------------------- */

double FixNVTRegulated::memory_usage()
{
  double bytes = (double)atom->nmax*8*sizeof(double);
  if (deterministic_flag) bytes += atom->nmax*3*sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
   allocate atom-based array for v_eta
------------------------------------------------------------------------- */

void FixNVTRegulated::grow_arrays(int nmax)
{
  memory->grow(c, nmax, "fix_nvt_regulated:c");
  memory->grow(pscale, nmax, "fix_nvt_regulated:pscale");
  memory->grow(ps, nmax, 3, "fix_nvt_regulated:ps");
  memory->grow(v_eta, nmax, 3, "fix_nvt_regulated:v_eta");
  if (deterministic_flag)
    memory->grow(eta, nmax, 3, "fix_nvt_regulated:eta");
}

/* ----------------------------------------------------------------------
   copy values within local atom-based array
------------------------------------------------------------------------- */

void FixNVTRegulated::copy_arrays(int i, int j, int /*delflag*/)
{
  c[j] = c[i];
  pscale[j] = pscale[i];
  ps[j][0] = ps[i][0];
  ps[j][1] = ps[i][1];
  ps[j][2] = ps[i][2];
  v_eta[j][0] = v_eta[i][0];
  v_eta[j][1] = v_eta[i][1];
  v_eta[j][2] = v_eta[i][2];
  if (deterministic_flag) {
    eta[j][0] = eta[i][0];
    eta[j][1] = eta[i][1];
    eta[j][2] = eta[i][2];
  }
}

/* ----------------------------------------------------------------------
   pack values in local atom-based array for exchange with another proc
------------------------------------------------------------------------- */

int FixNVTRegulated::pack_exchange(int i, double *buf)
{
  int n = 0;
  buf[n++] = c[i];
  buf[n++] = pscale[i];
  buf[n++] = ps[i][0];
  buf[n++] = ps[i][1];
  buf[n++] = ps[i][2];
  buf[n++] = v_eta[i][0];
  buf[n++] = v_eta[i][1];
  buf[n++] = v_eta[i][2];
  if (deterministic_flag) {
    buf[n++] = eta[i][0];
    buf[n++] = eta[i][1];
    buf[n++] = eta[i][2];
  }
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
  ps[nlocal][0] = buf[n++];
  ps[nlocal][1] = buf[n++];
  ps[nlocal][2] = buf[n++];
  v_eta[nlocal][0] = buf[n++];
  v_eta[nlocal][1] = buf[n++];
  v_eta[nlocal][2] = buf[n++];
  if (deterministic_flag) {
    eta[nlocal][0] = buf[n++];
    eta[nlocal][1] = buf[n++];
    eta[nlocal][2] = buf[n++];
  }
  return n;
}
