/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(nvt/regulated,FixNVTRegulated)

#else

#ifndef LMP_FIX_NVT_REGULATED_H
#define LMP_FIX_NVT_REGULATED_H

#include "fix.h"

namespace LAMMPS_NS {

class FixNVTRegulated : public Fix {
 public:
  FixNVTRegulated(class LAMMPS *, int, char **);
  virtual ~FixNVTRegulated() {}
  int setmask();
  void init();
  void setup(int);
  virtual void initial_integrate(int);
  virtual void final_integrate();
  virtual void initial_integrate_respa(int, int, int);
  virtual void final_integrate_respa(int, int);
  virtual void end_of_step();
  virtual double compute_scalar();
  virtual void reset_dt();

  void convert_velocities();

  double memory_usage();
  void grow_arrays(int);
  void copy_arrays(int, int, int);
  int pack_exchange(int, double *);
  int unpack_exchange(int, double *);

  static inline double logcosh(const double x)
  {
    double y = fabs(x);
    return y + log1p(exp(-2*y)) - log(2);
  }

 protected:
  double dtv,dtf;
  double *step_respa;

  double temp, tau, gamma;

  double n;
  int deterministic_flag;

  class RanMars *random;
  int seed;

  double *c, *pscale, **ps, **eta, **v_eta;
  double kT, Q_eta, efactor, vfactor;
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

*/
