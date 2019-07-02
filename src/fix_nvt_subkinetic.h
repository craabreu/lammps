/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(nvt/subkinetic,FixNVTSubkinetic)

#else

#ifndef LMP_FIX_NVT_SUBKINETIC_H
#define LMP_FIX_NVT_SUBKINETIC_H

#include "fix_nve.h"

namespace LAMMPS_NS {

class FixNVTSubkinetic : public FixNVE {
 public:
  FixNVTSubkinetic(class LAMMPS *, int, char **);
  virtual ~FixNVTSubkinetic();
  virtual void init();
  virtual void initial_integrate(int);
  virtual void final_integrate();
  double memory_usage();
  void grow_arrays(int);
  void copy_arrays(int, int, int);
  int pack_exchange(int, double *);
  int unpack_exchange(int, double *);

 protected:
  double kT;
  double Q_eta;
  double gamma;
  int L;
  double *v_max;
  double **v_eta;
  class RanMars *random;

  double a, b, amu, bmu;
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

W: Should not use fix nve/limit with fix shake or fix rattle

This will lead to invalid constraint forces in the SHAKE/RATTLE
computation.

*/
