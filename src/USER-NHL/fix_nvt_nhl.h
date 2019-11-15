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

FixStyle(nvt/nhl,FixNVT_NHL)

#else

#ifndef LMP_FIX_NVT_NHL_H
#define LMP_FIX_NVT_NHL_H

#include "fix_nhl.h"

namespace LAMMPS_NS {

class FixNVT_NHL : public FixNHL {
 public:
  FixNVT_NHL(class LAMMPS *, int, char **);
  ~FixNVT_NHL() {}
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Temperature control must be used with fix nvt

Self-explanatory.

E: Pressure control can not be used with fix nvt

Self-explanatory.

*/
