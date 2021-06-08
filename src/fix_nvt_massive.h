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
// clang-format off
FixStyle(nvt/massive,FixNVTMassive);
// clang-format on
#else

#ifndef LMP_FIX_NVT_MASSIVE_H
#define LMP_FIX_NVT_MASSIVE_H

#include "fix_nh_massive_molecular.h"

namespace LAMMPS_NS {

class FixNVTMassive : public FixNHMassiveMolecular {
 public:
  FixNVTMassive(class LAMMPS *, int, char **);
  ~FixNVTMassive() {}
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Temperature control must be used with fix nvt

Self-explanatory.

E: Pressure control can not be used with fix nvt

Self-explanatory.

*/
