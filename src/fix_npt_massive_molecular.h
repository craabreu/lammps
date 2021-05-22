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
FixStyle(npt/massive/molecular,FixNPTMassiveMolecular);
// clang-format on
#else

#ifndef LMP_FIX_NPT_MASSIVE_MOLECULAR_H
#define LMP_FIX_NPT_MASSIVE_MOLECULAR_H

#include "fix_nh_massive_molecular.h"

namespace LAMMPS_NS {

class FixNPTMassiveMolecular : public FixNHMassiveMolecular {
 public:
  FixNPTMassiveMolecular(class LAMMPS *, int, char **);
  ~FixNPTMassiveMolecular() {}
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Temperature control must be used with fix npt

Self-explanatory.

E: Pressure control must be used with fix npt

Self-explanatory.

*/
