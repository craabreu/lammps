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

#ifndef LMP_RANPCG_H
#define LMP_RANPCG_H

#include "pointers.h"

#define inc 0xda3e39cb94b95bdbULL

namespace LAMMPS_NS {

class RanPCG : protected Pointers {
 public:
  RanPCG(class LAMMPS *, int);
  void set_state(double);
  double get_state();
  double uniform();
  double gaussian();

 private:
  uint64_t state;
  int save;
  double second;

  inline uint32_t i32() {
    uint64_t oldstate = state;
    state = oldstate * 6364136223846793005ULL + inc;
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
  };

};

}

#endif
