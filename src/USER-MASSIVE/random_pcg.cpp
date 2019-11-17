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

// PCG Random Number Generator
// Website: http://www.pcg-random.org
// Author: M.E. O'Neill

#include "random_pcg.h"
#include <cmath>

#define AM (1.0/2147483647)

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

RanPCG::RanPCG(LAMMPS *lmp, int seed) : Pointers(lmp)
{
  state = 0u;
  i32();
  state += (uint64_t)seed;
  i32();
  save = 0;
}

/* ---------------------------------------------------------------------- */

void RanPCG::set_state(double value)
{
  memcpy(&state, &value, sizeof(double));
}

/* ---------------------------------------------------------------------- */

double RanPCG::get_state()
{
  double value;
  memcpy(&value, &state, sizeof(double));
  return value;
}

/* ----------------------------------------------------------------------
   uniform RN
------------------------------------------------------------------------- */

double RanPCG::uniform()
{
  return AM*i32();
}

/* ----------------------------------------------------------------------
   gaussian RN
------------------------------------------------------------------------- */

double RanPCG::gaussian()
{
  double first,v1,v2,rsq,fac;

  if (!save) {
    do {
      v1 = 2.0*uniform()-1.0;
      v2 = 2.0*uniform()-1.0;
      rsq = v1*v1 + v2*v2;
    } while ((rsq >= 1.0) || (rsq == 0.0));
    fac = sqrt(-2.0*log(rsq)/rsq);
    second = v1*fac;
    first = v2*fac;
    save = 1;
  } else {
    first = second;
    save = 0;
  }
  return first;
}
