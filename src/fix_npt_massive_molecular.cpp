// clang-format off
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

#include "fix_npt_massive_molecular.h"

#include "error.h"
#include "modify.h"
#include "compute.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixNPTMassiveMolecular::FixNPTMassiveMolecular(LAMMPS *lmp, int narg, char **arg) :
  FixNHMassiveMolecular(lmp, narg, arg)
{
  if (!tstat_flag)
    error->all(FLERR,"Temperature control must be used with fix npt");
  if (!pstat_flag)
    error->all(FLERR,"Pressure control must be used with fix npt");

  // create a new compute temp style
  // id = fix-ID + temp
  // compute group = all since pressure is always global (group all)
  // and thus its KE/temperature contribution should use group all

  id_temp = utils::strdup(std::string(id) + "_temp");
  modify->add_compute(fmt::format("{} all temp/molecular",id_temp));

  // char **newarg = new char*[2];
  // newarg[0] = (char *) "extra/dof";
  // newarg[1] = (char *) "0";
  // modify->compute[modify->ncompute-1]->modify_params(2, newarg);

  tcomputeflag = 1;

  // create a new compute pressure style
  // id = fix-ID + press, compute group = all
  // pass id_temp as 4th arg to pressure constructor

  id_press = utils::strdup(std::string(id) + "_press");
  modify->add_compute(fmt::format("{} all pressure/molecular {}",id_press, id_temp));
  pcomputeflag = 1;
}
