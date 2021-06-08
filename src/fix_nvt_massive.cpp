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

#include "fix_nvt_massive.h"

#include "error.h"
#include "group.h"
#include "modify.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixNVTMassive::FixNVTMassive(LAMMPS *lmp, int narg, char **arg) :
  FixNHMassiveMolecular(lmp, narg, arg)
{
  if (!tstat_flag)
    error->all(FLERR,"Temperature control must be used with fix nvt/massive");
  if (pstat_flag)
    error->all(FLERR,"Pressure control can not be used with fix nvt/massive");

  // create a new compute temp style
  // id = fix-ID + temp

  id_temp = utils::strdup(std::string(id) + "_temp");
  if (regulation_flag)
    modify->add_compute(fmt::format("{} all temp/molecular regulation {} {}",
                                    id_temp, regulation_parameter, t_target));
  else
    modify->add_compute(fmt::format("{} all temp/molecular",id_temp));

  tcomputeflag = 1;
}
