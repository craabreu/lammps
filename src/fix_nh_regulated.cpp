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

/* ======================================================================

   TO DO LIST:
     [x] Check if mtk_term1 and mtk_term2 are correct with massive tstat
     [ ] Implement massive thermostatting for barostat variables
     [ ] Check if pressure->compute_com() is unnecessary in some places
     [ ] Make it incompatible with RIGID BODIES
     [ ] Make it incompatible with TEMPERATURA BIAS
     [ ] Deactivate "drag" keyword argument
     [ ] Include massive-thermostat variables in restart

   ====================================================================== */

/* ----------------------------------------------------------------------
   Contributing authors: Mark Stevens (SNL), Aidan Thompson (SNL)
------------------------------------------------------------------------- */

#include "fix_nh_regulated.h"
#include "compute_pressure_molecular.h"
#include <cstring>
#include <cmath>
#include "atom.h"
#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "fix_deform.h"
#include "force.h"
#include "group.h"
#include "irregular.h"
#include "kspace.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"
#include "random_mars.h"
#include "respa.h"
#include "update.h"

using namespace LAMMPS_NS;
using namespace FixConst;

#define DELTAFLIP 0.1
#define TILTMAX 1.5
#define EPSILON 1.0e-6

enum{NONE,XYZ,XY,YZ,XZ};
enum{ISO,ANISO,TRICLINIC};
enum{XO_RESPA,MIDDLE_RESPA};

#define logcosh(x) (fabs(x) + log1p(exp(-2*fabs(x))) - log(2))

/* ----------------------------------------------------------------------
   NVT,NPH,NPT integrators for improved Nose-Hoover equations of motion
 ---------------------------------------------------------------------- */

FixNHRegulated::FixNHRegulated(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  rfix(nullptr), id_dilate(nullptr), irregular(nullptr),
  id_press(nullptr), eta(nullptr), v_eta(nullptr), umax(nullptr)
{
  if (narg < 4) error->all(FLERR,"Illegal fix nvt/npt/nph command");

  restart_global = 1;
  dynamic_group_allow = 1;
  time_integrate = 1;
  scalar_flag = 1;
  vector_flag = 1;
  global_freq = 1;
  extscalar = 1;
  extvector = 0;
  ecouple_flag = 1;

  // default values

  pcouple = NONE;
  drag = 0.0;
  allremap = 1;
  id_dilate = nullptr;
  nc_tchain = nc_pchain = 1;
  deviatoric_flag = 0;
  nreset_h0 = 0;
  eta_mass_flag = 1;
  omega_mass_flag = 0;
  etap_mass_flag = 0;
  flipflag = 1;
  dipole_flag = 0;
  dlm_flag = 0;
  p_temp_flag = 0;

  regulation_parameter = 1;
  respa_splitting = MIDDLE_RESPA;
  langevin_flag = 0;
  p_gamma_flag = 0;
  normalize_flag = 1;

  pcomputeflag = 0;
  id_press = nullptr;

  // turn on tilt factor scaling, whenever applicable

  dimension = domain->dimension;

  scaleyz = scalexz = scalexy = 0;
  if (domain->yperiodic && domain->xy != 0.0) scalexy = 1;
  if (domain->zperiodic && dimension == 3) {
    if (domain->yz != 0.0) scaleyz = 1;
    if (domain->xz != 0.0) scalexz = 1;
  }

  // set fixed-point to default = center of cell

  fixedpoint[0] = 0.5*(domain->boxlo[0]+domain->boxhi[0]);
  fixedpoint[1] = 0.5*(domain->boxlo[1]+domain->boxhi[1]);
  fixedpoint[2] = 0.5*(domain->boxlo[2]+domain->boxhi[2]);

  // used by FixNVTSllod to preserve non-default value

  tstat_flag = 0;
  double t_period = 0.0;

  double p_period[6];
  for (int i = 0; i < 6; i++) {
    p_start[i] = p_stop[i] = p_period[i] = p_target[i] = 0.0;
    p_flag[i] = 0;
  }

  // process keywords

  int iarg = 3;

  while (iarg < narg) {
    if (strcmp(arg[iarg],"temp") == 0) {
      if (iarg+5 > narg) error->all(FLERR,"Illegal fix nvt/npt/nph command");
      tstat_flag = 1;
      t_start = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      t_target = t_start;
      t_stop = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      t_period = utils::numeric(FLERR,arg[iarg+3],false,lmp);
      if (t_start <= 0.0 || t_stop <= 0.0)
        error->all(FLERR, "Target temperature for fix nvt/npt/nph cannot be 0.0");
      if (strcmp(arg[iarg+4],"NULL") == 0)
        langevin_flag = 0;
      else
        seed = utils::inumeric(FLERR,arg[iarg+4],false,lmp);
      iarg += 5;
    }
    else if (strcmp(arg[iarg],"iso") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix nvt/npt/nph command");
      pcouple = XYZ;
      p_start[0] = p_start[1] = p_start[2] = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      p_stop[0] = p_stop[1] = p_stop[2] = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      p_period[0] = p_period[1] = p_period[2] =
        utils::numeric(FLERR,arg[iarg+3],false,lmp);
      p_flag[0] = p_flag[1] = p_flag[2] = 1;
      if (dimension == 2) {
        p_start[2] = p_stop[2] = p_period[2] = 0.0;
        p_flag[2] = 0;
      }
      iarg += 4;
    }
    else if (strcmp(arg[iarg],"aniso") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix nvt/npt/nph command");
      pcouple = NONE;
      p_start[0] = p_start[1] = p_start[2] = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      p_stop[0] = p_stop[1] = p_stop[2] = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      p_period[0] = p_period[1] = p_period[2] =
        utils::numeric(FLERR,arg[iarg+3],false,lmp);
      p_flag[0] = p_flag[1] = p_flag[2] = 1;
      if (dimension == 2) {
        p_start[2] = p_stop[2] = p_period[2] = 0.0;
        p_flag[2] = 0;
      }
      iarg += 4;
    }
    else if (strcmp(arg[iarg],"tri") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix nvt/npt/nph command");
      pcouple = NONE;
      scalexy = scalexz = scaleyz = 0;
      p_start[0] = p_start[1] = p_start[2] = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      p_stop[0] = p_stop[1] = p_stop[2] = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      p_period[0] = p_period[1] = p_period[2] =
        utils::numeric(FLERR,arg[iarg+3],false,lmp);
      p_flag[0] = p_flag[1] = p_flag[2] = 1;
      p_start[3] = p_start[4] = p_start[5] = 0.0;
      p_stop[3] = p_stop[4] = p_stop[5] = 0.0;
      p_period[3] = p_period[4] = p_period[5] =
        utils::numeric(FLERR,arg[iarg+3],false,lmp);
      p_flag[3] = p_flag[4] = p_flag[5] = 1;
      if (dimension == 2) {
        p_start[2] = p_stop[2] = p_period[2] = 0.0;
        p_flag[2] = 0;
        p_start[3] = p_stop[3] = p_period[3] = 0.0;
        p_flag[3] = 0;
        p_start[4] = p_stop[4] = p_period[4] = 0.0;
        p_flag[4] = 0;
      }
      iarg += 4;
    }
    else if (strcmp(arg[iarg],"x") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix nvt/npt/nph command");
      p_start[0] = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      p_stop[0] = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      p_period[0] = utils::numeric(FLERR,arg[iarg+3],false,lmp);
      p_flag[0] = 1;
      deviatoric_flag = 1;
      iarg += 4;
    }
    else if (strcmp(arg[iarg],"y") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix nvt/npt/nph command");
      p_start[1] = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      p_stop[1] = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      p_period[1] = utils::numeric(FLERR,arg[iarg+3],false,lmp);
      p_flag[1] = 1;
      deviatoric_flag = 1;
      iarg += 4;
    }
    else if (strcmp(arg[iarg],"z") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix nvt/npt/nph command");
      p_start[2] = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      p_stop[2] = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      p_period[2] = utils::numeric(FLERR,arg[iarg+3],false,lmp);
      p_flag[2] = 1;
      deviatoric_flag = 1;
      iarg += 4;
      if (dimension == 2)
        error->all(FLERR,"Invalid fix nvt/npt/nph command for a 2d simulation");
    }
    else if (strcmp(arg[iarg],"yz") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix nvt/npt/nph command");
      p_start[3] = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      p_stop[3] = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      p_period[3] = utils::numeric(FLERR,arg[iarg+3],false,lmp);
      p_flag[3] = 1;
      deviatoric_flag = 1;
      scaleyz = 0;
      iarg += 4;
      if (dimension == 2)
        error->all(FLERR,"Invalid fix nvt/npt/nph command for a 2d simulation");
    }
    else if (strcmp(arg[iarg],"xz") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix nvt/npt/nph command");
      p_start[4] = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      p_stop[4] = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      p_period[4] = utils::numeric(FLERR,arg[iarg+3],false,lmp);
      p_flag[4] = 1;
      deviatoric_flag = 1;
      scalexz = 0;
      iarg += 4;
      if (dimension == 2)
        error->all(FLERR,"Invalid fix nvt/npt/nph command for a 2d simulation");
    }
    else if (strcmp(arg[iarg],"xy") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix nvt/npt/nph command");
      p_start[5] = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      p_stop[5] = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      p_period[5] = utils::numeric(FLERR,arg[iarg+3],false,lmp);
      p_flag[5] = 1;
      deviatoric_flag = 1;
      scalexy = 0;
      iarg += 4;
    }
    else if (strcmp(arg[iarg],"couple") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix nvt/npt/nph command");
      if (strcmp(arg[iarg+1],"xyz") == 0) pcouple = XYZ;
      else if (strcmp(arg[iarg+1],"xy") == 0) pcouple = XY;
      else if (strcmp(arg[iarg+1],"yz") == 0) pcouple = YZ;
      else if (strcmp(arg[iarg+1],"xz") == 0) pcouple = XZ;
      else if (strcmp(arg[iarg+1],"none") == 0) pcouple = NONE;
      else error->all(FLERR,"Illegal fix nvt/npt/nph command");
      iarg += 2;
    }
    else if (strcmp(arg[iarg],"drag") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix nvt/npt/nph command");
      drag = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      if (drag < 0.0) error->all(FLERR,"Illegal fix nvt/npt/nph command");
      iarg += 2;
    }
    else if (strcmp(arg[iarg],"ptemp") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix nvt/npt/nph command");
      p_temp = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      p_temp_flag = 1;
      if (p_temp <= 0.0) error->all(FLERR,"Illegal fix nvt/npt/nph command");
      iarg += 2;
    }
    else if (strcmp(arg[iarg],"dilate") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix nvt/npt/nph command");
      if (strcmp(arg[iarg+1],"all") == 0) allremap = 1;
      else {
        allremap = 0;
        delete [] id_dilate;
        id_dilate = utils::strdup(arg[iarg+1]);
        int idilate = group->find(id_dilate);
        if (idilate == -1)
          error->all(FLERR,"Fix nvt/npt/nph dilate group ID does not exist");
      }
      iarg += 2;
    }
    else if (strcmp(arg[iarg],"tloop") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix nvt/npt/nph command");
      nc_tchain = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      if (nc_tchain < 0) error->all(FLERR,"Illegal fix nvt/npt/nph command");
      iarg += 2;
    }
    else if (strcmp(arg[iarg],"ploop") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix nvt/npt/nph command");
      nc_pchain = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      if (nc_pchain < 0) error->all(FLERR,"Illegal fix nvt/npt/nph command");
      iarg += 2;
    }
    else if (strcmp(arg[iarg],"nreset") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix nvt/npt/nph command");
      nreset_h0 = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      if (nreset_h0 < 0) error->all(FLERR,"Illegal fix nvt/npt/nph command");
      iarg += 2;
    }
    else if (strcmp(arg[iarg],"scalexy") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix nvt/npt/nph command");
      if (strcmp(arg[iarg+1],"yes") == 0) scalexy = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) scalexy = 0;
      else error->all(FLERR,"Illegal fix nvt/npt/nph command");
      iarg += 2;
    }
    else if (strcmp(arg[iarg],"scalexz") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix nvt/npt/nph command");
      if (strcmp(arg[iarg+1],"yes") == 0) scalexz = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) scalexz = 0;
      else error->all(FLERR,"Illegal fix nvt/npt/nph command");
      iarg += 2;
    }
    else if (strcmp(arg[iarg],"scaleyz") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix nvt/npt/nph command");
      if (strcmp(arg[iarg+1],"yes") == 0) scaleyz = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) scaleyz = 0;
      else error->all(FLERR,"Illegal fix nvt/npt/nph command");
      iarg += 2;
    }
    else if (strcmp(arg[iarg],"flip") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix nvt/npt/nph command");
      if (strcmp(arg[iarg+1],"yes") == 0) flipflag = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) flipflag = 0;
      else error->all(FLERR,"Illegal fix nvt/npt/nph command");
      iarg += 2;
    }
    else if (strcmp(arg[iarg],"update") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix nvt/npt/nph command");
      if (strcmp(arg[iarg+1],"dipole") == 0) dipole_flag = 1;
      else if (strcmp(arg[iarg+1],"dipole/dlm") == 0) {
        dipole_flag = 1;
        dlm_flag = 1;
      } else error->all(FLERR,"Illegal fix nvt/npt/nph command");
      iarg += 2;
    }
    else if (strcmp(arg[iarg],"fixedpoint") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix nvt/npt/nph command");
      fixedpoint[0] = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      fixedpoint[1] = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      fixedpoint[2] = utils::numeric(FLERR,arg[iarg+3],false,lmp);
      iarg += 4;
    }
    else if (strcmp(arg[iarg],"respa_splitting") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix nvt/npt/nph command");
      if (strcmp(arg[iarg+1],"middle") == 0) respa_splitting = MIDDLE_RESPA;
      else if (strcmp(arg[iarg+1],"xo-respa") == 0) respa_splitting = XO_RESPA;
      iarg += 2;
    }
    else if (strcmp(arg[iarg],"regulation") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix nvt/npt/nph command");
      regulation_parameter = utils::numeric(FLERR, arg[iarg+1], false, lmp);
      if (regulation_parameter <= 0.0) error->all(FLERR,"Illegal fix nvt/npt/nph command");
      iarg += 2;
    }
    else if (strcmp(arg[iarg],"seed") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix nvt/npt/nph command");
      langevin_flag = 1;
      seed = utils::inumeric(FLERR, arg[iarg+2], false, lmp);
      if (t_gamma <= 0.0) error->all(FLERR,"Illegal fix nvt/npt/nph command");
      iarg += 3;
    }
    else if (strcmp(arg[iarg],"tgamma") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix nvt/npt/nph command");
      t_gamma_flag = 1;
      t_gamma = utils::numeric(FLERR, arg[iarg+1], false, lmp);
      if (p_gamma <= 0.0) error->all(FLERR,"Illegal fix nvt/npt/nph command");
      iarg += 2;
    }
    else if (strcmp(arg[iarg],"pgamma") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix nvt/npt/nph command");
      p_gamma_flag = 1;
      p_gamma = utils::numeric(FLERR, arg[iarg+1], false, lmp);
      if (p_gamma <= 0.0) error->all(FLERR,"Illegal fix nvt/npt/nph command");
      iarg += 2;
    }
    else if (strcmp(arg[iarg],"normalize") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix nvt/npt/nph command");
      if (strcmp(arg[iarg+1],"yes") == 0) normalize_flag = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) normalize_flag = 0;
      else error->all(FLERR,"Illegal fix nvt/npt/nph command");
      iarg += 2;

    // disc keyword is also parsed in fix/nh/sphere

    }
    else if (strcmp(arg[iarg],"disc") == 0) {
      iarg++;

    // keywords erate, strain, and ext are also parsed in fix/nh/uef

    }
    else if (strcmp(arg[iarg],"erate") == 0) {
      iarg += 3;
    }
    else if (strcmp(arg[iarg],"strain") == 0) {
      iarg += 3;
    }
    else if (strcmp(arg[iarg],"ext") == 0) {
      iarg += 2;

    }
    else
      error->all(FLERR,"Illegal fix nvt/npt/nph command");
  }

  if (langevin_flag) random = new RanMars(lmp, seed + 143*comm->me);

  // error checks

  if (dimension == 2 && (p_flag[2] || p_flag[3] || p_flag[4]))
    error->all(FLERR,"Invalid fix nvt/npt/nph command for a 2d simulation");
  if (dimension == 2 && (pcouple == YZ || pcouple == XZ))
    error->all(FLERR,"Invalid fix nvt/npt/nph command for a 2d simulation");
  if (dimension == 2 && (scalexz == 1 || scaleyz == 1 ))
    error->all(FLERR,"Invalid fix nvt/npt/nph command for a 2d simulation");

  if (pcouple == XYZ && (p_flag[0] == 0 || p_flag[1] == 0))
    error->all(FLERR,"Invalid fix nvt/npt/nph command pressure settings");
  if (pcouple == XYZ && dimension == 3 && p_flag[2] == 0)
    error->all(FLERR,"Invalid fix nvt/npt/nph command pressure settings");
  if (pcouple == XY && (p_flag[0] == 0 || p_flag[1] == 0))
    error->all(FLERR,"Invalid fix nvt/npt/nph command pressure settings");
  if (pcouple == YZ && (p_flag[1] == 0 || p_flag[2] == 0))
    error->all(FLERR,"Invalid fix nvt/npt/nph command pressure settings");
  if (pcouple == XZ && (p_flag[0] == 0 || p_flag[2] == 0))
    error->all(FLERR,"Invalid fix nvt/npt/nph command pressure settings");

  // require periodicity in tensile dimension

  if (p_flag[0] && domain->xperiodic == 0)
    error->all(FLERR,"Cannot use fix nvt/npt/nph on a non-periodic dimension");
  if (p_flag[1] && domain->yperiodic == 0)
    error->all(FLERR,"Cannot use fix nvt/npt/nph on a non-periodic dimension");
  if (p_flag[2] && domain->zperiodic == 0)
    error->all(FLERR,"Cannot use fix nvt/npt/nph on a non-periodic dimension");

  // require periodicity in 2nd dim of off-diagonal tilt component

  if (p_flag[3] && domain->zperiodic == 0)
    error->all(FLERR,
               "Cannot use fix nvt/npt/nph on a 2nd non-periodic dimension");
  if (p_flag[4] && domain->zperiodic == 0)
    error->all(FLERR,
               "Cannot use fix nvt/npt/nph on a 2nd non-periodic dimension");
  if (p_flag[5] && domain->yperiodic == 0)
    error->all(FLERR,
               "Cannot use fix nvt/npt/nph on a 2nd non-periodic dimension");

  if (scaleyz == 1 && domain->zperiodic == 0)
    error->all(FLERR,"Cannot use fix nvt/npt/nph "
               "with yz scaling when z is non-periodic dimension");
  if (scalexz == 1 && domain->zperiodic == 0)
    error->all(FLERR,"Cannot use fix nvt/npt/nph "
               "with xz scaling when z is non-periodic dimension");
  if (scalexy == 1 && domain->yperiodic == 0)
    error->all(FLERR,"Cannot use fix nvt/npt/nph "
               "with xy scaling when y is non-periodic dimension");

  if (p_flag[3] && scaleyz == 1)
    error->all(FLERR,"Cannot use fix nvt/npt/nph with "
               "both yz dynamics and yz scaling");
  if (p_flag[4] && scalexz == 1)
    error->all(FLERR,"Cannot use fix nvt/npt/nph with "
               "both xz dynamics and xz scaling");
  if (p_flag[5] && scalexy == 1)
    error->all(FLERR,"Cannot use fix nvt/npt/nph with "
               "both xy dynamics and xy scaling");

  if (!domain->triclinic && (p_flag[3] || p_flag[4] || p_flag[5]))
    error->all(FLERR,"Can not specify Pxy/Pxz/Pyz in "
               "fix nvt/npt/nph with non-triclinic box");

  if (pcouple == XYZ && dimension == 3 &&
      (p_start[0] != p_start[1] || p_start[0] != p_start[2] ||
       p_stop[0] != p_stop[1] || p_stop[0] != p_stop[2] ||
       p_period[0] != p_period[1] || p_period[0] != p_period[2]))
    error->all(FLERR,"Invalid fix nvt/npt/nph pressure settings");
  if (pcouple == XYZ && dimension == 2 &&
      (p_start[0] != p_start[1] || p_stop[0] != p_stop[1] ||
       p_period[0] != p_period[1]))
    error->all(FLERR,"Invalid fix nvt/npt/nph pressure settings");
  if (pcouple == XY &&
      (p_start[0] != p_start[1] || p_stop[0] != p_stop[1] ||
       p_period[0] != p_period[1]))
    error->all(FLERR,"Invalid fix nvt/npt/nph pressure settings");
  if (pcouple == YZ &&
      (p_start[1] != p_start[2] || p_stop[1] != p_stop[2] ||
       p_period[1] != p_period[2]))
    error->all(FLERR,"Invalid fix nvt/npt/nph pressure settings");
  if (pcouple == XZ &&
      (p_start[0] != p_start[2] || p_stop[0] != p_stop[2] ||
       p_period[0] != p_period[2]))
    error->all(FLERR,"Invalid fix nvt/npt/nph pressure settings");

  if (dipole_flag) {
    if (!atom->sphere_flag)
      error->all(FLERR,"Using update dipole flag requires atom style sphere");
    if (!atom->mu_flag)
      error->all(FLERR,"Using update dipole flag requires atom attribute mu");
  }

  if ((tstat_flag && t_period <= 0.0) ||
      (p_flag[0] && p_period[0] <= 0.0) ||
      (p_flag[1] && p_period[1] <= 0.0) ||
      (p_flag[2] && p_period[2] <= 0.0) ||
      (p_flag[3] && p_period[3] <= 0.0) ||
      (p_flag[4] && p_period[4] <= 0.0) ||
      (p_flag[5] && p_period[5] <= 0.0))
    error->all(FLERR,"Fix nvt/npt/nph damping parameters must be > 0.0");

  // check that ptemp is not defined with a thermostat
  if (tstat_flag && p_temp_flag)
    error->all(FLERR,"Thermostat in fix nvt/npt/nph is incompatible with ptemp command");

  // set pstat_flag and box change and restart_pbc variables

  pre_exchange_flag = 0;
  pstat_flag = 0;
  pstyle = ISO;

  for (int i = 0; i < 6; i++)
    if (p_flag[i]) pstat_flag = 1;

  if (pstat_flag) {
    if (p_flag[0]) box_change |= BOX_CHANGE_X;
    if (p_flag[1]) box_change |= BOX_CHANGE_Y;
    if (p_flag[2]) box_change |= BOX_CHANGE_Z;
    if (p_flag[3]) box_change |= BOX_CHANGE_YZ;
    if (p_flag[4]) box_change |= BOX_CHANGE_XZ;
    if (p_flag[5]) box_change |= BOX_CHANGE_XY;
    no_change_box = 1;
    if (allremap == 0) restart_pbc = 1;

    // pstyle = TRICLINIC if any off-diagonal term is controlled -> 6 dof
    // else pstyle = ISO if XYZ coupling or XY coupling in 2d -> 1 dof
    // else pstyle = ANISO -> 3 dof

    if (p_flag[3] || p_flag[4] || p_flag[5]) pstyle = TRICLINIC;
    else if (pcouple == XYZ || (dimension == 2 && pcouple == XY)) pstyle = ISO;
    else pstyle = ANISO;

    // pre_exchange only required if flips can occur due to shape changes

    if (flipflag && (p_flag[3] || p_flag[4] || p_flag[5]))
      pre_exchange_flag = pre_exchange_migrate = 1;
    if (flipflag && (domain->yz != 0.0 || domain->xz != 0.0 ||
                     domain->xy != 0.0))
      pre_exchange_flag = pre_exchange_migrate = 1;
  }

  // convert input periods to frequencies

  t_freq = 0.0;
  p_freq[0] = p_freq[1] = p_freq[2] = p_freq[3] = p_freq[4] = p_freq[5] = 0.0;

  if (tstat_flag) t_freq = 1.0 / t_period;
  if (p_flag[0]) p_freq[0] = 1.0 / p_period[0];
  if (p_flag[1]) p_freq[1] = 1.0 / p_period[1];
  if (p_flag[2]) p_freq[2] = 1.0 / p_period[2];
  if (p_flag[3]) p_freq[3] = 1.0 / p_period[3];
  if (p_flag[4]) p_freq[4] = 1.0 / p_period[4];
  if (p_flag[5]) p_freq[5] = 1.0 / p_period[5];

  // Nose/Hoover temp and pressure init

  size_vector = 0;

  if (tstat_flag) {
    grow_arrays(atom->nmax);
    atom->add_callback(Atom::GROW);
  }

  if (pstat_flag) {
    omega[0] = omega[1] = omega[2] = 0.0;
    omega_dot[0] = omega_dot[1] = omega_dot[2] = 0.0;
    omega_mass[0] = omega_mass[1] = omega_mass[2] = 0.0;
    omega[3] = omega[4] = omega[5] = 0.0;
    omega_dot[3] = omega_dot[4] = omega_dot[5] = 0.0;
    omega_mass[3] = omega_mass[4] = omega_mass[5] = 0.0;
    if (pstyle == ISO) size_vector += 2*2*1;
    else if (pstyle == ANISO) size_vector += 2*2*3;
    else if (pstyle == TRICLINIC) size_vector += 2*2*6;

    size_vector += 4;

    if (deviatoric_flag) size_vector += 1;
  }

  nrigid = 0;
  rfix = nullptr;

  if (pre_exchange_flag) irregular = new Irregular(lmp);
  else irregular = nullptr;

  // initialize vol0,t0 to zero to signal uninitialized
  // values then assigned in init(), if necessary

  vol0 = t0 = 0.0;
}

/* ---------------------------------------------------------------------- */

FixNHRegulated::~FixNHRegulated()
{
  if (copymode) return;

  delete [] id_dilate;
  delete [] rfix;

  delete irregular;

  if (tstat_flag) {
    memory->destroy(v_eta);
    if (!langevin_flag)
      memory->destroy(eta);
  }

  if (langevin_flag)
    delete [] random;

  // delete pressure if fix created them

  if (pstat_flag) {
    if (pcomputeflag)
      modify->delete_compute(id_press);
    delete [] id_press;
  }
}

/* ---------------------------------------------------------------------- */

int FixNHRegulated::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  mask |= INITIAL_INTEGRATE_RESPA;
  mask |= FINAL_INTEGRATE_RESPA;
  mask |= END_OF_STEP;
  if (pre_exchange_flag) mask |= PRE_EXCHANGE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixNHRegulated::init()
{
  // recheck that dilate group has not been deleted

  if (allremap == 0) {
    int idilate = group->find(id_dilate);
    if (idilate == -1)
      error->all(FLERR,"Fix nvt/npt/nph dilate group ID does not exist");
    dilate_group_bit = group->bitmask[idilate];
  }

  // ensure no conflict with fix deform

  if (pstat_flag)
    for (int i = 0; i < modify->nfix; i++)
      if (strcmp(modify->fix[i]->style,"deform") == 0) {
        int *dimflag = ((FixDeform *) modify->fix[i])->dimflag;
        if ((p_flag[0] && dimflag[0]) || (p_flag[1] && dimflag[1]) ||
            (p_flag[2] && dimflag[2]) || (p_flag[3] && dimflag[3]) ||
            (p_flag[4] && dimflag[4]) || (p_flag[5] && dimflag[5]))
          error->all(FLERR,"Cannot use fix npt and fix deform on "
                     "same component of stress tensor");
      }

  // set pressure ptr

  if (pstat_flag) {
    int icompute = modify->find_compute(id_press);
    if (icompute < 0)
      error->all(FLERR,"Pressure ID for fix npt/nph does not exist");
    pressure = (ComputePressureMolecular *) modify->compute[icompute];
  }

  // set timesteps and frequencies

  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
  dthalf = 0.5 * update->dt;
  dt4 = 0.25 * update->dt;
  dt8 = 0.125 * update->dt;
  dto = dthalf;

  p_freq_max = 0.0;
  if (pstat_flag) {
    p_freq_max = MAX(p_freq[0],p_freq[1]);
    p_freq_max = MAX(p_freq_max,p_freq[2]);
    if (pstyle == TRICLINIC) {
      p_freq_max = MAX(p_freq_max,p_freq[3]);
      p_freq_max = MAX(p_freq_max,p_freq[4]);
      p_freq_max = MAX(p_freq_max,p_freq[5]);
    }
    pdrag_factor = 1.0 - (update->dt * p_freq_max * drag / nc_pchain);
  }

  if (tstat_flag)
    tdrag_factor = 1.0 - (update->dt * t_freq * drag / nc_tchain);

  // set langevin parameters

  if (langevin_flag) {
    if (!t_gamma_flag) t_gamma = t_freq;
    if (pstat_flag && !p_gamma_flag) p_gamma = p_freq_max;
  }

  // tally the number of dimensions that are barostatted
  // set initial volume and reference cell, if not already done

  if (pstat_flag) {
    pdim = p_flag[0] + p_flag[1] + p_flag[2];
    if (vol0 == 0.0) {
      if (dimension == 3) vol0 = domain->xprd * domain->yprd * domain->zprd;
      else vol0 = domain->xprd * domain->yprd;
      h0_inv[0] = domain->h_inv[0];
      h0_inv[1] = domain->h_inv[1];
      h0_inv[2] = domain->h_inv[2];
      h0_inv[3] = domain->h_inv[3];
      h0_inv[4] = domain->h_inv[4];
      h0_inv[5] = domain->h_inv[5];
    }
  }

  boltz = force->boltz;
  nktv2p = force->nktv2p;
  mvv2e = force->mvv2e;

  if (force->kspace) kspace_flag = 1;
  else kspace_flag = 0;

  if (utils::strmatch(update->integrate_style,"^respa")) {
    nlevels_respa = ((Respa *) update->integrate)->nlevels;
    step_respa = ((Respa *) update->integrate)->step;
    dto = 0.5*step_respa[0];
  }

  // detect if any rigid fixes exist so rigid bodies move when box is remapped
  // rfix[] = indices to each fix rigid

  delete [] rfix;
  nrigid = 0;
  rfix = nullptr;

  for (int i = 0; i < modify->nfix; i++)
    if (modify->fix[i]->rigid_flag) nrigid++;
  if (nrigid) {
    rfix = new int[nrigid];
    nrigid = 0;
    for (int i = 0; i < modify->nfix; i++)
      if (modify->fix[i]->rigid_flag) rfix[nrigid++] = i;
  }

  if (tstat_flag) {
    double *rmass = atom->rmass;
    double *mass = atom->mass;
    int *type = atom->type;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
    if (igroup == atom->firstgroup)
      nlocal = atom->nfirst;
    double kT = boltz*t_target/mvv2e;
    double LkT = regulation_parameter*kT;
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        double massone = rmass ? rmass[i] : mass[type[i]];
        umax[i] = sqrt(LkT/massone);
        if (langevin_flag) {
          v_eta[i][0] = t_freq * random->gaussian();
          v_eta[i][1] = t_freq * random->gaussian();
          v_eta[i][2] = t_freq * random->gaussian();
        }
        else {
          eta[i][0] = eta[i][1] = eta[i][2] = 0.0;
          v_eta[i][0] = v_eta[i][1] = v_eta[i][2] = 0.0;
        }
      }

    if (langevin_flag) {
      etap_dot = p_freq_max * random->gaussian();
      MPI_Bcast(&etap_dot, 1, MPI_DOUBLE, 0, world);
    }
    else
      etap = etap_dot = 0.0;

    // scale velocities to comply with target temperature

    if (normalize_flag) {
      double **v = atom->v;
      double factor = 1.0;
      double stored = 0.0;
      while (fabs(factor - stored) > 1E-6) {
        stored = factor;
        double sum_vu = 0.0;
        for (int i = 0; i < nlocal; i++) {
          double mass_umax = (rmass ? rmass[i] : mass[type[i]])*umax[i];
          for (int j = 0; j < 3; j++)
            sum_vu += mass_umax*v[i][j]*tanh(factor*v[i][j]/umax[i]);
        }
        factor = 3*nlocal*kT/sum_vu;
      }
      for (int i = 0; i < nlocal; i++)
        for (int j = 0; j < 3; j++)
          v[i][j] *= factor;
    }
  }
}

/* ----------------------------------------------------------------------
   compute T,P before integrator starts
------------------------------------------------------------------------- */

void FixNHRegulated::setup(int /*vflag*/)
{
  // t_target is needed by NVT and NPT in compute_scalar()
  // If no thermostat or using fix nphug,
  // t_target must be defined by other means.

  if (tstat_flag && strstr(style,"nphug") == nullptr) {
    compute_temp_target();
  } else if (pstat_flag) {

    // t0 = reference temperature for masses
    // set equal to either ptemp or the current temperature
    // cannot be done in init() b/c temperature cannot be called there
    // is b/c Modify::init() inits computes after fixes due to dof dependence
    // error if T less than 1e-6
    // if it was read in from a restart file, leave it be

    if (t0 == 0.0) {
      if (p_temp_flag) {
        t0 = p_temp;
      } else {
        pressure->compute_scalar();
        t0 = pressure->temp;
        if (t0 < EPSILON)
          error->all(FLERR,"Current temperature too close to zero, "
                     "consider using ptemp setting");
      }
    }
    t_target = t0;
  }

  if (pstat_flag) compute_press_target();

  if (pstat_flag) {
    if (pstyle == ISO)
      pressure->compute_scalar();
    else
      pressure->compute_vector();
    couple();
    pressure->addstep(update->ntimestep+1);
  }

  // masses and initial forces on thermostat variables

  if (tstat_flag)
    Q_eta = boltz * t_target / (t_freq*t_freq);

  // masses and initial forces on barostat variables

  if (pstat_flag) {
    double kt = boltz * t_target;
    double nkt = (atom->natoms + 1) * kt;  // Should it be changed to Nmol?

    for (int i = 0; i < 3; i++)
      if (p_flag[i])
        omega_mass[i] = nkt/(p_freq[i]*p_freq[i]);

    if (pstyle == TRICLINIC) {
      for (int i = 3; i < 6; i++)
        if (p_flag[i]) omega_mass[i] = nkt/(p_freq[i]*p_freq[i]);
    }

    // mass of barostat thermostat variable

    etap_mass = boltz * t_target / (p_freq_max*p_freq_max);
  }
}

/* ----------------------------------------------------------------------
   1st half of Verlet update
------------------------------------------------------------------------- */

void FixNHRegulated::initial_integrate(int /*vflag*/)
{
  // update eta_press_dot

  if (pstat_flag) nhc_press_integrate();

  if (tstat_flag && respa_splitting == XO_RESPA) {
    compute_temp_target();
    nhc_temp_integrate(dthalf);
  }

  if (pstat_flag) {
    if (pstyle == ISO)
      pressure->compute_scalar();
    else
      pressure->compute_vector();
    couple();
    pressure->addstep(update->ntimestep+1);
  }

  if (pstat_flag) {
    compute_press_target();
    nh_omega_dot();
  }

  nve_v();

  // remap simulation box by 1/2 step

  if (pstat_flag) change_box();

  if (tstat_flag && respa_splitting == MIDDLE_RESPA) {
    nve_x(dthalf);
    compute_temp_target();
    nhc_temp_integrate(dtv);
    nve_x(dthalf);
  }
  else
    nve_x(dtv);

  // remap simulation box by 1/2 step
  // redo KSpace coeffs since volume has changed

  if (pstat_flag) {
    change_box();
    if (kspace_flag) force->kspace->setup();
  }
}

/* ----------------------------------------------------------------------
   2nd half of Verlet update
------------------------------------------------------------------------- */

void FixNHRegulated::final_integrate()
{
  nve_v();
}

/* ---------------------------------------------------------------------- */

void FixNHRegulated::initial_integrate_respa(int /*vflag*/, int ilevel, int /*iloop*/)
{
  // set timesteps by level

  dtv = step_respa[ilevel];
  dtf = 0.5 * step_respa[ilevel] * force->ftm2v;
  dthalf = 0.5 * step_respa[ilevel];

  // outermost level - update omega_dot, apply to v
  // all other levels - NVE update of v
  // x,v updates only performed for atoms in group

  if (ilevel == nlevels_respa-1) {

    // update eta_press_dot

    if (pstat_flag) nhc_press_integrate();

    if (tstat_flag && respa_splitting == XO_RESPA) {
      compute_temp_target();
      nhc_temp_integrate(dthalf);
    }

    // recompute pressure to account for change in KE
    // t_current is up-to-date, but compute_temperature is not
    // compute appropriately coupled elements of mvv_current

    if (pstat_flag) {
      if (pstyle == ISO)
        pressure->compute_scalar();
      else
        pressure->compute_vector();
      couple();
      pressure->addstep(update->ntimestep+1);
    }

    if (pstat_flag) {
      compute_press_target();
      nh_omega_dot();
    }

    nve_v();

  } else nve_v();

  // innermost level - also update x only for atoms in group
  // if barostat, perform 1/2 step remap before and after

  if (ilevel == 0) {
    if (pstat_flag) change_box();
    if (tstat_flag && respa_splitting == MIDDLE_RESPA) {
      nve_x(dthalf);
      compute_temp_target();
      nhc_temp_integrate(dtv);
      nve_x(dthalf);
    }
    else
      nve_x(dtv);
    if (pstat_flag) change_box();
  }

  // if barostat, redo KSpace coeffs at outermost level,
  // since volume has changed

  if (ilevel == nlevels_respa-1 && kspace_flag && pstat_flag)
    force->kspace->setup();
}

/* ---------------------------------------------------------------------- */

void FixNHRegulated::final_integrate_respa(int ilevel, int /*iloop*/)
{
  // set timesteps by level

  dtf = 0.5 * step_respa[ilevel] * force->ftm2v;
  dthalf = 0.5 * step_respa[ilevel];

  nve_v();
}

/* ---------------------------------------------------------------------- */

void FixNHRegulated::end_of_step()
{
  if (pstat_flag) {
    if (pstyle == ISO)
      pressure->compute_scalar();
    else
      pressure->compute_vector();
    couple();
    pressure->addstep(update->ntimestep+1);
  }

  if (pstat_flag) nh_omega_dot();

  // update eta_press_dot

  if (tstat_flag && respa_splitting == XO_RESPA)
    nhc_temp_integrate(dthalf);

  if (pstat_flag) nhc_press_integrate();
}

/* ---------------------------------------------------------------------- */

void FixNHRegulated::couple()
{
  double *tensor = pressure->vector;

  if (pstyle == ISO)
    p_current[0] = p_current[1] = p_current[2] = pressure->scalar;
  else if (pcouple == XYZ) {
    double ave = 1.0/3.0 * (tensor[0] + tensor[1] + tensor[2]);
    p_current[0] = p_current[1] = p_current[2] = ave;
  } else if (pcouple == XY) {
    double ave = 0.5 * (tensor[0] + tensor[1]);
    p_current[0] = p_current[1] = ave;
    p_current[2] = tensor[2];
  } else if (pcouple == YZ) {
    double ave = 0.5 * (tensor[1] + tensor[2]);
    p_current[1] = p_current[2] = ave;
    p_current[0] = tensor[0];
  } else if (pcouple == XZ) {
    double ave = 0.5 * (tensor[0] + tensor[2]);
    p_current[0] = p_current[2] = ave;
    p_current[1] = tensor[1];
  } else {
    p_current[0] = tensor[0];
    p_current[1] = tensor[1];
    p_current[2] = tensor[2];
  }

  if (!std::isfinite(p_current[0]) || !std::isfinite(p_current[1]) || !std::isfinite(p_current[2]))
    error->all(FLERR,"Non-numeric pressure - simulation unstable");

  // switch order from xy-xz-yz to Voigt ordering

  if (pstyle == TRICLINIC) {
    p_current[3] = tensor[5];
    p_current[4] = tensor[4];
    p_current[5] = tensor[3];

    if (!std::isfinite(p_current[3]) || !std::isfinite(p_current[4]) || !std::isfinite(p_current[5]))
      error->all(FLERR,"Non-numeric pressure - simulation unstable");
  }
}

/* ----------------------------------------------------------------------
   change box size
   scale all center-of-mass positions and velocities
   translate atom positions and velocities accordingly
------------------------------------------------------------------------- */

void FixNHRegulated::change_box()
{
  int i, imol;
  double oldlo,oldhi;
  double expfac;

  double **x = atom->x;
  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  double *h = domain->h;

  // omega is not used, except for book-keeping

  for (i = 0; i < 6; i++)
    omega[i] += dto*omega_dot[i];

  pressure->compute_com();

  double **rcm = pressure->rcm;
  double **vcm = pressure->vcm;
  tagint *molindex = atom->molecule;
  int nmol = pressure->nmolecules;

  for (i = 0; i < nlocal; i++) {
    imol = molindex[i] - 1;
    x[i][0] -= rcm[imol][0];
    x[i][1] -= rcm[imol][1];
    x[i][2] -= rcm[imol][2];
    v[i][0] -= vcm[imol][0];
    v[i][1] -= vcm[imol][1];
    v[i][2] -= vcm[imol][2];
  }

  for (imol = 0; imol < nmol; imol++)
    domain->x2lamda(rcm[imol], rcm[imol]);

  // reset global and local box to new size/shape

  // this operation corresponds to applying the
  // translate and scale operations
  // corresponding to the solution of the following ODE:
  //
  // h_dot = omega_dot * h
  //
  // where h_dot, omega_dot and h are all upper-triangular
  // 3x3 tensors. In Voigt ordering, the elements of the
  // RHS product tensor are:
  // h_dot = [0*0, 1*1, 2*2, 1*3+3*2, 0*4+5*3+4*2, 0*5+5*1]
  //
  // Ordering of operations preserves time symmetry.

  double dto2 = dto/2.0;
  double dto4 = dto/4.0;
  double dto8 = dto/8.0;

  // off-diagonal components, first half

  if (pstyle == TRICLINIC) {

    if (p_flag[4]) {
      expfac = exp(dto8*omega_dot[0]);
      h[4] *= expfac;
      h[4] += dto4*(omega_dot[5]*h[3]+omega_dot[4]*h[2]);
      h[4] *= expfac;
    }

    if (p_flag[3]) {
      expfac = exp(dto4*omega_dot[1]);
      h[3] *= expfac;
      h[3] += dto2*(omega_dot[3]*h[2]);
      h[3] *= expfac;
    }

    if (p_flag[5]) {
      expfac = exp(dto4*omega_dot[0]);
      h[5] *= expfac;
      h[5] += dto2*(omega_dot[5]*h[1]);
      h[5] *= expfac;
    }

    if (p_flag[4]) {
      expfac = exp(dto8*omega_dot[0]);
      h[4] *= expfac;
      h[4] += dto4*(omega_dot[5]*h[3]+omega_dot[4]*h[2]);
      h[4] *= expfac;
    }
  }

  // scale diagonal components
  // scale tilt factors with cell, if set

  if (p_flag[0]) {
    oldlo = domain->boxlo[0];
    oldhi = domain->boxhi[0];
    expfac = exp(dto*omega_dot[0]);
    domain->boxlo[0] = (oldlo-fixedpoint[0])*expfac + fixedpoint[0];
    domain->boxhi[0] = (oldhi-fixedpoint[0])*expfac + fixedpoint[0];
  }

  if (p_flag[1]) {
    oldlo = domain->boxlo[1];
    oldhi = domain->boxhi[1];
    expfac = exp(dto*omega_dot[1]);
    domain->boxlo[1] = (oldlo-fixedpoint[1])*expfac + fixedpoint[1];
    domain->boxhi[1] = (oldhi-fixedpoint[1])*expfac + fixedpoint[1];
    if (scalexy) h[5] *= expfac;
  }

  if (p_flag[2]) {
    oldlo = domain->boxlo[2];
    oldhi = domain->boxhi[2];
    expfac = exp(dto*omega_dot[2]);
    domain->boxlo[2] = (oldlo-fixedpoint[2])*expfac + fixedpoint[2];
    domain->boxhi[2] = (oldhi-fixedpoint[2])*expfac + fixedpoint[2];
    if (scalexz) h[4] *= expfac;
    if (scaleyz) h[3] *= expfac;
  }

  // off-diagonal components, second half

  if (pstyle == TRICLINIC) {

    if (p_flag[4]) {
      expfac = exp(dto8*omega_dot[0]);
      h[4] *= expfac;
      h[4] += dto4*(omega_dot[5]*h[3]+omega_dot[4]*h[2]);
      h[4] *= expfac;
    }

    if (p_flag[3]) {
      expfac = exp(dto4*omega_dot[1]);
      h[3] *= expfac;
      h[3] += dto2*(omega_dot[3]*h[2]);
      h[3] *= expfac;
    }

    if (p_flag[5]) {
      expfac = exp(dto4*omega_dot[0]);
      h[5] *= expfac;
      h[5] += dto2*(omega_dot[5]*h[1]);
      h[5] *= expfac;
    }

    if (p_flag[4]) {
      expfac = exp(dto8*omega_dot[0]);
      h[4] *= expfac;
      h[4] += dto4*(omega_dot[5]*h[3]+omega_dot[4]*h[2]);
      h[4] *= expfac;
    }

  }

  domain->yz = h[3];
  domain->xz = h[4];
  domain->xy = h[5];

  // tilt factor to cell length ratio can not exceed TILTMAX in one step

  if (domain->yz < -TILTMAX*domain->yprd ||
      domain->yz > TILTMAX*domain->yprd ||
      domain->xz < -TILTMAX*domain->xprd ||
      domain->xz > TILTMAX*domain->xprd ||
      domain->xy < -TILTMAX*domain->xprd ||
      domain->xy > TILTMAX*domain->xprd)
    error->all(FLERR,"Fix npt/nph has tilted box too far in one step - "
               "periodic cell is too far from equilibrium state");

  domain->set_global_box();
  domain->set_local_box();

  for (imol = 0; imol < nmol; imol++)
    domain->lamda2x(rcm[imol], rcm[imol]);

  // Scale center-of-mass velocities

  double mtk_term2 = 0.0;
  for (int i = 0; i < 3; i++)
    if (p_flag[i])
      mtk_term2 += omega_dot[i];
  if (pdim > 0) mtk_term2 /= pdim * pressure->nmolecules;

  double factor[3];
  factor[0] = exp(-dt4*(omega_dot[0]+mtk_term2));
  factor[1] = exp(-dt4*(omega_dot[1]+mtk_term2));
  factor[2] = exp(-dt4*(omega_dot[2]+mtk_term2));

  for (imol = 0; imol < nmol; imol++) {
    vcm[imol][0] *= factor[0];
    vcm[imol][1] *= factor[1];
    vcm[imol][2] *= factor[2];
    if (pstyle == TRICLINIC) {
      vcm[imol][0] += -dthalf*(vcm[imol][1]*omega_dot[5] + vcm[imol][2]*omega_dot[4]);
      vcm[imol][1] += -dthalf*vcm[imol][2]*omega_dot[3];
    }
    vcm[imol][0] *= factor[0];
    vcm[imol][1] *= factor[1];
    vcm[imol][2] *= factor[2];
  }

  for (i = 0; i < nlocal; i++) {
    imol = molindex[i] - 1;
    x[i][0] += rcm[imol][0];
    x[i][1] += rcm[imol][1];
    x[i][2] += rcm[imol][2];
    v[i][0] += vcm[imol][0];
    v[i][1] += vcm[imol][1];
    v[i][2] += vcm[imol][2];
  }
}

/* ----------------------------------------------------------------------
   pack entire state of Fix into one write
------------------------------------------------------------------------- */

void FixNHRegulated::write_restart(FILE *fp)
{
  int nsize = size_restart_global();

  double *list;
  memory->create(list,nsize,"nh:list");

  pack_restart_data(list);

  if (comm->me == 0) {
    int size = nsize * sizeof(double);
    fwrite(&size,sizeof(int),1,fp);
    fwrite(list,sizeof(double),nsize,fp);
  }

  memory->destroy(list);
}

/* ----------------------------------------------------------------------
    calculate the number of data to be packed
------------------------------------------------------------------------- */

int FixNHRegulated::size_restart_global()
{
  int nsize = 2;
  if (tstat_flag) {
    // ADD THERMOSTAT VARIABLES HERE
  }
  if (pstat_flag) {
    nsize += 16 + 2;
    if (deviatoric_flag) nsize += 6;
  }

  return nsize;
}

/* ----------------------------------------------------------------------
   pack restart data
------------------------------------------------------------------------- */

int FixNHRegulated::pack_restart_data(double *list)
{
  int n = 0;

  list[n++] = tstat_flag;
  if (tstat_flag) {
    // ADD THERMOSTAT VARIABLES HERE
  }

  list[n++] = pstat_flag;
  if (pstat_flag) {
    list[n++] = omega[0];
    list[n++] = omega[1];
    list[n++] = omega[2];
    list[n++] = omega[3];
    list[n++] = omega[4];
    list[n++] = omega[5];
    list[n++] = omega_dot[0];
    list[n++] = omega_dot[1];
    list[n++] = omega_dot[2];
    list[n++] = omega_dot[3];
    list[n++] = omega_dot[4];
    list[n++] = omega_dot[5];
    list[n++] = vol0;
    list[n++] = t0;
    list[n++] = etap;
    list[n++] = etap_dot;

    list[n++] = deviatoric_flag;
    if (deviatoric_flag) {
      list[n++] = h0_inv[0];
      list[n++] = h0_inv[1];
      list[n++] = h0_inv[2];
      list[n++] = h0_inv[3];
      list[n++] = h0_inv[4];
      list[n++] = h0_inv[5];
    }
  }

  return n;
}

/* ----------------------------------------------------------------------
   use state info from restart file to restart the Fix
------------------------------------------------------------------------- */

void FixNHRegulated::restart(char *buf)
{
  int n = 0;
  double *list = (double *) buf;
  int flag = static_cast<int> (list[n++]);
  if (flag) {
    // ADD THERMOSTAT VARIABLES HERE
  }
  flag = static_cast<int> (list[n++]);
  if (flag) {
    omega[0] = list[n++];
    omega[1] = list[n++];
    omega[2] = list[n++];
    omega[3] = list[n++];
    omega[4] = list[n++];
    omega[5] = list[n++];
    omega_dot[0] = list[n++];
    omega_dot[1] = list[n++];
    omega_dot[2] = list[n++];
    omega_dot[3] = list[n++];
    omega_dot[4] = list[n++];
    omega_dot[5] = list[n++];
    vol0 = list[n++];
    t0 = list[n++];
    etap = list[n++];
    etap_dot = list[n++];
    flag = static_cast<int> (list[n++]);
    if (flag) {
      h0_inv[0] = list[n++];
      h0_inv[1] = list[n++];
      h0_inv[2] = list[n++];
      h0_inv[3] = list[n++];
      h0_inv[4] = list[n++];
      h0_inv[5] = list[n++];
    }
  }
}

/* ---------------------------------------------------------------------- */

int FixNHRegulated::modify_param(int narg, char **arg)
{
  if (strcmp(arg[0],"press") == 0) {
    if (narg < 2) error->all(FLERR,"Illegal fix_modify command");
    if (!pstat_flag) error->all(FLERR,"Illegal fix_modify command");
    if (pcomputeflag) {
      modify->delete_compute(id_press);
      pcomputeflag = 0;
    }
    delete [] id_press;
    id_press = utils::strdup(arg[1]);

    int icompute = modify->find_compute(arg[1]);
    if (icompute < 0) error->all(FLERR,"Could not find fix_modify pressure ID");
    pressure = (ComputePressureMolecular *) modify->compute[icompute];

    if (pressure->pressflag == 0)
      error->all(FLERR,"Fix_modify pressure ID does not compute pressure");
    return 2;
  }

  return 0;
}

/* ---------------------------------------------------------------------- */

double FixNHRegulated::compute_scalar()
{
  int i;
  double volume;
  double energy, eproc;
  double kt = boltz*t_target;
  double lkt_press = 0.0;

  if (dimension == 3)
    volume = domain->xprd * domain->yprd * domain->zprd;
  else
    volume = domain->xprd * domain->yprd;

  // add regulation kinetic energy and subtract conventional one

  double **v = atom->v;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup)
    nlocal = atom->nfirst;
  double ke = 0.0;
  double regke = 0.0;
  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      double imass = rmass ? rmass[i] : mass[type[i]];
      regke += logcosh(v[i][0]/umax[i]) + logcosh(v[i][1]/umax[i]) + logcosh(v[i][2]/umax[i]);
      ke += imass*(v[i][0]*v[i][0] + v[i][1]*v[i][1] + v[i][2]*v[i][2]);
    }
  eproc = regulation_parameter*kt*regke - 0.5*mvv2e*ke;

  // thermostat energy is a massive-thermostatting version of
  // Eq. (2) in Martyna, Tuckerman, Tobias, Klein, Mol Phys, 87, 1117

  if (tstat_flag) {
    double sum_vsq = 0.0;
    double sum_eta = 0.0;
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        sum_vsq += v_eta[i][0]*v_eta[i][0] + v_eta[i][1]*v_eta[i][1] + v_eta[i][2]*v_eta[i][2];
        if (!langevin_flag)
          sum_eta += eta[i][0] + eta[i][1] + eta[i][2];
      }
    eproc += kt*sum_eta + 0.5*Q_eta*sum_vsq;
  }

  MPI_Allreduce(&eproc, &energy, 1, MPI_DOUBLE, MPI_SUM, world);

  // barostat energy is equivalent to Eq. (8) in
  // Martyna, Tuckerman, Tobias, Klein, Mol Phys, 87, 1117
  // Sum(0.5*p_omega^2/W + P*V),
  // where N = natoms
  //       p_omega = W*omega_dot
  //       W = N*k*T/p_freq^2
  //       sum is over barostatted dimensions

  if (pstat_flag) {
    for (i = 0; i < 3; i++) {
      if (p_flag[i]) {
        energy += 0.5*omega_dot[i]*omega_dot[i]*omega_mass[i] +
          p_hydro*(volume-vol0) / (pdim*nktv2p);
        lkt_press += kt;
      }
    }

    if (pstyle == TRICLINIC) {
      for (i = 3; i < 6; i++) {
        if (p_flag[i]) {
          energy += 0.5*omega_dot[i]*omega_dot[i]*omega_mass[i];
          lkt_press += kt;
        }
      }
    }

    // extra contributions from thermostat chain for barostat

    energy += lkt_press * etap + 0.5*etap_mass*etap_dot*etap_dot;

    // extra contribution from strain energy

    if (deviatoric_flag) energy += compute_strain_energy();
  }

  return energy;
}

/* ----------------------------------------------------------------------
   return a single element of the following vectors, in this order:
      omega[ndof], omega_dot[ndof]
      etap, etap_dot, PE_eta, KE_eta_dot
      PE_omega[ndof], KE_omega_dot[ndof], PE_etap[, KE_etap_dot
      PE_strain[1]
  if no thermostat exists, related quantities are omitted from the list
  if no barostat exists, related quantities are omitted from the list
  ndof = 1,3,6 degrees of freedom for pstyle = ISO,ANISO,TRI
------------------------------------------------------------------------- */

double FixNHRegulated::compute_vector(int n)
{
  int ilen;

  if (tstat_flag) {
    // ADD THERMOSTAT VARIABLES HERE???
  }

  if (pstat_flag) {
    if (pstyle == ISO) {
      ilen = 1;
      if (n < ilen) return omega[n];
      n -= ilen;
    } else if (pstyle == ANISO) {
      ilen = 3;
      if (n < ilen) return omega[n];
      n -= ilen;
    } else {
      ilen = 6;
      if (n < ilen) return omega[n];
      n -= ilen;
    }

    if (pstyle == ISO) {
      ilen = 1;
      if (n < ilen) return omega_dot[n];
      n -= ilen;
    } else if (pstyle == ANISO) {
      ilen = 3;
      if (n < ilen) return omega_dot[n];
      n -= ilen;
    } else {
      ilen = 6;
      if (n < ilen) return omega_dot[n];
      n -= ilen;
    }

    if (n == 0) return etap;
    n--;
    if (n == 0) return etap_dot;
    n--;
  }

  double volume;
  double kt = boltz * t_target;
  double lkt_press = kt;
  int ich;
  if (dimension == 3) volume = domain->xprd * domain->yprd * domain->zprd;
  else volume = domain->xprd * domain->yprd;

  if (tstat_flag) {
    // ADD THERMOSTAT VARIABLES HERE???
  }

  if (pstat_flag) {
    if (pstyle == ISO) {
      ilen = 1;
      if (n < ilen)
        return p_hydro*(volume-vol0) / nktv2p;
      n -= ilen;
    } else if (pstyle == ANISO) {
      ilen = 3;
      if (n < ilen) {
        if (p_flag[n])
          return p_hydro*(volume-vol0) / (pdim*nktv2p);
        else
          return 0.0;
      }
      n -= ilen;
    } else {
      ilen = 6;
      if (n < ilen) {
        if (n > 2) return 0.0;
        else if (p_flag[n])
          return p_hydro*(volume-vol0) / (pdim*nktv2p);
        else
          return 0.0;
      }
      n -= ilen;
    }

    if (pstyle == ISO) {
      ilen = 1;
      if (n < ilen)
        return pdim*0.5*omega_dot[n]*omega_dot[n]*omega_mass[n];
      n -= ilen;
    } else if (pstyle == ANISO) {
      ilen = 3;
      if (n < ilen) {
        if (p_flag[n])
          return 0.5*omega_dot[n]*omega_dot[n]*omega_mass[n];
        else return 0.0;
      }
      n -= ilen;
    } else {
      ilen = 6;
      if (n < ilen) {
        if (p_flag[n])
          return 0.5*omega_dot[n]*omega_dot[n]*omega_mass[n];
        else return 0.0;
      }
      n -= ilen;
    }

    if (n == 0) return lkt_press * etap;
    n--;
    if (n == 0) return 0.5*etap_mass*etap_dot*etap_dot;
    n--;

    if (deviatoric_flag) {
      if (n == 0) return compute_strain_energy();
      n--;
    }
  }

  return 0.0;
}

/* ---------------------------------------------------------------------- */

void FixNHRegulated::reset_target(double t_new)
{
  t_target = t_start = t_stop = t_new;
}

/* ---------------------------------------------------------------------- */

void FixNHRegulated::reset_dt()
{
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
  dthalf = 0.5 * update->dt;
  dt4 = 0.25 * update->dt;
  dt8 = 0.125 * update->dt;
  dto = dthalf;

  // If using respa, then remap is performed in innermost level

  if (utils::strmatch(update->integrate_style,"^respa"))
    dto = 0.5*step_respa[0];

  if (pstat_flag)
    pdrag_factor = 1.0 - (update->dt * p_freq_max * drag / nc_pchain);

  if (tstat_flag)
    tdrag_factor = 1.0 - (update->dt * t_freq * drag / nc_tchain);
}

/* ----------------------------------------------------------------------
   extract thermostat properties
------------------------------------------------------------------------- */

void *FixNHRegulated::extract(const char *str, int &dim)
{
  dim=0;
  if (tstat_flag && strcmp(str,"t_target") == 0) {
    return &t_target;
  } else if (tstat_flag && strcmp(str,"t_start") == 0) {
    return &t_start;
  } else if (tstat_flag && strcmp(str,"t_stop") == 0) {
    return &t_stop;
  }
  dim=1;
  if (pstat_flag && strcmp(str,"etap") == 0) {
    return &etap;
  } else if (pstat_flag && strcmp(str,"p_flag") == 0) {
    return &p_flag;
  } else if (pstat_flag && strcmp(str,"p_start") == 0) {
    return &p_start;
  } else if (pstat_flag && strcmp(str,"p_stop") == 0) {
    return &p_stop;
  } else if (pstat_flag && strcmp(str,"p_target") == 0) {
    return &p_target;
  }
  return nullptr;
}

/* ----------------------------------------------------------------------
   perform half-step update of chain thermostat variables
------------------------------------------------------------------------- */

void FixNHRegulated::nhc_temp_integrate(double dt)
{
  double **v = atom->v;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup)
    nlocal = atom->nfirst;

  // Update mass to preserve initial freq, if flag set

  double kT = boltz*t_target;
  if (eta_mass_flag) {
    Q_eta = kT/(t_freq*t_freq);
    if (!t_gamma_flag) t_gamma = t_freq;
  }

  double ldt = dt/nc_tchain;
  double ldt2 = 0.5*ldt;
  double ldt2Q = ldt2/Q_eta;

  double a = exp(-ldt*t_gamma);
  double b = sqrt((1.0 - a*a)*kT/Q_eta);

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      double umax_inv = 1.0/umax[i];
      double massone = rmass ? rmass[i] : mass[type[i]];
      double mass_umax = mvv2e*massone*umax[i];
      for (int iloop = 0; iloop < nc_tchain; iloop++) {
        v_eta[i][0] += (mass_umax*v[i][0]*tanh(v[i][0]*umax_inv) - kT)*ldt2Q;
        v_eta[i][1] += (mass_umax*v[i][1]*tanh(v[i][1]*umax_inv) - kT)*ldt2Q;
        v_eta[i][2] += (mass_umax*v[i][2]*tanh(v[i][2]*umax_inv) - kT)*ldt2Q;

        if (langevin_flag) {
          v[i][0] *= exp(-ldt2*v_eta[i][0]);
          v[i][1] *= exp(-ldt2*v_eta[i][1]);
          v[i][2] *= exp(-ldt2*v_eta[i][2]);

          v_eta[i][0] = a*v_eta[i][0] + b*random->gaussian();
          v_eta[i][1] = a*v_eta[i][1] + b*random->gaussian();
          v_eta[i][2] = a*v_eta[i][2] + b*random->gaussian();

          v[i][0] *= exp(-ldt2*v_eta[i][0]);
          v[i][1] *= exp(-ldt2*v_eta[i][1]);
          v[i][2] *= exp(-ldt2*v_eta[i][2]);
        }
        else {
          v[i][0] *= exp(-ldt*v_eta[i][0]);
          v[i][1] *= exp(-ldt*v_eta[i][1]);
          v[i][2] *= exp(-ldt*v_eta[i][2]);

          eta[i][0] += ldt*v_eta[i][0];
          eta[i][1] += ldt*v_eta[i][1];
          eta[i][2] += ldt*v_eta[i][2];
        }

        v_eta[i][0] += (mass_umax*v[i][0]*tanh(v[i][0]*umax_inv) - kT)*ldt2Q;
        v_eta[i][1] += (mass_umax*v[i][1]*tanh(v[i][1]*umax_inv) - kT)*ldt2Q;
        v_eta[i][2] += (mass_umax*v[i][2]*tanh(v[i][2]*umax_inv) - kT)*ldt2Q;
      }
    }
}

/* ----------------------------------------------------------------------
   perform half-step update of chain thermostat variables for barostat
   scale barostat velocities
------------------------------------------------------------------------- */

void FixNHRegulated::nhc_press_integrate()
{
  int i, pdof;
  double expfac, factor_etap, kecurrent;
  double kt = boltz*t_target;
  double lkt_press;

  // Update masses, to preserve initial freq, if flag set

  if (omega_mass_flag) {
    double nkt = (atom->natoms + 1)*kt;  // Change to nmolecules?
    for (i = 0; i < 3; i++)
      if (p_flag[i])
        omega_mass[i] = nkt/(p_freq[i]*p_freq[i]);

    if (pstyle == TRICLINIC)
      for (i = 3; i < 6; i++)
        if (p_flag[i])
          omega_mass[i] = nkt/(p_freq[i]*p_freq[i]);
  }

  if (etap_mass_flag) {
    etap_mass = kt/(p_freq_max*p_freq_max);
    if (!p_gamma_flag) p_gamma = p_freq_max;
  }

  kecurrent = 0.0;
  pdof = 0;
  for (i = 0; i < 3; i++)
    if (p_flag[i]) {
      kecurrent += omega_mass[i]*omega_dot[i]*omega_dot[i];
      pdof++;
    }

  if (pstyle == TRICLINIC)
    for (i = 3; i < 6; i++)
      if (p_flag[i]) {
        kecurrent += omega_mass[i]*omega_dot[i]*omega_dot[i];
        pdof++;
      }

  if (pstyle == ISO)
    lkt_press = kt;
  else
    lkt_press = pdof*kt;

  double ncfac = 1.0/nc_pchain;

  double a = exp(-ncfac*dthalf*p_gamma);
  double b = sqrt((1.0 - a*a)*kt/etap_mass);

  for (int iloop = 0; iloop < nc_pchain; iloop++) {
    etap_dot += ncfac*dt4*(kecurrent - lkt_press)/etap_mass;

    if (langevin_flag) {
      factor_etap = exp(-ncfac*dt4*etap_dot);
      for (i = 0; i < 3; i++)
        if (p_flag[i])
          omega_dot[i] *= factor_etap;

      if (pstyle == TRICLINIC)
        for (i = 3; i < 6; i++)
          if (p_flag[i])
            omega_dot[i] *= factor_etap;

      etap_dot = a*etap_dot + b*random->gaussian();

      factor_etap = exp(-ncfac*dt4*etap_dot);
      for (i = 0; i < 3; i++)
        if (p_flag[i])
          omega_dot[i] *= factor_etap;

      if (pstyle == TRICLINIC)
        for (i = 3; i < 6; i++)
          if (p_flag[i])
            omega_dot[i] *= factor_etap;
    }
    else {
      etap += ncfac*dthalf*etap_dot;

      factor_etap = exp(-ncfac*dthalf*etap_dot);
      for (i = 0; i < 3; i++)
        if (p_flag[i])
          omega_dot[i] *= factor_etap;

      if (pstyle == TRICLINIC)
        for (i = 3; i < 6; i++)
          if (p_flag[i])
            omega_dot[i] *= factor_etap;
    }

    kecurrent = 0.0;
    for (i = 0; i < 3; i++)
      if (p_flag[i])
        kecurrent += omega_mass[i]*omega_dot[i]*omega_dot[i];

    if (pstyle == TRICLINIC)
      for (i = 3; i < 6; i++)
        if (p_flag[i])
          kecurrent += omega_mass[i]*omega_dot[i]*omega_dot[i];

    etap_dot += ncfac*dt4*(kecurrent - lkt_press)/etap_mass;
  }
}

/* ----------------------------------------------------------------------
   perform half-step update of velocities
-----------------------------------------------------------------------*/

void FixNHRegulated::nve_v()
{
  double dtfm;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  if (rmass) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        dtfm = dtf / rmass[i];
        v[i][0] += dtfm*f[i][0];
        v[i][1] += dtfm*f[i][1];
        v[i][2] += dtfm*f[i][2];
      }
    }
  } else {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        dtfm = dtf / mass[type[i]];
        v[i][0] += dtfm*f[i][0];
        v[i][1] += dtfm*f[i][1];
        v[i][2] += dtfm*f[i][2];
      }
    }
  }
}

/* ----------------------------------------------------------------------
   perform full-step update of positions
-----------------------------------------------------------------------*/

void FixNHRegulated::nve_x(double dtv)
{
  double **x = atom->x;
  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup)
    nlocal = atom->nfirst;

  // x update by full step only for atoms in group

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      double dtv_umax = dtv*umax[i];
      double inv_umax = 1.0/umax[i];
      x[i][0] += dtv_umax*tanh(v[i][0]*inv_umax);
      x[i][1] += dtv_umax*tanh(v[i][1]*inv_umax);
      x[i][2] += dtv_umax*tanh(v[i][2]*inv_umax);
    }
}

/* ----------------------------------------------------------------------
   compute sigma tensor
   needed whenever p_target or h0_inv changes
-----------------------------------------------------------------------*/

void FixNHRegulated::compute_sigma()
{
  // if nreset_h0 > 0, reset vol0 and h0_inv
  // every nreset_h0 timesteps

  if (nreset_h0 > 0) {
    int delta = update->ntimestep - update->beginstep;
    if (delta % nreset_h0 == 0) {
      if (dimension == 3) vol0 = domain->xprd * domain->yprd * domain->zprd;
      else vol0 = domain->xprd * domain->yprd;
      h0_inv[0] = domain->h_inv[0];
      h0_inv[1] = domain->h_inv[1];
      h0_inv[2] = domain->h_inv[2];
      h0_inv[3] = domain->h_inv[3];
      h0_inv[4] = domain->h_inv[4];
      h0_inv[5] = domain->h_inv[5];
    }
  }

  // generate upper-triangular half of
  // sigma = vol0*h0inv*(p_target-p_hydro)*h0inv^t
  // units of sigma are are PV/L^2 e.g. atm.A
  //
  // [ 0 5 4 ]   [ 0 5 4 ] [ 0 5 4 ] [ 0 - - ]
  // [ 5 1 3 ] = [ - 1 3 ] [ 5 1 3 ] [ 5 1 - ]
  // [ 4 3 2 ]   [ - - 2 ] [ 4 3 2 ] [ 4 3 2 ]

  sigma[0] =
    vol0*(h0_inv[0]*((p_target[0]-p_hydro)*h0_inv[0] +
                     p_target[5]*h0_inv[5]+p_target[4]*h0_inv[4]) +
          h0_inv[5]*(p_target[5]*h0_inv[0] +
                     (p_target[1]-p_hydro)*h0_inv[5]+p_target[3]*h0_inv[4]) +
          h0_inv[4]*(p_target[4]*h0_inv[0]+p_target[3]*h0_inv[5] +
                     (p_target[2]-p_hydro)*h0_inv[4]));
  sigma[1] =
    vol0*(h0_inv[1]*((p_target[1]-p_hydro)*h0_inv[1] +
                     p_target[3]*h0_inv[3]) +
          h0_inv[3]*(p_target[3]*h0_inv[1] +
                     (p_target[2]-p_hydro)*h0_inv[3]));
  sigma[2] =
    vol0*(h0_inv[2]*((p_target[2]-p_hydro)*h0_inv[2]));
  sigma[3] =
    vol0*(h0_inv[1]*(p_target[3]*h0_inv[2]) +
          h0_inv[3]*((p_target[2]-p_hydro)*h0_inv[2]));
  sigma[4] =
    vol0*(h0_inv[0]*(p_target[4]*h0_inv[2]) +
          h0_inv[5]*(p_target[3]*h0_inv[2]) +
          h0_inv[4]*((p_target[2]-p_hydro)*h0_inv[2]));
  sigma[5] =
    vol0*(h0_inv[0]*(p_target[5]*h0_inv[1]+p_target[4]*h0_inv[3]) +
          h0_inv[5]*((p_target[1]-p_hydro)*h0_inv[1]+p_target[3]*h0_inv[3]) +
          h0_inv[4]*(p_target[3]*h0_inv[1]+(p_target[2]-p_hydro)*h0_inv[3]));
}

/* ----------------------------------------------------------------------
   compute strain energy
-----------------------------------------------------------------------*/

double FixNHRegulated::compute_strain_energy()
{
  // compute strain energy = 0.5*Tr(sigma*h*h^t) in energy units

  double* h = domain->h;
  double d0,d1,d2;

  d0 =
    sigma[0]*(h[0]*h[0]+h[5]*h[5]+h[4]*h[4]) +
    sigma[5]*(          h[1]*h[5]+h[3]*h[4]) +
    sigma[4]*(                    h[2]*h[4]);
  d1 =
    sigma[5]*(          h[5]*h[1]+h[4]*h[3]) +
    sigma[1]*(          h[1]*h[1]+h[3]*h[3]) +
    sigma[3]*(                    h[2]*h[3]);
  d2 =
    sigma[4]*(                    h[4]*h[2]) +
    sigma[3]*(                    h[3]*h[2]) +
    sigma[2]*(                    h[2]*h[2]);

  double energy = 0.5*(d0+d1+d2)/nktv2p;
  return energy;
}

/* ----------------------------------------------------------------------
   compute deviatoric barostat force = h*sigma*h^t
-----------------------------------------------------------------------*/

void FixNHRegulated::compute_deviatoric()
{
  // generate upper-triangular part of h*sigma*h^t
  // units of fdev are are PV, e.g. atm*A^3
  // [ 0 5 4 ]   [ 0 5 4 ] [ 0 5 4 ] [ 0 - - ]
  // [ 5 1 3 ] = [ - 1 3 ] [ 5 1 3 ] [ 5 1 - ]
  // [ 4 3 2 ]   [ - - 2 ] [ 4 3 2 ] [ 4 3 2 ]

  double* h = domain->h;

  fdev[0] =
    h[0]*(sigma[0]*h[0]+sigma[5]*h[5]+sigma[4]*h[4]) +
    h[5]*(sigma[5]*h[0]+sigma[1]*h[5]+sigma[3]*h[4]) +
    h[4]*(sigma[4]*h[0]+sigma[3]*h[5]+sigma[2]*h[4]);
  fdev[1] =
    h[1]*(              sigma[1]*h[1]+sigma[3]*h[3]) +
    h[3]*(              sigma[3]*h[1]+sigma[2]*h[3]);
  fdev[2] =
    h[2]*(                            sigma[2]*h[2]);
  fdev[3] =
    h[1]*(                            sigma[3]*h[2]) +
    h[3]*(                            sigma[2]*h[2]);
  fdev[4] =
    h[0]*(                            sigma[4]*h[2]) +
    h[5]*(                            sigma[3]*h[2]) +
    h[4]*(                            sigma[2]*h[2]);
  fdev[5] =
    h[0]*(              sigma[5]*h[1]+sigma[4]*h[3]) +
    h[5]*(              sigma[1]*h[1]+sigma[3]*h[3]) +
    h[4]*(              sigma[3]*h[1]+sigma[2]*h[3]);
}

/* ----------------------------------------------------------------------
   compute target temperature and kinetic energy
-----------------------------------------------------------------------*/

void FixNHRegulated::compute_temp_target()
{
  double delta = update->ntimestep - update->beginstep;
  if (delta != 0.0) delta /= update->endstep - update->beginstep;

  t_target = t_start + delta * (t_stop-t_start);
  ke_target = tdof * boltz * t_target;
}

/* ----------------------------------------------------------------------
   compute hydrostatic target pressure
-----------------------------------------------------------------------*/

void FixNHRegulated::compute_press_target()
{
  double delta = update->ntimestep - update->beginstep;
  if (delta != 0.0) delta /= update->endstep - update->beginstep;

  p_hydro = 0.0;
  for (int i = 0; i < 3; i++)
    if (p_flag[i]) {
      p_target[i] = p_start[i] + delta * (p_stop[i]-p_start[i]);
      p_hydro += p_target[i];
    }
  if (pdim > 0) p_hydro /= pdim;

  if (pstyle == TRICLINIC)
    for (int i = 3; i < 6; i++)
      p_target[i] = p_start[i] + delta * (p_stop[i]-p_start[i]);

  // if deviatoric, recompute sigma each time p_target changes

  if (deviatoric_flag) compute_sigma();
}

/* ----------------------------------------------------------------------
   update omega_dot, omega
-----------------------------------------------------------------------*/

void FixNHRegulated::nh_omega_dot()
{
  double f_omega,volume;

  if (dimension == 3) volume = domain->xprd*domain->yprd*domain->zprd;
  else volume = domain->xprd*domain->yprd;

  if (deviatoric_flag) compute_deviatoric();

  double mtk_term1 = 0.0;
  if (pstyle == ISO) {
    mtk_term1 = pressure->dof * boltz * pressure->temp;
    mtk_term1 /= pdim * pressure->nmolecules;
  } else {
    double *mvv_current = pressure->vector;
    for (int i = 0; i < 3; i++)
      if (p_flag[i])
        mtk_term1 += mvv_current[i];
    mtk_term1 /= pdim * pressure->nmolecules;
  }

  for (int i = 0; i < 3; i++)
    if (p_flag[i]) {
      f_omega = (p_current[i]-p_hydro)*volume /
        (omega_mass[i] * nktv2p) + mtk_term1 / omega_mass[i];
      if (deviatoric_flag) f_omega -= fdev[i]/(omega_mass[i] * nktv2p);
      omega_dot[i] += f_omega*dthalf;
      omega_dot[i] *= pdrag_factor;
    }

  if (pstyle == TRICLINIC) {
    for (int i = 3; i < 6; i++) {
      if (p_flag[i]) {
        f_omega = p_current[i]*volume/(omega_mass[i] * nktv2p);
        if (deviatoric_flag)
          f_omega -= fdev[i]/(omega_mass[i] * nktv2p);
        omega_dot[i] += f_omega*dthalf;
        omega_dot[i] *= pdrag_factor;
      }
    }
  }
}

/* ----------------------------------------------------------------------
  if any tilt ratios exceed limits, set flip = 1 and compute new tilt values
  do not flip in x or y if non-periodic (can tilt but not flip)
    this is b/c the box length would be changed (dramatically) by flip
  if yz tilt exceeded, adjust C vector by one B vector
  if xz tilt exceeded, adjust C vector by one A vector
  if xy tilt exceeded, adjust B vector by one A vector
  check yz first since it may change xz, then xz check comes after
  if any flip occurs, create new box in domain
  image_flip() adjusts image flags due to box shape change induced by flip
  change_box() puts atoms outside the new box back into the new box
  perform irregular on atoms in lamda coords to migrate atoms to new procs
  important that image_flip comes before remap, since remap may change
    image flags to new values, making eqs in doc of Domain:image_flip incorrect
------------------------------------------------------------------------- */

void FixNHRegulated::pre_exchange()
{
  double xprd = domain->xprd;
  double yprd = domain->yprd;

  // flip is only triggered when tilt exceeds 0.5 by DELTAFLIP
  // this avoids immediate re-flipping due to tilt oscillations

  double xtiltmax = (0.5+DELTAFLIP)*xprd;
  double ytiltmax = (0.5+DELTAFLIP)*yprd;

  int flipxy,flipxz,flipyz;
  flipxy = flipxz = flipyz = 0;

  if (domain->yperiodic) {
    if (domain->yz < -ytiltmax) {
      domain->yz += yprd;
      domain->xz += domain->xy;
      flipyz = 1;
    } else if (domain->yz >= ytiltmax) {
      domain->yz -= yprd;
      domain->xz -= domain->xy;
      flipyz = -1;
    }
  }

  if (domain->xperiodic) {
    if (domain->xz < -xtiltmax) {
      domain->xz += xprd;
      flipxz = 1;
    } else if (domain->xz >= xtiltmax) {
      domain->xz -= xprd;
      flipxz = -1;
    }
    if (domain->xy < -xtiltmax) {
      domain->xy += xprd;
      flipxy = 1;
    } else if (domain->xy >= xtiltmax) {
      domain->xy -= xprd;
      flipxy = -1;
    }
  }

  int flip = 0;
  if (flipxy || flipxz || flipyz) flip = 1;

  if (flip) {
    domain->set_global_box();
    domain->set_local_box();

    domain->image_flip(flipxy,flipxz,flipyz);

    double **x = atom->x;
    imageint *image = atom->image;
    int nlocal = atom->nlocal;
    for (int i = 0; i < nlocal; i++) domain->remap(x[i],image[i]);

    domain->x2lamda(atom->nlocal);
    irregular->migrate_atoms();
    domain->lamda2x(atom->nlocal);
  }
}

/* ----------------------------------------------------------------------
   memory usage of per-atom arrays
------------------------------------------------------------------------- */

double FixNHRegulated::memory_usage()
{
  double bytes = (double)atom->nmax*7*sizeof(double);
  if (irregular)
    bytes += irregular->memory_usage();
  return bytes;
}

/* ----------------------------------------------------------------------
   allocate per-atom arrays
------------------------------------------------------------------------- */

void FixNHRegulated::grow_arrays(int nmax)
{
  memory->grow(v_eta, nmax, 3, "fix_nh_massive:v_eta");
  memory->grow(eta, nmax, 3, "fix_nh_massive:eta");
  memory->grow(umax, nmax, "fix_nh_massive:umax");
}

/* ----------------------------------------------------------------------
   copy values within local per-atom array
------------------------------------------------------------------------- */

void FixNHRegulated::copy_arrays(int i, int j, int /*delflag*/)
{
  v_eta[j][0] = v_eta[i][0];
  v_eta[j][1] = v_eta[i][1];
  v_eta[j][2] = v_eta[i][2];
  if (!langevin_flag) {
    eta[j][0] = eta[i][0];
    eta[j][1] = eta[i][1];
    eta[j][2] = eta[i][2];
  }
  umax[j] = umax[i];
}

/* ----------------------------------------------------------------------
   pack values in local per-atom array for exchange with another proc
------------------------------------------------------------------------- */

int FixNHRegulated::pack_exchange(int i, double *buf)
{
  int n = 0;
  buf[n++] = v_eta[i][0];
  buf[n++] = v_eta[i][1];
  buf[n++] = v_eta[i][2];
  if (!langevin_flag) {
    buf[n++] = eta[i][0];
    buf[n++] = eta[i][1];
    buf[n++] = eta[i][2];
  }
  buf[n++] = umax[i];
  return n;
}

/* ----------------------------------------------------------------------
   unpack values in local per-atom array from exchange with another proc
------------------------------------------------------------------------- */

int FixNHRegulated::unpack_exchange(int nlocal, double *buf)
{
  int n = 0;
  v_eta[nlocal][0] = buf[n++];
  v_eta[nlocal][1] = buf[n++];
  v_eta[nlocal][2] = buf[n++];
  if (!langevin_flag) {
    eta[nlocal][0] = buf[n++];
    eta[nlocal][1] = buf[n++];
    eta[nlocal][2] = buf[n++];
  }
  umax[nlocal] = buf[n++];
  return n;
}
