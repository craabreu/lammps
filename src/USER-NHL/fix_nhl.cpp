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

/* ----------------------------------------------------------------------
   Contributing author: Charlles Abreu (UFRJ)
   Adapted from FixNH (authors: Mark Stevens and Aidan Thompson - SNL)
------------------------------------------------------------------------- */

#include "fix_nhl.h"
#include <cstring>
#include <cmath>
#include "atom.h"
#include "force.h"
#include "group.h"
#include "comm.h"
#include "neighbor.h"
#include "irregular.h"
#include "modify.h"
#include "fix_deform.h"
#include "compute.h"
#include "kspace.h"
#include "update.h"
#include "respa.h"
#include "domain.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace FixConst;

#define DELTAFLIP 0.1
#define TILTMAX 1.5

enum{NOBIAS,BIAS};
enum{NONE,XYZ,XY,YZ,XZ};
enum{ISO,ANISO,TRICLINIC};

/* ----------------------------------------------------------------------
   NVT,NPH,NPT integrators for Nose-Hoover-Langevin equations of motion
 ---------------------------------------------------------------------- */

FixNHL::FixNHL(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  rfix(NULL), id_dilate(NULL), irregular(NULL), id_temp(NULL), id_press(NULL),
  eta(NULL), eta_dot(NULL), eta_dotdot(NULL),
  eta_mass(NULL), etap(NULL), etap_dot(NULL), etap_dotdot(NULL),
  etap_mass(NULL)
{
  if (narg < 4) error->all(FLERR,"Illegal fix <ensemble>/nhl command");

  restart_global = 1;
  dynamic_group_allow = 1;
  time_integrate = 1;
  scalar_flag = 1;
  vector_flag = 1;
  global_freq = 1;
  extscalar = 1;
  extvector = 0;

  // default values

  pcouple = NONE;
  drag = 0.0;
  allremap = 1;
  id_dilate = NULL;
  nc_tchain = nc_pchain = 1;
  mtk_flag = 1;
  deviatoric_flag = 0;
  nreset_h0 = 0;
  eta_mass_flag = 1;
  omega_mass_flag = 0;
  etap_mass_flag = 0;
  flipflag = 1;
  dipole_flag = 0;
  dlm_flag = 0;

  tcomputeflag = 0;
  pcomputeflag = 0;
  id_temp = NULL;
  id_press = NULL;

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
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix <ensemble>/nhl command");
      tstat_flag = 1;
      t_start = force->numeric(FLERR,arg[iarg+1]);
      t_target = t_start;
      t_stop = force->numeric(FLERR,arg[iarg+2]);
      t_period = force->numeric(FLERR,arg[iarg+3]);
      if (t_start <= 0.0 || t_stop <= 0.0)
        error->all(FLERR,
                   "Target temperature for fix <ensemble>/nhl cannot be 0.0");
      iarg += 4;

    } else if (strcmp(arg[iarg],"iso") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix <ensemble>/nhl command");
      pcouple = XYZ;
      p_start[0] = p_start[1] = p_start[2] = force->numeric(FLERR,arg[iarg+1]);
      p_stop[0] = p_stop[1] = p_stop[2] = force->numeric(FLERR,arg[iarg+2]);
      p_period[0] = p_period[1] = p_period[2] =
        force->numeric(FLERR,arg[iarg+3]);
      p_flag[0] = p_flag[1] = p_flag[2] = 1;
      if (dimension == 2) {
        p_start[2] = p_stop[2] = p_period[2] = 0.0;
        p_flag[2] = 0;
      }
      iarg += 4;
    } else if (strcmp(arg[iarg],"aniso") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix <ensemble>/nhl command");
      pcouple = NONE;
      p_start[0] = p_start[1] = p_start[2] = force->numeric(FLERR,arg[iarg+1]);
      p_stop[0] = p_stop[1] = p_stop[2] = force->numeric(FLERR,arg[iarg+2]);
      p_period[0] = p_period[1] = p_period[2] =
        force->numeric(FLERR,arg[iarg+3]);
      p_flag[0] = p_flag[1] = p_flag[2] = 1;
      if (dimension == 2) {
        p_start[2] = p_stop[2] = p_period[2] = 0.0;
        p_flag[2] = 0;
      }
      iarg += 4;
    } else if (strcmp(arg[iarg],"tri") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix <ensemble>/nhl command");
      pcouple = NONE;
      scalexy = scalexz = scaleyz = 0;
      p_start[0] = p_start[1] = p_start[2] = force->numeric(FLERR,arg[iarg+1]);
      p_stop[0] = p_stop[1] = p_stop[2] = force->numeric(FLERR,arg[iarg+2]);
      p_period[0] = p_period[1] = p_period[2] =
        force->numeric(FLERR,arg[iarg+3]);
      p_flag[0] = p_flag[1] = p_flag[2] = 1;
      p_start[3] = p_start[4] = p_start[5] = 0.0;
      p_stop[3] = p_stop[4] = p_stop[5] = 0.0;
      p_period[3] = p_period[4] = p_period[5] =
        force->numeric(FLERR,arg[iarg+3]);
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
    } else if (strcmp(arg[iarg],"x") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix <ensemble>/nhl command");
      p_start[0] = force->numeric(FLERR,arg[iarg+1]);
      p_stop[0] = force->numeric(FLERR,arg[iarg+2]);
      p_period[0] = force->numeric(FLERR,arg[iarg+3]);
      p_flag[0] = 1;
      deviatoric_flag = 1;
      iarg += 4;
    } else if (strcmp(arg[iarg],"y") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix <ensemble>/nhl command");
      p_start[1] = force->numeric(FLERR,arg[iarg+1]);
      p_stop[1] = force->numeric(FLERR,arg[iarg+2]);
      p_period[1] = force->numeric(FLERR,arg[iarg+3]);
      p_flag[1] = 1;
      deviatoric_flag = 1;
      iarg += 4;
    } else if (strcmp(arg[iarg],"z") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix <ensemble>/nhl command");
      p_start[2] = force->numeric(FLERR,arg[iarg+1]);
      p_stop[2] = force->numeric(FLERR,arg[iarg+2]);
      p_period[2] = force->numeric(FLERR,arg[iarg+3]);
      p_flag[2] = 1;
      deviatoric_flag = 1;
      iarg += 4;
      if (dimension == 2)
        error->all(FLERR,"Invalid fix <ensemble>/nhl command for a 2d simulation");

    } else if (strcmp(arg[iarg],"yz") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix <ensemble>/nhl command");
      p_start[3] = force->numeric(FLERR,arg[iarg+1]);
      p_stop[3] = force->numeric(FLERR,arg[iarg+2]);
      p_period[3] = force->numeric(FLERR,arg[iarg+3]);
      p_flag[3] = 1;
      deviatoric_flag = 1;
      scaleyz = 0;
      iarg += 4;
      if (dimension == 2)
        error->all(FLERR,"Invalid fix <ensemble>/nhl command for a 2d simulation");
    } else if (strcmp(arg[iarg],"xz") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix <ensemble>/nhl command");
      p_start[4] = force->numeric(FLERR,arg[iarg+1]);
      p_stop[4] = force->numeric(FLERR,arg[iarg+2]);
      p_period[4] = force->numeric(FLERR,arg[iarg+3]);
      p_flag[4] = 1;
      deviatoric_flag = 1;
      scalexz = 0;
      iarg += 4;
      if (dimension == 2)
        error->all(FLERR,"Invalid fix <ensemble>/nhl command for a 2d simulation");
    } else if (strcmp(arg[iarg],"xy") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix <ensemble>/nhl command");
      p_start[5] = force->numeric(FLERR,arg[iarg+1]);
      p_stop[5] = force->numeric(FLERR,arg[iarg+2]);
      p_period[5] = force->numeric(FLERR,arg[iarg+3]);
      p_flag[5] = 1;
      deviatoric_flag = 1;
      scalexy = 0;
      iarg += 4;

    } else if (strcmp(arg[iarg],"couple") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix <ensemble>/nhl command");
      if (strcmp(arg[iarg+1],"xyz") == 0) pcouple = XYZ;
      else if (strcmp(arg[iarg+1],"xy") == 0) pcouple = XY;
      else if (strcmp(arg[iarg+1],"yz") == 0) pcouple = YZ;
      else if (strcmp(arg[iarg+1],"xz") == 0) pcouple = XZ;
      else if (strcmp(arg[iarg+1],"none") == 0) pcouple = NONE;
      else error->all(FLERR,"Illegal fix <ensemble>/nhl command");
      iarg += 2;

    } else if (strcmp(arg[iarg],"drag") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix <ensemble>/nhl command");
      drag = force->numeric(FLERR,arg[iarg+1]);
      if (drag < 0.0) error->all(FLERR,"Illegal fix <ensemble>/nhl command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"dilate") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix <ensemble>/nhl command");
      if (strcmp(arg[iarg+1],"all") == 0) allremap = 1;
      else {
        allremap = 0;
        delete [] id_dilate;
        int n = strlen(arg[iarg+1]) + 1;
        id_dilate = new char[n];
        strcpy(id_dilate,arg[iarg+1]);
        int idilate = group->find(id_dilate);
        if (idilate == -1)
          error->all(FLERR,"Fix <ensemble>/nhl dilate group ID does not exist");
      }
      iarg += 2;

    } else if (strcmp(arg[iarg],"mtk") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix <ensemble>/nhl command");
      if (strcmp(arg[iarg+1],"yes") == 0) mtk_flag = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) mtk_flag = 0;
      else error->all(FLERR,"Illegal fix <ensemble>/nhl command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"tloop") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix <ensemble>/nhl command");
      nc_tchain = force->inumeric(FLERR,arg[iarg+1]);
      if (nc_tchain < 0) error->all(FLERR,"Illegal fix <ensemble>/nhl command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"ploop") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix <ensemble>/nhl command");
      nc_pchain = force->inumeric(FLERR,arg[iarg+1]);
      if (nc_pchain < 0) error->all(FLERR,"Illegal fix <ensemble>/nhl command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"nreset") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix <ensemble>/nhl command");
      nreset_h0 = force->inumeric(FLERR,arg[iarg+1]);
      if (nreset_h0 < 0) error->all(FLERR,"Illegal fix <ensemble>/nhl command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"scalexy") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix <ensemble>/nhl command");
      if (strcmp(arg[iarg+1],"yes") == 0) scalexy = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) scalexy = 0;
      else error->all(FLERR,"Illegal fix <ensemble>/nhl command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"scalexz") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix <ensemble>/nhl command");
      if (strcmp(arg[iarg+1],"yes") == 0) scalexz = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) scalexz = 0;
      else error->all(FLERR,"Illegal fix <ensemble>/nhl command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"scaleyz") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix <ensemble>/nhl command");
      if (strcmp(arg[iarg+1],"yes") == 0) scaleyz = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) scaleyz = 0;
      else error->all(FLERR,"Illegal fix <ensemble>/nhl command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"flip") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix <ensemble>/nhl command");
      if (strcmp(arg[iarg+1],"yes") == 0) flipflag = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) flipflag = 0;
      else error->all(FLERR,"Illegal fix <ensemble>/nhl command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"update") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix <ensemble>/nhl command");
      if (strcmp(arg[iarg+1],"dipole") == 0) dipole_flag = 1;
      else if (strcmp(arg[iarg+1],"dipole/dlm") == 0) {
        dipole_flag = 1;
        dlm_flag = 1;
      } else error->all(FLERR,"Illegal fix <ensemble>/nhl command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"fixedpoint") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix <ensemble>/nhl command");
      fixedpoint[0] = force->numeric(FLERR,arg[iarg+1]);
      fixedpoint[1] = force->numeric(FLERR,arg[iarg+2]);
      fixedpoint[2] = force->numeric(FLERR,arg[iarg+3]);
      iarg += 4;

    // disc keyword is also parsed in fix/nh/sphere

    } else if (strcmp(arg[iarg],"disc") == 0) {
      iarg++;

    // keywords erate, strain, and ext are also parsed in fix/nh/uef

    } else if (strcmp(arg[iarg],"erate") == 0) {
      iarg += 3;
    } else if (strcmp(arg[iarg],"strain") == 0) {
      iarg += 3;
    } else if (strcmp(arg[iarg],"ext") == 0) {
      iarg += 2;

    } else error->all(FLERR,"Illegal fix <ensemble>/nhl command");
  }

  // error checks

  if (dimension == 2 && (p_flag[2] || p_flag[3] || p_flag[4]))
    error->all(FLERR,"Invalid fix <ensemble>/nhl command for a 2d simulation");
  if (dimension == 2 && (pcouple == YZ || pcouple == XZ))
    error->all(FLERR,"Invalid fix <ensemble>/nhl command for a 2d simulation");
  if (dimension == 2 && (scalexz == 1 || scaleyz == 1 ))
    error->all(FLERR,"Invalid fix <ensemble>/nhl command for a 2d simulation");

  if (pcouple == XYZ && (p_flag[0] == 0 || p_flag[1] == 0))
    error->all(FLERR,"Invalid fix <ensemble>/nhl command pressure settings");
  if (pcouple == XYZ && dimension == 3 && p_flag[2] == 0)
    error->all(FLERR,"Invalid fix <ensemble>/nhl command pressure settings");
  if (pcouple == XY && (p_flag[0] == 0 || p_flag[1] == 0))
    error->all(FLERR,"Invalid fix <ensemble>/nhl command pressure settings");
  if (pcouple == YZ && (p_flag[1] == 0 || p_flag[2] == 0))
    error->all(FLERR,"Invalid fix <ensemble>/nhl command pressure settings");
  if (pcouple == XZ && (p_flag[0] == 0 || p_flag[2] == 0))
    error->all(FLERR,"Invalid fix <ensemble>/nhl command pressure settings");

  // require periodicity in tensile dimension

  if (p_flag[0] && domain->xperiodic == 0)
    error->all(FLERR,"Cannot use fix <ensemble>/nhl on a non-periodic dimension");
  if (p_flag[1] && domain->yperiodic == 0)
    error->all(FLERR,"Cannot use fix <ensemble>/nhl on a non-periodic dimension");
  if (p_flag[2] && domain->zperiodic == 0)
    error->all(FLERR,"Cannot use fix <ensemble>/nhl on a non-periodic dimension");

  // require periodicity in 2nd dim of off-diagonal tilt component

  if (p_flag[3] && domain->zperiodic == 0)
    error->all(FLERR,
               "Cannot use fix <ensemble>/nhl on a 2nd non-periodic dimension");
  if (p_flag[4] && domain->zperiodic == 0)
    error->all(FLERR,
               "Cannot use fix <ensemble>/nhl on a 2nd non-periodic dimension");
  if (p_flag[5] && domain->yperiodic == 0)
    error->all(FLERR,
               "Cannot use fix <ensemble>/nhl on a 2nd non-periodic dimension");

  if (scaleyz == 1 && domain->zperiodic == 0)
    error->all(FLERR,"Cannot use fix <ensemble>/nhl "
               "with yz scaling when z is non-periodic dimension");
  if (scalexz == 1 && domain->zperiodic == 0)
    error->all(FLERR,"Cannot use fix <ensemble>/nhl "
               "with xz scaling when z is non-periodic dimension");
  if (scalexy == 1 && domain->yperiodic == 0)
    error->all(FLERR,"Cannot use fix <ensemble>/nhl "
               "with xy scaling when y is non-periodic dimension");

  if (p_flag[3] && scaleyz == 1)
    error->all(FLERR,"Cannot use fix <ensemble>/nhl with "
               "both yz dynamics and yz scaling");
  if (p_flag[4] && scalexz == 1)
    error->all(FLERR,"Cannot use fix <ensemble>/nhl with "
               "both xz dynamics and xz scaling");
  if (p_flag[5] && scalexy == 1)
    error->all(FLERR,"Cannot use fix <ensemble>/nhl with "
               "both xy dynamics and xy scaling");

  if (!domain->triclinic && (p_flag[3] || p_flag[4] || p_flag[5]))
    error->all(FLERR,"Can not specify Pxy/Pxz/Pyz in "
               "fix <ensemble>/nhl with non-triclinic box");

  if (pcouple == XYZ && dimension == 3 &&
      (p_start[0] != p_start[1] || p_start[0] != p_start[2] ||
       p_stop[0] != p_stop[1] || p_stop[0] != p_stop[2] ||
       p_period[0] != p_period[1] || p_period[0] != p_period[2]))
    error->all(FLERR,"Invalid fix <ensemble>/nhl pressure settings");
  if (pcouple == XYZ && dimension == 2 &&
      (p_start[0] != p_start[1] || p_stop[0] != p_stop[1] ||
       p_period[0] != p_period[1]))
    error->all(FLERR,"Invalid fix <ensemble>/nhl pressure settings");
  if (pcouple == XY &&
      (p_start[0] != p_start[1] || p_stop[0] != p_stop[1] ||
       p_period[0] != p_period[1]))
    error->all(FLERR,"Invalid fix <ensemble>/nhl pressure settings");
  if (pcouple == YZ &&
      (p_start[1] != p_start[2] || p_stop[1] != p_stop[2] ||
       p_period[1] != p_period[2]))
    error->all(FLERR,"Invalid fix <ensemble>/nhl pressure settings");
  if (pcouple == XZ &&
      (p_start[0] != p_start[2] || p_stop[0] != p_stop[2] ||
       p_period[0] != p_period[2]))
    error->all(FLERR,"Invalid fix <ensemble>/nhl pressure settings");

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
    error->all(FLERR,"Fix <ensemble>/nhl damping parameters must be > 0.0");

  // set pstat_flag and box change and restart_pbc variables

  pre_exchange_flag = 0;
  pstat_flag = 0;
  pstyle = ISO;

  for (int i = 0; i < 6; i++)
    if (p_flag[i]) pstat_flag = 1;

  if (pstat_flag) {
    if (p_flag[0] || p_flag[1] || p_flag[2]) box_change_size = 1;
    if (p_flag[3] || p_flag[4] || p_flag[5]) box_change_shape = 1;
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
      pre_exchange_flag = 1;
    if (flipflag && (domain->yz != 0.0 || domain->xz != 0.0 ||
                     domain->xy != 0.0))
      pre_exchange_flag = 1;
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
    eta = new double[1];

    // add one extra dummy thermostat, set to zero

    eta_dot = new double[1+1];
    eta_dot[1] = 0.0;
    eta_dotdot = new double[1];
    eta[0] = eta_dot[0] = eta_dotdot[0] = 0.0;
    eta_mass = new double[1];
    size_vector += 2*2*1;
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

    etap = new double[1];

    // add one extra dummy thermostat, set to zero

    etap_dot = new double[1+1];
    etap_dot[1] = 0.0;
    etap_dotdot = new double[1];
    etap[0] = etap_dot[0] = etap_dotdot[0] = 0.0;
    etap_mass = new double[1];
    size_vector += 2*2*1;

    if (deviatoric_flag) size_vector += 1;
  }

  nrigid = 0;
  rfix = NULL;

  if (pre_exchange_flag) irregular = new Irregular(lmp);
  else irregular = NULL;

  // initialize vol0,t0 to zero to signal uninitialized
  // values then assigned in init(), if necessary

  vol0 = t0 = 0.0;
}

/* ---------------------------------------------------------------------- */

FixNHL::~FixNHL()
{
  if (copymode) return;

  delete [] id_dilate;
  delete [] rfix;

  delete irregular;

  // delete temperature and pressure if fix created them

  if (tcomputeflag) modify->delete_compute(id_temp);
  delete [] id_temp;

  if (tstat_flag) {
    delete [] eta;
    delete [] eta_dot;
    delete [] eta_dotdot;
    delete [] eta_mass;
  }

  if (pstat_flag) {
    if (pcomputeflag) modify->delete_compute(id_press);
    delete [] id_press;
    if (1) {
      delete [] etap;
      delete [] etap_dot;
      delete [] etap_dotdot;
      delete [] etap_mass;
    }
  }
}

/* ---------------------------------------------------------------------- */

int FixNHL::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  mask |= THERMO_ENERGY;
  mask |= INITIAL_INTEGRATE_RESPA;
  mask |= FINAL_INTEGRATE_RESPA;
  if (pre_exchange_flag) mask |= PRE_EXCHANGE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixNHL::init()
{
  // recheck that dilate group has not been deleted

  if (allremap == 0) {
    int idilate = group->find(id_dilate);
    if (idilate == -1)
      error->all(FLERR,"Fix <ensemble>/nhl dilate group ID does not exist");
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

  // set temperature and pressure ptrs

  int icompute = modify->find_compute(id_temp);
  if (icompute < 0)
    error->all(FLERR,"Temperature ID for fix nvt/npt does not exist");
  temperature = modify->compute[icompute];

  if (temperature->tempbias) which = BIAS;
  else which = NOBIAS;

  if (pstat_flag) {
    icompute = modify->find_compute(id_press);
    if (icompute < 0)
      error->all(FLERR,"Pressure ID for fix npt/nph does not exist");
    pressure = modify->compute[icompute];
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

  if (force->kspace) kspace_flag = 1;
  else kspace_flag = 0;

  if (strstr(update->integrate_style,"respa")) {
    nlevels_respa = ((Respa *) update->integrate)->nlevels;
    step_respa = ((Respa *) update->integrate)->step;
    dto = 0.5*step_respa[0];
  }

  // detect if any rigid fixes exist so rigid bodies move when box is remapped
  // rfix[] = indices to each fix rigid

  delete [] rfix;
  nrigid = 0;
  rfix = NULL;

  for (int i = 0; i < modify->nfix; i++)
    if (modify->fix[i]->rigid_flag) nrigid++;
  if (nrigid) {
    rfix = new int[nrigid];
    nrigid = 0;
    for (int i = 0; i < modify->nfix; i++)
      if (modify->fix[i]->rigid_flag) rfix[nrigid++] = i;
  }
}

/* ----------------------------------------------------------------------
   compute T,P before integrator starts
------------------------------------------------------------------------- */

void FixNHL::setup(int /*vflag*/)
{
  // tdof needed by compute_temp_target()

  t_current = temperature->compute_scalar();
  tdof = temperature->dof;

  // t_target is needed by NVT and NPT in compute_scalar()
  // If no thermostat or using fix nphug,
  // t_target must be defined by other means.

  if (tstat_flag && strstr(style,"nphug") == NULL) {
    compute_temp_target();
  } else if (pstat_flag) {

    // t0 = reference temperature for masses
    // cannot be done in init() b/c temperature cannot be called there
    // is b/c Modify::init() inits computes after fixes due to dof dependence
    // guesstimate a unit-dependent t0 if actual T = 0.0
    // if it was read in from a restart file, leave it be

    if (t0 == 0.0) {
      t0 = temperature->compute_scalar();
      if (t0 == 0.0) {
        if (strcmp(update->unit_style,"lj") == 0) t0 = 1.0;
        else t0 = 300.0;
      }
    }
    t_target = t0;
  }

  if (pstat_flag) compute_press_target();

  if (pstat_flag) {
    if (pstyle == ISO) pressure->compute_scalar();
    else pressure->compute_vector();
    couple();
    pressure->addstep(update->ntimestep+1);
  }

  // masses and initial forces on thermostat variables

  if (tstat_flag)
    eta_mass[0] = tdof * boltz * t_target / (t_freq*t_freq);

  // masses and initial forces on barostat variables

  if (pstat_flag) {
    double kt = boltz * t_target;
    double nkt = (atom->natoms + 1) * kt;

    for (int i = 0; i < 3; i++)
      if (p_flag[i])
        omega_mass[i] = nkt/(p_freq[i]*p_freq[i]);

    if (pstyle == TRICLINIC) {
      for (int i = 3; i < 6; i++)
        if (p_flag[i]) omega_mass[i] = nkt/(p_freq[i]*p_freq[i]);
    }

  // masses and initial forces on barostat thermostat variables

    etap_mass[0] = boltz * t_target / (p_freq_max*p_freq_max);
  }
}

/* ----------------------------------------------------------------------
   1st half of Verlet update
------------------------------------------------------------------------- */

void FixNHL::initial_integrate(int /*vflag*/)
{
  // update eta_press_dot

  if (pstat_flag && 1) nhc_press_integrate();

  // update eta_dot

  if (tstat_flag) {
    compute_temp_target();
    nhc_temp_integrate();
  }

  // need to recompute pressure to account for change in KE
  // t_current is up-to-date, but compute_temperature is not
  // compute appropriately coupled elements of mvv_current

  if (pstat_flag) {
    if (pstyle == ISO) {
      temperature->compute_scalar();
      pressure->compute_scalar();
    } else {
      temperature->compute_vector();
      pressure->compute_vector();
    }
    couple();
    pressure->addstep(update->ntimestep+1);
  }

  if (pstat_flag) {
    compute_press_target();
    nh_omega_dot();
    nh_v_press();
  }

  nve_v();

  // remap simulation box by 1/2 step

  if (pstat_flag) remap();

  nve_x();

  // remap simulation box by 1/2 step
  // redo KSpace coeffs since volume has changed

  if (pstat_flag) {
    remap();
    if (kspace_flag) force->kspace->setup();
  }
}

/* ----------------------------------------------------------------------
   2nd half of Verlet update
------------------------------------------------------------------------- */

void FixNHL::final_integrate()
{
  nve_v();

  // re-compute temp before nh_v_press()
  // only needed for temperature computes with BIAS on reneighboring steps:
  //   b/c some biases store per-atom values (e.g. temp/profile)
  //   per-atom values are invalid if reneigh/comm occurred
  //     since temp->compute() in initial_integrate()

  if (which == BIAS && neighbor->ago == 0)
    t_current = temperature->compute_scalar();

  if (pstat_flag) nh_v_press();

  // compute new T,P after velocities rescaled by nh_v_press()
  // compute appropriately coupled elements of mvv_current

  t_current = temperature->compute_scalar();
  tdof = temperature->dof;

  // need to recompute pressure to account for change in KE
  // t_current is up-to-date, but compute_temperature is not
  // compute appropriately coupled elements of mvv_current

  if (pstat_flag) {
    if (pstyle == ISO) pressure->compute_scalar();
    else {
      temperature->compute_vector();
      pressure->compute_vector();
    }
    couple();
    pressure->addstep(update->ntimestep+1);
  }

  if (pstat_flag) nh_omega_dot();

  // update eta_dot
  // update eta_press_dot

  if (tstat_flag) nhc_temp_integrate();
  if (pstat_flag && 1) nhc_press_integrate();
}

/* ---------------------------------------------------------------------- */

void FixNHL::initial_integrate_respa(int /*vflag*/, int ilevel, int /*iloop*/)
{
  // set timesteps by level

  dtv = step_respa[ilevel];
  dtf = 0.5 * step_respa[ilevel] * force->ftm2v;
  dthalf = 0.5 * step_respa[ilevel];

  // outermost level - update eta_dot and omega_dot, apply to v
  // all other levels - NVE update of v
  // x,v updates only performed for atoms in group

  if (ilevel == nlevels_respa-1) {

    // update eta_press_dot

    if (pstat_flag && 1) nhc_press_integrate();

    // update eta_dot

    if (tstat_flag) {
      compute_temp_target();
      nhc_temp_integrate();
    }

    // recompute pressure to account for change in KE
    // t_current is up-to-date, but compute_temperature is not
    // compute appropriately coupled elements of mvv_current

    if (pstat_flag) {
      if (pstyle == ISO) {
        temperature->compute_scalar();
        pressure->compute_scalar();
      } else {
        temperature->compute_vector();
        pressure->compute_vector();
      }
      couple();
      pressure->addstep(update->ntimestep+1);
    }

    if (pstat_flag) {
      compute_press_target();
      nh_omega_dot();
      nh_v_press();
    }

    nve_v();

  } else nve_v();

  // innermost level - also update x only for atoms in group
  // if barostat, perform 1/2 step remap before and after

  if (ilevel == 0) {
    if (pstat_flag) remap();
    nve_x();
    if (pstat_flag) remap();
  }

  // if barostat, redo KSpace coeffs at outermost level,
  // since volume has changed

  if (ilevel == nlevels_respa-1 && kspace_flag && pstat_flag)
    force->kspace->setup();
}

/* ---------------------------------------------------------------------- */

void FixNHL::final_integrate_respa(int ilevel, int /*iloop*/)
{
  // set timesteps by level

  dtf = 0.5 * step_respa[ilevel] * force->ftm2v;
  dthalf = 0.5 * step_respa[ilevel];

  // outermost level - update eta_dot and omega_dot, apply via final_integrate
  // all other levels - NVE update of v

  if (ilevel == nlevels_respa-1) final_integrate();
  else nve_v();
}

/* ---------------------------------------------------------------------- */

void FixNHL::couple()
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

  // switch order from xy-xz-yz to Voigt

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
   remap all atoms or dilate group atoms depending on allremap flag
   if rigid bodies exist, scale rigid body centers-of-mass
------------------------------------------------------------------------- */

void FixNHL::remap()
{
  int i;
  double oldlo,oldhi;
  double expfac;

  double **x = atom->x;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  double *h = domain->h;

  // omega is not used, except for book-keeping

  for (int i = 0; i < 6; i++) omega[i] += dto*omega_dot[i];

  // convert pertinent atoms and rigid bodies to lamda coords

  if (allremap) domain->x2lamda(nlocal);
  else {
    for (i = 0; i < nlocal; i++)
      if (mask[i] & dilate_group_bit)
        domain->x2lamda(x[i],x[i]);
  }

  if (nrigid)
    for (i = 0; i < nrigid; i++)
      modify->fix[rfix[i]]->deform(0);

  // reset global and local box to new size/shape

  // this operation corresponds to applying the
  // translate and scale operations
  // corresponding to the solution of the following ODE:
  //
  // h_dot = omega_dot * h
  //
  // where h_dot, omega_dot and h are all upper-triangular
  // 3x3 tensors. In Voigt notation, the elements of the
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

  // convert pertinent atoms and rigid bodies back to box coords

  if (allremap) domain->lamda2x(nlocal);
  else {
    for (i = 0; i < nlocal; i++)
      if (mask[i] & dilate_group_bit)
        domain->lamda2x(x[i],x[i]);
  }

  if (nrigid)
    for (i = 0; i < nrigid; i++)
      modify->fix[rfix[i]]->deform(1);
}

/* ----------------------------------------------------------------------
   pack entire state of Fix into one write
------------------------------------------------------------------------- */

void FixNHL::write_restart(FILE *fp)
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

int FixNHL::size_restart_global()
{
  int nsize = 2;
  if (tstat_flag) nsize += 1 + 2*1;
  if (pstat_flag) {
    nsize += 16 + 2*1;
    if (deviatoric_flag) nsize += 6;
  }

  return nsize;
}

/* ----------------------------------------------------------------------
   pack restart data
------------------------------------------------------------------------- */

int FixNHL::pack_restart_data(double *list)
{
  int n = 0;

  list[n++] = tstat_flag;
  if (tstat_flag) {
    list[n++] = 1;
    list[n++] = eta[0];
    list[n++] = eta_dot[0];
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
    list[n++] = 1;
    list[n++] = etap[0];
    list[n++] = etap_dot[0];

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

void FixNHL::restart(char *buf)
{
  int n = 0;
  double *list = (double *) buf;
  int flag = static_cast<int> (list[n++]);
  if (flag) {
    int m = static_cast<int> (list[n++]);
    if (tstat_flag && m == 1) {
      eta[0] = list[n++];
      eta_dot[0] = list[n++];
    } else n += 2*m;
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
    int m = static_cast<int> (list[n++]);
    if (pstat_flag && m == 1) {
      etap[0] = list[n++];
      etap_dot[0] = list[n++];
    } else n+=2*m;
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

int FixNHL::modify_param(int narg, char **arg)
{
  if (strcmp(arg[0],"temp") == 0) {
    if (narg < 2) error->all(FLERR,"Illegal fix_modify command");
    if (tcomputeflag) {
      modify->delete_compute(id_temp);
      tcomputeflag = 0;
    }
    delete [] id_temp;
    int n = strlen(arg[1]) + 1;
    id_temp = new char[n];
    strcpy(id_temp,arg[1]);

    int icompute = modify->find_compute(arg[1]);
    if (icompute < 0)
      error->all(FLERR,"Could not find fix_modify temperature ID");
    temperature = modify->compute[icompute];

    if (temperature->tempflag == 0)
      error->all(FLERR,
                 "Fix_modify temperature ID does not compute temperature");
    if (temperature->igroup != 0 && comm->me == 0)
      error->warning(FLERR,"Temperature for fix modify is not for group all");

    // reset id_temp of pressure to new temperature ID

    if (pstat_flag) {
      icompute = modify->find_compute(id_press);
      if (icompute < 0)
        error->all(FLERR,"Pressure ID for fix modify does not exist");
      modify->compute[icompute]->reset_extra_compute_fix(id_temp);
    }

    return 2;

  } else if (strcmp(arg[0],"press") == 0) {
    if (narg < 2) error->all(FLERR,"Illegal fix_modify command");
    if (!pstat_flag) error->all(FLERR,"Illegal fix_modify command");
    if (pcomputeflag) {
      modify->delete_compute(id_press);
      pcomputeflag = 0;
    }
    delete [] id_press;
    int n = strlen(arg[1]) + 1;
    id_press = new char[n];
    strcpy(id_press,arg[1]);

    int icompute = modify->find_compute(arg[1]);
    if (icompute < 0) error->all(FLERR,"Could not find fix_modify pressure ID");
    pressure = modify->compute[icompute];

    if (pressure->pressflag == 0)
      error->all(FLERR,"Fix_modify pressure ID does not compute pressure");
    return 2;
  }

  return 0;
}

/* ---------------------------------------------------------------------- */

double FixNHL::compute_scalar()
{
  int i;
  double volume;
  double energy;
  double kt = boltz * t_target;
  double lkt_press = 0.0;
  int ich;
  if (dimension == 3) volume = domain->xprd * domain->yprd * domain->zprd;
  else volume = domain->xprd * domain->yprd;

  energy = 0.0;

  // thermostat chain energy is equivalent to Eq. (2) in
  // Martyna, Tuckerman, Tobias, Klein, Mol Phys, 87, 1117
  // Sum(0.5*p_eta_k^2/Q_k,k=1,M) + L*k*T*eta_1 + Sum(k*T*eta_k,k=2,M),
  // where L = tdof
  //       M = 1
  //       p_eta_k = Q_k*eta_dot[k-1]
  //       Q_1 = L*k*T/t_freq^2
  //       Q_k = k*T/t_freq^2, k > 1

  if (tstat_flag) {
    energy += ke_target * eta[0] + 0.5*eta_mass[0]*eta_dot[0]*eta_dot[0];
    for (ich = 1; ich < 1; ich++)
      energy += kt * eta[ich] + 0.5*eta_mass[ich]*eta_dot[ich]*eta_dot[ich];
  }

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

    if (1) {
      energy += lkt_press * etap[0] + 0.5*etap_mass[0]*etap_dot[0]*etap_dot[0];
      for (ich = 1; ich < 1; ich++)
        energy += kt * etap[ich] +
          0.5*etap_mass[ich]*etap_dot[ich]*etap_dot[ich];
    }

    // extra contribution from strain energy

    if (deviatoric_flag) energy += compute_strain_energy();
  }

  return energy;
}

/* ----------------------------------------------------------------------
   return a single element of the following vectors, in this order:
      eta[tchain], eta_dot[tchain], omega[ndof], omega_dot[ndof]
      etap[pchain], etap_dot[pchain], PE_eta[tchain], KE_eta_dot[tchain]
      PE_omega[ndof], KE_omega_dot[ndof], PE_etap[pchain], KE_etap_dot[pchain]
      PE_strain[1]
  if no thermostat exists, related quantities are omitted from the list
  if no barostat exists, related quantities are omitted from the list
  ndof = 1,3,6 degrees of freedom for pstyle = ISO,ANISO,TRI
------------------------------------------------------------------------- */

double FixNHL::compute_vector(int n)
{
  int ilen;

  if (tstat_flag) {
    ilen = 1;
    if (n < ilen) return eta[n];
    n -= ilen;
    ilen = 1;
    if (n < ilen) return eta_dot[n];
    n -= ilen;
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

    if (1) {
      ilen = 1;
      if (n < ilen) return etap[n];
      n -= ilen;
      ilen = 1;
      if (n < ilen) return etap_dot[n];
      n -= ilen;
    }
  }

  double volume;
  double kt = boltz * t_target;
  double lkt_press = kt;
  int ich;
  if (dimension == 3) volume = domain->xprd * domain->yprd * domain->zprd;
  else volume = domain->xprd * domain->yprd;

  if (tstat_flag) {
    ilen = 1;
    if (n < ilen) {
      ich = n;
      if (ich == 0)
        return ke_target * eta[0];
      else
        return kt * eta[ich];
    }
    n -= ilen;
    ilen = 1;
    if (n < ilen) {
      ich = n;
      if (ich == 0)
        return 0.5*eta_mass[0]*eta_dot[0]*eta_dot[0];
      else
        return 0.5*eta_mass[ich]*eta_dot[ich]*eta_dot[ich];
    }
    n -= ilen;
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

    if (1) {
      ilen = 1;
      if (n < ilen) {
        ich = n;
        if (ich == 0) return lkt_press * etap[0];
        else return kt * etap[ich];
      }
      n -= ilen;
      ilen = 1;
      if (n < ilen) {
        ich = n;
        if (ich == 0)
          return 0.5*etap_mass[0]*etap_dot[0]*etap_dot[0];
        else
          return 0.5*etap_mass[ich]*etap_dot[ich]*etap_dot[ich];
      }
      n -= ilen;
    }

    if (deviatoric_flag) {
      ilen = 1;
      if (n < ilen)
        return compute_strain_energy();
      n -= ilen;
    }
  }

  return 0.0;
}

/* ---------------------------------------------------------------------- */

void FixNHL::reset_target(double t_new)
{
  t_target = t_start = t_stop = t_new;
}

/* ---------------------------------------------------------------------- */

void FixNHL::reset_dt()
{
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
  dthalf = 0.5 * update->dt;
  dt4 = 0.25 * update->dt;
  dt8 = 0.125 * update->dt;
  dto = dthalf;

  // If using respa, then remap is performed in innermost level

  if (strstr(update->integrate_style,"respa"))
    dto = 0.5*step_respa[0];

  if (pstat_flag)
    pdrag_factor = 1.0 - (update->dt * p_freq_max * drag / nc_pchain);

  if (tstat_flag)
    tdrag_factor = 1.0 - (update->dt * t_freq * drag / nc_tchain);
}

/* ----------------------------------------------------------------------
   extract thermostat properties
------------------------------------------------------------------------- */

void *FixNHL::extract(const char *str, int &dim)
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
  if (tstat_flag && strcmp(str,"eta") == 0) {
    return &eta;
  } else if (pstat_flag && strcmp(str,"etap") == 0) {
    return &eta;
  } else if (pstat_flag && strcmp(str,"p_flag") == 0) {
    return &p_flag;
  } else if (pstat_flag && strcmp(str,"p_start") == 0) {
    return &p_start;
  } else if (pstat_flag && strcmp(str,"p_stop") == 0) {
    return &p_stop;
  } else if (pstat_flag && strcmp(str,"p_target") == 0) {
    return &p_target;
  }
  return NULL;
}

/* ----------------------------------------------------------------------
   perform half-step update of chain thermostat variables
------------------------------------------------------------------------- */

void FixNHL::nhc_temp_integrate()
{
  double expfac;
  double kecurrent = tdof * boltz * t_current;

  // Update masses, to preserve initial freq, if flag set

  if (eta_mass_flag)
    eta_mass[0] = tdof * boltz * t_target / (t_freq*t_freq);

  if (eta_mass[0] > 0.0)
    eta_dotdot[0] = (kecurrent - ke_target)/eta_mass[0];
  else eta_dotdot[0] = 0.0;

  double ncfac = 1.0/nc_tchain;
  for (int iloop = 0; iloop < nc_tchain; iloop++) {

    expfac = exp(-ncfac*dt8*eta_dot[1]);
    eta_dot[0] *= expfac;
    eta_dot[0] += eta_dotdot[0] * ncfac*dt4;
    eta_dot[0] *= tdrag_factor;
    eta_dot[0] *= expfac;

    factor_eta = exp(-ncfac*dthalf*eta_dot[0]);
    nh_v_temp();

    // rescale temperature due to velocity scaling
    // should not be necessary to explicitly recompute the temperature

    t_current *= factor_eta*factor_eta;
    kecurrent = tdof * boltz * t_current;

    if (eta_mass[0] > 0.0)
      eta_dotdot[0] = (kecurrent - ke_target)/eta_mass[0];
    else eta_dotdot[0] = 0.0;

    eta[0] += ncfac*dthalf*eta_dot[0];

    eta_dot[0] *= expfac;
    eta_dot[0] += eta_dotdot[0] * ncfac*dt4;
    eta_dot[0] *= expfac;
  }
}

/* ----------------------------------------------------------------------
   perform half-step update of chain thermostat variables for barostat
   scale barostat velocities
------------------------------------------------------------------------- */

void FixNHL::nhc_press_integrate()
{
  int i,pdof;
  double expfac,factor_etap,kecurrent;
  double kt = boltz * t_target;
  double lkt_press;

  // Update masses, to preserve initial freq, if flag set

  if (omega_mass_flag) {
    double nkt = (atom->natoms + 1) * kt;
    for (int i = 0; i < 3; i++)
      if (p_flag[i])
        omega_mass[i] = nkt/(p_freq[i]*p_freq[i]);

    if (pstyle == TRICLINIC) {
      for (int i = 3; i < 6; i++)
        if (p_flag[i]) omega_mass[i] = nkt/(p_freq[i]*p_freq[i]);
    }
  }

  if (etap_mass_flag)
    etap_mass[0] = boltz * t_target / (p_freq_max*p_freq_max);

  kecurrent = 0.0;
  pdof = 0;
  for (i = 0; i < 3; i++)
    if (p_flag[i]) {
      kecurrent += omega_mass[i]*omega_dot[i]*omega_dot[i];
      pdof++;
    }

  if (pstyle == TRICLINIC) {
    for (i = 3; i < 6; i++)
      if (p_flag[i]) {
        kecurrent += omega_mass[i]*omega_dot[i]*omega_dot[i];
        pdof++;
      }
  }

  if (pstyle == ISO) lkt_press = kt;
  else lkt_press = pdof * kt;
  etap_dotdot[0] = (kecurrent - lkt_press)/etap_mass[0];

  double ncfac = 1.0/nc_pchain;
  for (int iloop = 0; iloop < nc_pchain; iloop++) {

    expfac = exp(-ncfac*dt8*etap_dot[1]);
    etap_dot[0] *= expfac;
    etap_dot[0] += etap_dotdot[0] * ncfac*dt4;
    etap_dot[0] *= pdrag_factor;
    etap_dot[0] *= expfac;

    etap[0] += ncfac*dthalf*etap_dot[0];

    factor_etap = exp(-ncfac*dthalf*etap_dot[0]);
    for (i = 0; i < 3; i++)
      if (p_flag[i]) omega_dot[i] *= factor_etap;

    if (pstyle == TRICLINIC) {
      for (i = 3; i < 6; i++)
        if (p_flag[i]) omega_dot[i] *= factor_etap;
    }

    kecurrent = 0.0;
    for (i = 0; i < 3; i++)
      if (p_flag[i]) kecurrent += omega_mass[i]*omega_dot[i]*omega_dot[i];

    if (pstyle == TRICLINIC) {
      for (i = 3; i < 6; i++)
        if (p_flag[i]) kecurrent += omega_mass[i]*omega_dot[i]*omega_dot[i];
    }

    etap_dotdot[0] = (kecurrent - lkt_press)/etap_mass[0];

    etap_dot[0] *= expfac;
    etap_dot[0] += etap_dotdot[0] * ncfac*dt4;
    etap_dot[0] *= expfac;
  }
}

/* ----------------------------------------------------------------------
   perform half-step barostat scaling of velocities
-----------------------------------------------------------------------*/

void FixNHL::nh_v_press()
{
  double factor[3];
  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  factor[0] = exp(-dt4*(omega_dot[0]+mtk_term2));
  factor[1] = exp(-dt4*(omega_dot[1]+mtk_term2));
  factor[2] = exp(-dt4*(omega_dot[2]+mtk_term2));

  if (which == NOBIAS) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        v[i][0] *= factor[0];
        v[i][1] *= factor[1];
        v[i][2] *= factor[2];
        if (pstyle == TRICLINIC) {
          v[i][0] += -dthalf*(v[i][1]*omega_dot[5] + v[i][2]*omega_dot[4]);
          v[i][1] += -dthalf*v[i][2]*omega_dot[3];
        }
        v[i][0] *= factor[0];
        v[i][1] *= factor[1];
        v[i][2] *= factor[2];
      }
    }
  } else if (which == BIAS) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        temperature->remove_bias(i,v[i]);
        v[i][0] *= factor[0];
        v[i][1] *= factor[1];
        v[i][2] *= factor[2];
        if (pstyle == TRICLINIC) {
          v[i][0] += -dthalf*(v[i][1]*omega_dot[5] + v[i][2]*omega_dot[4]);
          v[i][1] += -dthalf*v[i][2]*omega_dot[3];
        }
        v[i][0] *= factor[0];
        v[i][1] *= factor[1];
        v[i][2] *= factor[2];
        temperature->restore_bias(i,v[i]);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   perform half-step update of velocities
-----------------------------------------------------------------------*/

void FixNHL::nve_v()
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

void FixNHL::nve_x()
{
  double **x = atom->x;
  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  // x update by full step only for atoms in group

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      x[i][0] += dtv * v[i][0];
      x[i][1] += dtv * v[i][1];
      x[i][2] += dtv * v[i][2];
    }
  }
}

/* ----------------------------------------------------------------------
   perform half-step thermostat scaling of velocities
-----------------------------------------------------------------------*/

void FixNHL::nh_v_temp()
{
  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  if (which == NOBIAS) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        v[i][0] *= factor_eta;
        v[i][1] *= factor_eta;
        v[i][2] *= factor_eta;
      }
    }
  } else if (which == BIAS) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        temperature->remove_bias(i,v[i]);
        v[i][0] *= factor_eta;
        v[i][1] *= factor_eta;
        v[i][2] *= factor_eta;
        temperature->restore_bias(i,v[i]);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   compute sigma tensor
   needed whenever p_target or h0_inv changes
-----------------------------------------------------------------------*/

void FixNHL::compute_sigma()
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

double FixNHL::compute_strain_energy()
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

void FixNHL::compute_deviatoric()
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

void FixNHL::compute_temp_target()
{
  double delta = update->ntimestep - update->beginstep;
  if (delta != 0.0) delta /= update->endstep - update->beginstep;

  t_target = t_start + delta * (t_stop-t_start);
  ke_target = tdof * boltz * t_target;
}

/* ----------------------------------------------------------------------
   compute hydrostatic target pressure
-----------------------------------------------------------------------*/

void FixNHL::compute_press_target()
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

void FixNHL::nh_omega_dot()
{
  double f_omega,volume;

  if (dimension == 3) volume = domain->xprd*domain->yprd*domain->zprd;
  else volume = domain->xprd*domain->yprd;

  if (deviatoric_flag) compute_deviatoric();

  mtk_term1 = 0.0;
  if (mtk_flag) {
    if (pstyle == ISO) {
      mtk_term1 = tdof * boltz * t_current;
      mtk_term1 /= pdim * atom->natoms;
    } else {
      double *mvv_current = temperature->vector;
      for (int i = 0; i < 3; i++)
        if (p_flag[i])
          mtk_term1 += mvv_current[i];
      mtk_term1 /= pdim * atom->natoms;
    }
  }

  for (int i = 0; i < 3; i++)
    if (p_flag[i]) {
      f_omega = (p_current[i]-p_hydro)*volume /
        (omega_mass[i] * nktv2p) + mtk_term1 / omega_mass[i];
      if (deviatoric_flag) f_omega -= fdev[i]/(omega_mass[i] * nktv2p);
      omega_dot[i] += f_omega*dthalf;
      omega_dot[i] *= pdrag_factor;
    }

  mtk_term2 = 0.0;
  if (mtk_flag) {
    for (int i = 0; i < 3; i++)
      if (p_flag[i])
        mtk_term2 += omega_dot[i];
    if (pdim > 0) mtk_term2 /= pdim * atom->natoms;
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
  remap() puts atoms outside the new box back into the new box
  perform irregular on atoms in lamda coords to migrate atoms to new procs
  important that image_flip comes before remap, since remap may change
    image flags to new values, making eqs in doc of Domain:image_flip incorrect
------------------------------------------------------------------------- */

void FixNHL::pre_exchange()
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
   memory usage of Irregular
------------------------------------------------------------------------- */

double FixNHL::memory_usage()
{
  double bytes = 0.0;
  if (irregular) bytes += irregular->memory_usage();
  return bytes;
}
