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
   Massive Thermostating for NVT and NPT simulations
   Contributing author: Charlles Abreu (UFRJ)
   Adapted from FixNH (authors: Mark Stevens and Aidan Thompson - SNL)
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
TO DO LIST:
-------------------------------------------------------------------------
1. Implement restart management involving random number generator
2. Include keyword "molecular yes/no" to enable barostatting with molecular
   rather than atomic pressure.
3. Implement alternative barostat with fixed temperature and include keyword
   "mkt yes/no" (default=no) to enable original MTK barostat.
4. Implement isokinetic constraints and include keyword "speedlim" with
   options "none/isok/geneq", where isok stands for isokinetic and geneq
   stands for generalized equipartition.
------------------------------------------------------------------------- */

#include "fix_nh_massive.h"
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
#include "random_mars.h"

using namespace LAMMPS_NS;
using namespace FixConst;

#define DELTAFLIP 0.1
#define TILTMAX 1.5

enum{NONE,XYZ,XY,YZ,XZ};
enum{ISO,ANISO,TRICLINIC};

/* ----------------------------------------------------------------------
   NVT,NPT integrators for massive thermostatting equations of motion
 ---------------------------------------------------------------------- */

FixNHMassive::FixNHMassive(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  rfix(NULL), id_dilate(NULL), irregular(NULL), id_temp(NULL), id_press(NULL)
{
  if (narg < 4) error->all(FLERR,"Illegal fix <ensemble>/massive command");

  restart_global = 1;
  dynamic_group_allow = 1;
  time_integrate = 1;

  // default values

  pcouple = NONE;
  allremap = 1;
  id_dilate = NULL;
  deviatoric_flag = 0;
  nreset_h0 = 0;
  eta_mass_flag = 1;
  omega_mass_flag = 0;
  etap_mass_flag = 0;
  flipflag = 1;
  dipole_flag = 0;
  dlm_flag = 0;
  langevin_flag = 0;
  gamma_langevin = 0.0;

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

  int tstat_flag = 0;
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
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix <ensemble>/massive command");
      tstat_flag = 1;
      t_start = force->numeric(FLERR,arg[iarg+1]);
      t_target = t_start;
      t_stop = force->numeric(FLERR,arg[iarg+2]);
      t_period = force->numeric(FLERR,arg[iarg+3]);
      if (t_start <= 0.0 || t_stop <= 0.0)
        error->all(FLERR,
                   "Target temperature for fix <ensemble>/massive cannot be 0.0");
      iarg += 4;

    } else if (strcmp(arg[iarg],"iso") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix <ensemble>/massive command");
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
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix <ensemble>/massive command");
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
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix <ensemble>/massive command");
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
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix <ensemble>/massive command");
      p_start[0] = force->numeric(FLERR,arg[iarg+1]);
      p_stop[0] = force->numeric(FLERR,arg[iarg+2]);
      p_period[0] = force->numeric(FLERR,arg[iarg+3]);
      p_flag[0] = 1;
      deviatoric_flag = 1;
      iarg += 4;
    } else if (strcmp(arg[iarg],"y") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix <ensemble>/massive command");
      p_start[1] = force->numeric(FLERR,arg[iarg+1]);
      p_stop[1] = force->numeric(FLERR,arg[iarg+2]);
      p_period[1] = force->numeric(FLERR,arg[iarg+3]);
      p_flag[1] = 1;
      deviatoric_flag = 1;
      iarg += 4;
    } else if (strcmp(arg[iarg],"z") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix <ensemble>/massive command");
      p_start[2] = force->numeric(FLERR,arg[iarg+1]);
      p_stop[2] = force->numeric(FLERR,arg[iarg+2]);
      p_period[2] = force->numeric(FLERR,arg[iarg+3]);
      p_flag[2] = 1;
      deviatoric_flag = 1;
      iarg += 4;
      if (dimension == 2)
        error->all(FLERR,"Invalid fix <ensemble>/massive command for a 2d simulation");

    } else if (strcmp(arg[iarg],"yz") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix <ensemble>/massive command");
      p_start[3] = force->numeric(FLERR,arg[iarg+1]);
      p_stop[3] = force->numeric(FLERR,arg[iarg+2]);
      p_period[3] = force->numeric(FLERR,arg[iarg+3]);
      p_flag[3] = 1;
      deviatoric_flag = 1;
      scaleyz = 0;
      iarg += 4;
      if (dimension == 2)
        error->all(FLERR,"Invalid fix <ensemble>/massive command for a 2d simulation");
    } else if (strcmp(arg[iarg],"xz") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix <ensemble>/massive command");
      p_start[4] = force->numeric(FLERR,arg[iarg+1]);
      p_stop[4] = force->numeric(FLERR,arg[iarg+2]);
      p_period[4] = force->numeric(FLERR,arg[iarg+3]);
      p_flag[4] = 1;
      deviatoric_flag = 1;
      scalexz = 0;
      iarg += 4;
      if (dimension == 2)
        error->all(FLERR,"Invalid fix <ensemble>/massive command for a 2d simulation");
    } else if (strcmp(arg[iarg],"xy") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix <ensemble>/massive command");
      p_start[5] = force->numeric(FLERR,arg[iarg+1]);
      p_stop[5] = force->numeric(FLERR,arg[iarg+2]);
      p_period[5] = force->numeric(FLERR,arg[iarg+3]);
      p_flag[5] = 1;
      deviatoric_flag = 1;
      scalexy = 0;
      iarg += 4;

    } else if (strcmp(arg[iarg],"couple") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix <ensemble>/massive command");
      if (strcmp(arg[iarg+1],"xyz") == 0) pcouple = XYZ;
      else if (strcmp(arg[iarg+1],"xy") == 0) pcouple = XY;
      else if (strcmp(arg[iarg+1],"yz") == 0) pcouple = YZ;
      else if (strcmp(arg[iarg+1],"xz") == 0) pcouple = XZ;
      else if (strcmp(arg[iarg+1],"none") == 0) pcouple = NONE;
      else error->all(FLERR,"Illegal fix <ensemble>/massive command");
      iarg += 2;

    } else if (strcmp(arg[iarg],"dilate") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix <ensemble>/massive command");
      if (strcmp(arg[iarg+1],"all") == 0) allremap = 1;
      else {
        allremap = 0;
        delete [] id_dilate;
        int n = strlen(arg[iarg+1]) + 1;
        id_dilate = new char[n];
        strcpy(id_dilate,arg[iarg+1]);
        int idilate = group->find(id_dilate);
        if (idilate == -1)
          error->all(FLERR,"Fix <ensemble>/massive dilate group ID does not exist");
      }
      iarg += 2;

    } else if (strcmp(arg[iarg],"nreset") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix <ensemble>/massive command");
      nreset_h0 = force->inumeric(FLERR,arg[iarg+1]);
      if (nreset_h0 < 0) error->all(FLERR,"Illegal fix <ensemble>/massive command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"scalexy") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix <ensemble>/massive command");
      if (strcmp(arg[iarg+1],"yes") == 0) scalexy = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) scalexy = 0;
      else error->all(FLERR,"Illegal fix <ensemble>/massive command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"scalexz") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix <ensemble>/massive command");
      if (strcmp(arg[iarg+1],"yes") == 0) scalexz = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) scalexz = 0;
      else error->all(FLERR,"Illegal fix <ensemble>/massive command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"scaleyz") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix <ensemble>/massive command");
      if (strcmp(arg[iarg+1],"yes") == 0) scaleyz = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) scaleyz = 0;
      else error->all(FLERR,"Illegal fix <ensemble>/massive command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"flip") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix <ensemble>/massive command");
      if (strcmp(arg[iarg+1],"yes") == 0) flipflag = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) flipflag = 0;
      else error->all(FLERR,"Illegal fix <ensemble>/massive command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"update") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix <ensemble>/massive command");
      if (strcmp(arg[iarg+1],"dipole") == 0) dipole_flag = 1;
      else if (strcmp(arg[iarg+1],"dipole/dlm") == 0) {
        dipole_flag = 1;
        dlm_flag = 1;
      } else error->all(FLERR,"Illegal fix <ensemble>/massive command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"fixedpoint") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix <ensemble>/massive command");
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

    } else if (strcmp(arg[iarg],"langevin") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal fix <ensemble>/massive command");
      langevin_flag = 1;
      int seed = force->inumeric(FLERR,arg[iarg+1]);
      random = new RanMars(lmp, seed + comm->me);
      double damp = force->numeric(FLERR,arg[iarg+2]);
      gamma_langevin = 1.0/damp;
      iarg += 3;

    } else error->all(FLERR,"Illegal fix <ensemble>/massive command");
  }

  if (!tstat_flag)
    error->all(FLERR,"Temperature control must be used with fix <ensemble>/massive");

  // error checks

  if (dimension == 2 && (p_flag[2] || p_flag[3] || p_flag[4]))
    error->all(FLERR,"Invalid fix <ensemble>/massive command for a 2d simulation");
  if (dimension == 2 && (pcouple == YZ || pcouple == XZ))
    error->all(FLERR,"Invalid fix <ensemble>/massive command for a 2d simulation");
  if (dimension == 2 && (scalexz == 1 || scaleyz == 1 ))
    error->all(FLERR,"Invalid fix <ensemble>/massive command for a 2d simulation");

  if (pcouple == XYZ && (p_flag[0] == 0 || p_flag[1] == 0))
    error->all(FLERR,"Invalid fix <ensemble>/massive command pressure settings");
  if (pcouple == XYZ && dimension == 3 && p_flag[2] == 0)
    error->all(FLERR,"Invalid fix <ensemble>/massive command pressure settings");
  if (pcouple == XY && (p_flag[0] == 0 || p_flag[1] == 0))
    error->all(FLERR,"Invalid fix <ensemble>/massive command pressure settings");
  if (pcouple == YZ && (p_flag[1] == 0 || p_flag[2] == 0))
    error->all(FLERR,"Invalid fix <ensemble>/massive command pressure settings");
  if (pcouple == XZ && (p_flag[0] == 0 || p_flag[2] == 0))
    error->all(FLERR,"Invalid fix <ensemble>/massive command pressure settings");

  // require periodicity in tensile dimension

  if (p_flag[0] && domain->xperiodic == 0)
    error->all(FLERR,"Cannot use fix <ensemble>/massive on a non-periodic dimension");
  if (p_flag[1] && domain->yperiodic == 0)
    error->all(FLERR,"Cannot use fix <ensemble>/massive on a non-periodic dimension");
  if (p_flag[2] && domain->zperiodic == 0)
    error->all(FLERR,"Cannot use fix <ensemble>/massive on a non-periodic dimension");

  // require periodicity in 2nd dim of off-diagonal tilt component

  if (p_flag[3] && domain->zperiodic == 0)
    error->all(FLERR,
               "Cannot use fix <ensemble>/massive on a 2nd non-periodic dimension");
  if (p_flag[4] && domain->zperiodic == 0)
    error->all(FLERR,
               "Cannot use fix <ensemble>/massive on a 2nd non-periodic dimension");
  if (p_flag[5] && domain->yperiodic == 0)
    error->all(FLERR,
               "Cannot use fix <ensemble>/massive on a 2nd non-periodic dimension");

  if (scaleyz == 1 && domain->zperiodic == 0)
    error->all(FLERR,"Cannot use fix <ensemble>/massive "
               "with yz scaling when z is non-periodic dimension");
  if (scalexz == 1 && domain->zperiodic == 0)
    error->all(FLERR,"Cannot use fix <ensemble>/massive "
               "with xz scaling when z is non-periodic dimension");
  if (scalexy == 1 && domain->yperiodic == 0)
    error->all(FLERR,"Cannot use fix <ensemble>/massive "
               "with xy scaling when y is non-periodic dimension");

  if (p_flag[3] && scaleyz == 1)
    error->all(FLERR,"Cannot use fix <ensemble>/massive with "
               "both yz dynamics and yz scaling");
  if (p_flag[4] && scalexz == 1)
    error->all(FLERR,"Cannot use fix <ensemble>/massive with "
               "both xz dynamics and xz scaling");
  if (p_flag[5] && scalexy == 1)
    error->all(FLERR,"Cannot use fix <ensemble>/massive with "
               "both xy dynamics and xy scaling");

  if (!domain->triclinic && (p_flag[3] || p_flag[4] || p_flag[5]))
    error->all(FLERR,"Can not specify Pxy/Pxz/Pyz in "
               "fix <ensemble>/massive with non-triclinic box");

  if (pcouple == XYZ && dimension == 3 &&
      (p_start[0] != p_start[1] || p_start[0] != p_start[2] ||
       p_stop[0] != p_stop[1] || p_stop[0] != p_stop[2] ||
       p_period[0] != p_period[1] || p_period[0] != p_period[2]))
    error->all(FLERR,"Invalid fix <ensemble>/massive pressure settings");
  if (pcouple == XYZ && dimension == 2 &&
      (p_start[0] != p_start[1] || p_stop[0] != p_stop[1] ||
       p_period[0] != p_period[1]))
    error->all(FLERR,"Invalid fix <ensemble>/massive pressure settings");
  if (pcouple == XY &&
      (p_start[0] != p_start[1] || p_stop[0] != p_stop[1] ||
       p_period[0] != p_period[1]))
    error->all(FLERR,"Invalid fix <ensemble>/massive pressure settings");
  if (pcouple == YZ &&
      (p_start[1] != p_start[2] || p_stop[1] != p_stop[2] ||
       p_period[1] != p_period[2]))
    error->all(FLERR,"Invalid fix <ensemble>/massive pressure settings");
  if (pcouple == XZ &&
      (p_start[0] != p_start[2] || p_stop[0] != p_stop[2] ||
       p_period[0] != p_period[2]))
    error->all(FLERR,"Invalid fix <ensemble>/massive pressure settings");

  if (dipole_flag) {
    if (!atom->sphere_flag)
      error->all(FLERR,"Using update dipole flag requires atom style sphere");
    if (!atom->mu_flag)
      error->all(FLERR,"Using update dipole flag requires atom attribute mu");
  }

  if ((t_period <= 0.0) ||
      (p_flag[0] && p_period[0] <= 0.0) ||
      (p_flag[1] && p_period[1] <= 0.0) ||
      (p_flag[2] && p_period[2] <= 0.0) ||
      (p_flag[3] && p_period[3] <= 0.0) ||
      (p_flag[4] && p_period[4] <= 0.0) ||
      (p_flag[5] && p_period[5] <= 0.0))
    error->all(FLERR,"Fix <ensemble>/massive damping parameters must be > 0.0");

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

  t_freq = 1.0 / t_period;
  if (p_flag[0]) p_freq[0] = 1.0 / p_period[0];
  if (p_flag[1]) p_freq[1] = 1.0 / p_period[1];
  if (p_flag[2]) p_freq[2] = 1.0 / p_period[2];
  if (p_flag[3]) p_freq[3] = 1.0 / p_period[3];
  if (p_flag[4]) p_freq[4] = 1.0 / p_period[4];
  if (p_flag[5]) p_freq[5] = 1.0 / p_period[5];

  // Nose/Hoover temp and pressure init

  eta_dot = NULL;
  grow_arrays(atom->nmax);
  atom->add_callback(0);
  for (int i = 0; i < atom->nmax; i++)
    eta_dot[i][0] = eta_dot[i][1] = eta_dot[i][2] = 0.0;

  if (pstat_flag) {
    omega_dot[0] = omega_dot[1] = omega_dot[2] = 0.0;
    omega_mass[0] = omega_mass[1] = omega_mass[2] = 0.0;
    etap_dot[0] = etap_dot[1] = etap_dot[2] = 0.0;
    omega_dot[3] = omega_dot[4] = omega_dot[5] = 0.0;
    omega_mass[3] = omega_mass[4] = omega_mass[5] = 0.0;
    etap_dot[3] = etap_dot[4] = etap_dot[5] = 0.0;
  }

  nrigid = 0;
  rfix = NULL;

  if (pre_exchange_flag)
    irregular = new Irregular(lmp);
  else
    irregular = NULL;

  // initialize vol0,t0 to zero to signal uninitialized
  // values then assigned in init(), if necessary

  vol0 = t0 = 0.0;
}

/* ---------------------------------------------------------------------- */

FixNHMassive::~FixNHMassive()
{
  if (copymode) return;

  delete [] id_dilate;
  delete [] rfix;

  delete irregular;

  memory->destroy(eta_dot);
  atom->delete_callback(id,0);

  // delete temperature and pressure if fix created them

  if (tcomputeflag) modify->delete_compute(id_temp);
  delete [] id_temp;

  if (pstat_flag) {
    if (pcomputeflag) modify->delete_compute(id_press);
    delete [] id_press;
  }
}

/* ---------------------------------------------------------------------- */

int FixNHMassive::setmask()
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

void FixNHMassive::init()
{
  // recheck that dilate group has not been deleted

  if (allremap == 0) {
    int idilate = group->find(id_dilate);
    if (idilate == -1)
      error->all(FLERR,"Fix <ensemble>/massive dilate group ID does not exist");
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

  if (pstat_flag) {
    icompute = modify->find_compute(id_press);
    if (icompute < 0)
      error->all(FLERR,"Pressure ID for fix npt does not exist");
    pressure = modify->compute[icompute];
  }

  // set timesteps and frequencies

  dtfull = update->dt;
  dthalf = 0.5 * update->dt;

  p_freq_max = 0.0;
  if (pstat_flag) {
    p_freq_max = MAX(p_freq[0],p_freq[1]);
    p_freq_max = MAX(p_freq_max,p_freq[2]);
    if (pstyle == TRICLINIC) {
      p_freq_max = MAX(p_freq_max,p_freq[3]);
      p_freq_max = MAX(p_freq_max,p_freq[4]);
      p_freq_max = MAX(p_freq_max,p_freq[5]);
    }
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

  if (force->kspace) kspace_flag = 1;
  else kspace_flag = 0;

  if (strstr(update->integrate_style,"respa")) {
    nlevels_respa = ((Respa *) update->integrate)->nlevels;
    step_respa = ((Respa *) update->integrate)->step;
  }

  // detect if any rigid fixes exist so rigid bodies move when box is remapped
  // rfix[] = indices to each fix rigid

  delete [] rfix;
  nrigid = 0;
  rfix = NULL;

  for (int i = 0; i < modify->nfix; i++)
    if (modify->fix[i]->rigid_flag) nrigid++;
  if (nrigid) {
    error->all(FLERR,"Pressure ID for fix npt/nph does not exist");
    rfix = new int[nrigid];
    nrigid = 0;
    for (int i = 0; i < modify->nfix; i++)
      if (modify->fix[i]->rigid_flag) rfix[nrigid++] = i;
  }

  // Rigid bodies and constraints can be considered in future versions
  // For the time being, just throw an error message

  if (nrigid)
    error->all(FLERR,"No support for rigid bodies in fix <ensemble>/nph");
  for (int i = 0; i < modify->nfix; i++)
    if (strcmp(modify->fix[i]->style,"shake") == 0 ||
        strcmp(modify->fix[i]->style,"rattle") == 0)
      error->all(FLERR,"No support for constraints in fix <ensemble>/nph");
}

/* ----------------------------------------------------------------------
   compute T,P before integrator starts
------------------------------------------------------------------------- */

void FixNHMassive::setup(int /*vflag*/)
{
  // tdof needed by compute_temp_target()

  t_current = temperature->compute_scalar();
  tdof = temperature->dof;

  // t_target is needed by NVT and NPT in compute_scalar()
  // If no thermostat or using fix nphug,
  // t_target must be defined by other means.

  if (strstr(style,"nphug") == NULL) {
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

  eta_mass = boltz * t_target / (t_freq*t_freq);

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

    etap_mass = boltz * t_target / (p_freq_max*p_freq_max);
  }
}

/* ----------------------------------------------------------------------
   1st half of Verlet update
------------------------------------------------------------------------- */

void FixNHMassive::initial_integrate(int /*vflag*/)
{
  compute_temp_target();

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
    nh_omega_dot(dthalf);
    nh_v_press(dthalf);
  }

  nve_v(dthalf);

  // remap simulation box by 1/2 step

  if (pstat_flag) remap(dthalf);

  nve_x(dthalf);
  nh_temp_integrate(dtfull);
  if (pstat_flag) nh_press_integrate(dtfull);
  nve_x(dthalf);

  // remap simulation box by 1/2 step
  // redo KSpace coeffs since volume has changed

  if (pstat_flag) {
    remap(dthalf);
    if (kspace_flag) force->kspace->setup();
  }
}

/* ----------------------------------------------------------------------
   2nd half of Verlet update
------------------------------------------------------------------------- */

void FixNHMassive::final_integrate()
{
  nve_v(dthalf);

  if (pstat_flag) nh_v_press(dthalf);

  // compute new T,P after velocities rescaled by nh_v_press(dthalf)
  // compute appropriately coupled elements of mvv_current

  t_current = temperature->compute_scalar();
  tdof = temperature->dof;

  // need to recompute pressure to account for change in KE
  // t_current is up-to-date, but compute_temperature is not
  // compute appropriately coupled elements of mvv_current

  if (pstat_flag) {
    if (pstyle == ISO)
      pressure->compute_scalar();
    else {
      temperature->compute_vector();
      pressure->compute_vector();
    }
    couple();
    pressure->addstep(update->ntimestep+1);
  }

  if (pstat_flag) nh_omega_dot(dthalf);
}

/* ---------------------------------------------------------------------- */

void FixNHMassive::initial_integrate_respa(int /*vflag*/, int ilevel, int /*iloop*/)
{
  // set timesteps by level

  dtfull = step_respa[ilevel];
  dthalf = 0.5 * step_respa[ilevel];

  // outermost level - update eta_dot and omega_dot, apply to v
  // all other levels - NVE update of v
  // x,v updates only performed for atoms in group

  if (ilevel == nlevels_respa-1) {

    // update eta_dot

    compute_temp_target();

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
      nh_omega_dot(dthalf);
      nh_v_press(dthalf);
    }
  }

  nve_v(dthalf);

  // innermost level - also update x only for atoms in group
  // if barostat, perform 1/2 step remap before and after

  if (ilevel == 0) {
    if (pstat_flag) remap(dthalf);
    nve_x(dthalf);
    nh_temp_integrate(dtfull);
    if (pstat_flag) nh_press_integrate(dtfull);
    nve_x(dthalf);
    if (pstat_flag) remap(dthalf);
  }

  // if barostat, redo KSpace coeffs at outermost level,
  // since volume has changed

  if (ilevel == nlevels_respa-1 && kspace_flag && pstat_flag)
    force->kspace->setup();
}

/* ---------------------------------------------------------------------- */

void FixNHMassive::final_integrate_respa(int ilevel, int /*iloop*/)
{
  // set timesteps by level

  dthalf = 0.5 * step_respa[ilevel];

  // outermost level - update eta_dot and omega_dot, apply via final_integrate
  // all other levels - NVE update of v

  if (ilevel == nlevels_respa-1)
    final_integrate();
  else
    nve_v(dthalf);
}

/* ---------------------------------------------------------------------- */

void FixNHMassive::couple()
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

void FixNHMassive::remap(double dt)
{
  int i;
  double oldlo,oldhi;
  double expfac;

  double **x = atom->x;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  double *h = domain->h;

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

  double dt2 = dt/2.0;
  double dt4 = dt/4.0;
  double dt8 = dt/8.0;

  // off-diagonal components, first half

  if (pstyle == TRICLINIC) {

    if (p_flag[4]) {
      expfac = exp(dt8*omega_dot[0]);
      h[4] *= expfac;
      h[4] += dt4*(omega_dot[5]*h[3]+omega_dot[4]*h[2]);
      h[4] *= expfac;
    }

    if (p_flag[3]) {
      expfac = exp(dt4*omega_dot[1]);
      h[3] *= expfac;
      h[3] += dt2*(omega_dot[3]*h[2]);
      h[3] *= expfac;
    }

    if (p_flag[5]) {
      expfac = exp(dt4*omega_dot[0]);
      h[5] *= expfac;
      h[5] += dt2*(omega_dot[5]*h[1]);
      h[5] *= expfac;
    }

    if (p_flag[4]) {
      expfac = exp(dt8*omega_dot[0]);
      h[4] *= expfac;
      h[4] += dt4*(omega_dot[5]*h[3]+omega_dot[4]*h[2]);
      h[4] *= expfac;
    }
  }

  // scale diagonal components
  // scale tilt factors with cell, if set

  if (p_flag[0]) {
    oldlo = domain->boxlo[0];
    oldhi = domain->boxhi[0];
    expfac = exp(dt*omega_dot[0]);
    domain->boxlo[0] = (oldlo-fixedpoint[0])*expfac + fixedpoint[0];
    domain->boxhi[0] = (oldhi-fixedpoint[0])*expfac + fixedpoint[0];
  }

  if (p_flag[1]) {
    oldlo = domain->boxlo[1];
    oldhi = domain->boxhi[1];
    expfac = exp(dt*omega_dot[1]);
    domain->boxlo[1] = (oldlo-fixedpoint[1])*expfac + fixedpoint[1];
    domain->boxhi[1] = (oldhi-fixedpoint[1])*expfac + fixedpoint[1];
    if (scalexy) h[5] *= expfac;
  }

  if (p_flag[2]) {
    oldlo = domain->boxlo[2];
    oldhi = domain->boxhi[2];
    expfac = exp(dt*omega_dot[2]);
    domain->boxlo[2] = (oldlo-fixedpoint[2])*expfac + fixedpoint[2];
    domain->boxhi[2] = (oldhi-fixedpoint[2])*expfac + fixedpoint[2];
    if (scalexz) h[4] *= expfac;
    if (scaleyz) h[3] *= expfac;
  }

  // off-diagonal components, second half

  if (pstyle == TRICLINIC) {

    if (p_flag[4]) {
      expfac = exp(dt8*omega_dot[0]);
      h[4] *= expfac;
      h[4] += dt4*(omega_dot[5]*h[3]+omega_dot[4]*h[2]);
      h[4] *= expfac;
    }

    if (p_flag[3]) {
      expfac = exp(dt4*omega_dot[1]);
      h[3] *= expfac;
      h[3] += dt2*(omega_dot[3]*h[2]);
      h[3] *= expfac;
    }

    if (p_flag[5]) {
      expfac = exp(dt4*omega_dot[0]);
      h[5] *= expfac;
      h[5] += dt2*(omega_dot[5]*h[1]);
      h[5] *= expfac;
    }

    if (p_flag[4]) {
      expfac = exp(dt8*omega_dot[0]);
      h[4] *= expfac;
      h[4] += dt4*(omega_dot[5]*h[3]+omega_dot[4]*h[2]);
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

void FixNHMassive::write_restart(FILE *fp)
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

int FixNHMassive::size_restart_global()
{
  int nsize = 1;
  nsize += 1 + 3*atom->nlocal;
  if (pstat_flag) {
    nsize += 15;
    if (deviatoric_flag) nsize += 6;
  }
  return nsize;
}

/* ----------------------------------------------------------------------
   pack restart data
------------------------------------------------------------------------- */

int FixNHMassive::pack_restart_data(double *list)
{
  int n = 0;

  list[n++] = atom->nlocal;
  for (int i = 0; i < atom->nlocal; i++) {
    list[n++] = eta_dot[i][0];
    list[n++] = eta_dot[i][1];
    list[n++] = eta_dot[i][2];
  }

  list[n++] = pstat_flag;
  if (pstat_flag) {
    list[n++] = omega_dot[0];
    list[n++] = omega_dot[1];
    list[n++] = omega_dot[2];
    list[n++] = omega_dot[3];
    list[n++] = omega_dot[4];
    list[n++] = omega_dot[5];
    list[n++] = vol0;
    list[n++] = t0;
    list[n++] = etap_dot[0];
    list[n++] = etap_dot[1];
    list[n++] = etap_dot[2];
    list[n++] = etap_dot[3];
    list[n++] = etap_dot[4];
    list[n++] = etap_dot[5];

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

void FixNHMassive::restart(char *buf)
{
  int n = 0;
  double *list = (double *) buf;
  int nlocal = static_cast<int> (list[n++]);
  for (int i; i < nlocal; i++) {
    eta_dot[i][0] = list[n++];
    eta_dot[i][1] = list[n++];
    eta_dot[i][2] = list[n++];
  }
  int flag = static_cast<int> (list[n++]);
  if (flag) {
    omega_dot[0] = list[n++];
    omega_dot[1] = list[n++];
    omega_dot[2] = list[n++];
    omega_dot[3] = list[n++];
    omega_dot[4] = list[n++];
    omega_dot[5] = list[n++];
    vol0 = list[n++];
    t0 = list[n++];
    if (pstat_flag) {
      etap_dot[0] = list[n++];
      etap_dot[1] = list[n++];
      etap_dot[2] = list[n++];
      etap_dot[3] = list[n++];
      etap_dot[4] = list[n++];
      etap_dot[5] = list[n++];
    }
    else
      n += 6;
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

int FixNHMassive::modify_param(int narg, char **arg)
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

void FixNHMassive::reset_target(double t_new)
{
  t_target = t_start = t_stop = t_new;
}

/* ---------------------------------------------------------------------- */

void FixNHMassive::reset_dt()
{
  dtfull = update->dt;
  dthalf = 0.5 * update->dt;
}

/* ----------------------------------------------------------------------
   extract thermostat properties
------------------------------------------------------------------------- */

void *FixNHMassive::extract(const char *str, int &dim)
{
  dim=0;
  if (strcmp(str,"t_target") == 0) {
    return &t_target;
  } else if (strcmp(str,"t_start") == 0) {
    return &t_start;
  } else if (strcmp(str,"t_stop") == 0) {
    return &t_stop;
  }
  dim=1;
  if (pstat_flag && strcmp(str,"p_flag") == 0) {
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
   perform update of thermostat variables
------------------------------------------------------------------------- */

void FixNHMassive::nh_temp_integrate(double dt)
{
  double kT = boltz * t_target;

  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  double *rmass = atom->rmass;
  double *tmass = atom->mass;
  int *type = atom->type;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  // Update masses, to preserve initial freq, if flag set
  if (eta_mass_flag) eta_mass = kT / (t_freq*t_freq);

  double dt2 = 0.5*dt;
  double dt2m = dt2/eta_mass;
  if (langevin_flag) {
    double factor = exp(-gamma_langevin*dt);
    double sigma = t_freq*sqrt(1.0 - factor*factor);
    for (int i = 0; i < atom->nlocal; i ++)
      if (mask[i] & groupbit) {
        double mass = rmass ? rmass[i] : tmass[type[i]];
        for (int j = 0; j < 3; j++) {
          eta_dot[i][j] += (mass*v[i][j]*v[i][j] - kT)*dt2m;
          v[i][j] *= exp(-dt2*eta_dot[i][j]);
          eta_dot[i][j] = factor*eta_dot[i][j] + sigma*random->gaussian();
          v[i][j] *= exp(-dt2*eta_dot[i][j]);
          eta_dot[i][j] += (mass*v[i][j]*v[i][j] - kT)*dt2m;
        }
      }
  }
  else {
    for (int i = 0; i < atom->nlocal; i ++)
      if (mask[i] & groupbit) {
        double mass = rmass ? rmass[i] : tmass[type[i]];
        for (int j = 0; j < 3; j++) {
          eta_dot[i][j] += (mass*v[i][j]*v[i][j] - kT)*dt2m;
          v[i][j] *= exp(-dt*eta_dot[i][j]);
          eta_dot[i][j] += (mass*v[i][j]*v[i][j] - kT)*dt2m;
        }
      }
  }
}

/* ----------------------------------------------------------------------
   perform update of thermostat variables for barostat
   scale barostat velocities
------------------------------------------------------------------------- */

void FixNHMassive::nh_press_integrate(double dt)
{
  int i;
  double kt = boltz * t_target;
  int number = pstyle == TRICLINIC ? 6 : 3;

  // Update masses, to preserve initial freq, if flag set
  if (omega_mass_flag) {
    double nkt = (atom->natoms + 1) * kt;
    for (int i = 0; i < number; i++)
      if (p_flag[i])
        omega_mass[i] = nkt/(p_freq[i]*p_freq[i]);
  }

  if (etap_mass_flag)
    etap_mass = kt / (p_freq_max*p_freq_max);

  double dt2 = 0.5*dt;
  double dt2m = dt2/etap_mass;
  if (langevin_flag) {
    double factor = exp(-gamma_langevin*dt);
    double sigma = p_freq_max*sqrt(1.0 - factor*factor);
    for (i = 0; i < number; i++)
      if (p_flag[i]) {
        etap_dot[i] += (omega_mass[i]*omega_dot[i]*omega_dot[i] - kt)*dt2m;
        omega_dot[i] *= exp(-dt2*etap_dot[i]);
        etap_dot[i] = factor*etap_dot[i] + sigma*random->gaussian();
        omega_dot[i] *= exp(-dt2*etap_dot[i]);
        etap_dot[i] += (omega_mass[i]*omega_dot[i]*omega_dot[i] - kt)*dt2m;
      }
  }
  else {
    for (i = 0; i < number; i++)
      if (p_flag[i]) {
        etap_dot[i] += (omega_mass[i]*omega_dot[i]*omega_dot[i] - kt)*dt2m;
        omega_dot[i] *= exp(-dt*etap_dot[i]);
        etap_dot[i] += (omega_mass[i]*omega_dot[i]*omega_dot[i] - kt)*dt2m;
      }
  }
}

/* ----------------------------------------------------------------------
   perform barostat scaling of velocities
-----------------------------------------------------------------------*/

void FixNHMassive::nh_v_press(double dt)
{
  double dthalf = 0.5*dt;
  double factor[3];
  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  factor[0] = exp(-dthalf*(omega_dot[0]+mtk_term2));
  factor[1] = exp(-dthalf*(omega_dot[1]+mtk_term2));
  factor[2] = exp(-dthalf*(omega_dot[2]+mtk_term2));

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      v[i][0] *= factor[0];
      v[i][1] *= factor[1];
      v[i][2] *= factor[2];
      if (pstyle == TRICLINIC) {
        v[i][0] += -dt*(v[i][1]*omega_dot[5] + v[i][2]*omega_dot[4]);
        v[i][1] += -dt*v[i][2]*omega_dot[3];
      }
      v[i][0] *= factor[0];
      v[i][1] *= factor[1];
      v[i][2] *= factor[2];
    }
  }
}

/* ----------------------------------------------------------------------
   perform update of velocities
-----------------------------------------------------------------------*/

void FixNHMassive::nve_v(double dt)
{
  double dtf = dt * force->ftm2v;
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
   perform update of positions
-----------------------------------------------------------------------*/

void FixNHMassive::nve_x(double dt)
{
  double **x = atom->x;
  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  // x update by full step only for atoms in group

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      x[i][0] += dt * v[i][0];
      x[i][1] += dt * v[i][1];
      x[i][2] += dt * v[i][2];
    }
  }
}

/* ----------------------------------------------------------------------
   compute sigma tensor
   needed whenever p_target or h0_inv changes
-----------------------------------------------------------------------*/

void FixNHMassive::compute_sigma()
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

double FixNHMassive::compute_strain_energy()
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

void FixNHMassive::compute_deviatoric()
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

void FixNHMassive::compute_temp_target()
{
  double delta = update->ntimestep - update->beginstep;
  if (delta != 0.0) delta /= update->endstep - update->beginstep;

  t_target = t_start + delta * (t_stop-t_start);
}

/* ----------------------------------------------------------------------
   compute hydrostatic target pressure
-----------------------------------------------------------------------*/

void FixNHMassive::compute_press_target()
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
   update omega_dot
-----------------------------------------------------------------------*/

void FixNHMassive::nh_omega_dot(double dt)
{
  double f_omega,volume;

  if (dimension == 3) volume = domain->xprd*domain->yprd*domain->zprd;
  else volume = domain->xprd*domain->yprd;

  if (deviatoric_flag) compute_deviatoric();

  mtk_term1 = 0.0;
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

  for (int i = 0; i < 3; i++)
    if (p_flag[i]) {
      f_omega = (p_current[i]-p_hydro)*volume /
        (omega_mass[i] * nktv2p) + mtk_term1 / omega_mass[i];
      if (deviatoric_flag) f_omega -= fdev[i]/(omega_mass[i] * nktv2p);
      omega_dot[i] += f_omega*dt;
    }

  mtk_term2 = 0.0;
  for (int i = 0; i < 3; i++)
    if (p_flag[i])
      mtk_term2 += omega_dot[i];
  if (pdim > 0) mtk_term2 /= pdim * atom->natoms;

  if (pstyle == TRICLINIC) {
    for (int i = 3; i < 6; i++) {
      if (p_flag[i]) {
        f_omega = p_current[i]*volume/(omega_mass[i] * nktv2p);
        if (deviatoric_flag)
          f_omega -= fdev[i]/(omega_mass[i] * nktv2p);
        omega_dot[i] += f_omega*dt;
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

void FixNHMassive::pre_exchange()
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

double FixNHMassive::memory_usage()
{
  double bytes = 0.0;
  if (irregular) bytes += irregular->memory_usage();
  bytes += atom->nmax*2*sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
   allocate atom-based arrays
------------------------------------------------------------------------- */

void FixNHMassive::grow_arrays(int nmax)
{
  memory->grow(eta_dot,nmax,3,"fix_nh:eta_dot");
}

/* ----------------------------------------------------------------------
   copy values within local atom-based array
------------------------------------------------------------------------- */

void FixNHMassive::copy_arrays(int i, int j, int /*delflag*/)
{
  eta_dot[j][0] = eta_dot[i][0];
  eta_dot[j][1] = eta_dot[i][1];
  eta_dot[j][2] = eta_dot[i][2];
}

/* ----------------------------------------------------------------------
   pack values in local atom-based array for exchange with another proc
------------------------------------------------------------------------- */

int FixNHMassive::pack_exchange(int i, double *buf)
{
  int n = 0;
  buf[n++] = eta_dot[i][0];
  buf[n++] = eta_dot[i][1];
  buf[n++] = eta_dot[i][2];
  return n;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based array from exchange with another proc
------------------------------------------------------------------------- */

int FixNHMassive::unpack_exchange(int nlocal, double *buf)
{
  int n = 0;
  eta_dot[nlocal][0] = buf[n++];
  eta_dot[nlocal][1] = buf[n++];
  eta_dot[nlocal][2] = buf[n++];
  return n;
}
