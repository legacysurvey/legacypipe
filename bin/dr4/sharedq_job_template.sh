#!/bin/bash -l

#SBATCH -p shared
#SBATCH -n 2
#SBATCH -t 00:10:00
#SBATCH --account=desi
#SBATCH -J repack
#SBATCH -L SCRATCH,project,projecta
#SBATCH -C haswell
#--qos=scavenger
#--mail-user=kburleigh@lbl.gov
#--mail-type=END,FAIL

