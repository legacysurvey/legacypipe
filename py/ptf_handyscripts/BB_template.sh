#!/bin/bash -l

#SBATCH -p debug  
#SBATCH -N 1  
#SBATCH -t 00:20:00    
#SBATCH -J ptf_timing_BB.o%j      
#SBATCH -o ptf_timing_BB.o%j
#SBATCH --mail-user=kburleigh@lbl.gov
#SBATCH --mail-type=END,FAIL
#DW jobdw capacity=50GB access_mode=striped type=scratch


