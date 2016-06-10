'''
this script makes the bash script for submitting Tractor jobs to Cori's BB
BB scripts currenlty cannot define variables that contain the BB stage in/out dirs, so the script is ~ necessary 
'''
import os
import sys
import argparse

def bash(cmd):
    ret= os.system('%s' % cmd)
    if ret:
        print 'command failed: %s' % cmd
        sys.exit() 

def ap(cmd):
    fin.write('%s\n' % cmd)

parser = argparse.ArgumentParser(description="test")
parser.add_argument("-machine",choices=['cori','edison'],action="store",help='brick to process',required=True)
args = parser.parse_args()

bricks=['2523p355','2616p277','1306p247','1931p197','2508p167','2414p142','2493p097','2430p047','3293m002','0364m052']
for brick in bricks: 
    #bash script
    fn='submit_%s.sh' % brick
    if os.path.exists(fn): bash('rm %s' % fn)
    fin=open(fn,'w')

    #header
    ap('#!/bin/bash -l\n')
    ap('#SBATCH -p debug') 
    ap('#SBATCH -N 1')  
    ap('#SBATCH -t 00:30:00')
    ap('#SBATCH -J %s' % brick) 
    pref='#SBATCH -o %s' % brick    
    ap(pref+'.o%j')
    ap('#SBATCH --mail-user=kburleigh@lbl.gov')
    ap('#SBATCH --mail-type=END,FAIL')
    if args.machine == 'edison': ap('#SBATCH -L SCRATCH,project')
    #code
    if args.machine == 'cori': ap('export OMP_NUM_THREADS=32') 
    else: ap('export OMP_NUM_THREADS=24') 
    ap('cd $SLURM_SUBMIT_DIR')
    ap('export DECALS_SIM_DIR=${SLURM_SUBMIT_DIR}/ten_bricks')
    ap('srun -n 1 -c ${OMP_NUM_THREADS} python legacyanalysis/decals_sim.py --brick %s -n 500 -o STAR --threads ${OMP_NUM_THREADS}' % brick)
    ap('echo DONE')
    fin.close()

