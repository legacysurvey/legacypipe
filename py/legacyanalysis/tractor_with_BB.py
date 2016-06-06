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
parser.add_argument("-brick",action="store",help='brick to process',required=True)
parser.add_argument("-cores",type=int,action="store",help='# of images to copy from /project to /scratch',required=True)
parser.add_argument("-hours",type=int,choices=[1,2,3,4,5,6],action="store",help='how many hours run for',required=True)
parser.add_argument("-legacypipe_dir",action="store",help='absolute path, MUST be on scratch',required=True)
parser.add_argument("-decals_dir",action="store",help='absolute path, MUST be on scratch',required=True)
parser.add_argument("-scratch_outdir",action="store",help='where to write output when stage out from BB',required=True)
parser.add_argument("-images_dir",action="store",help='absolute path to images that Tractor will read, MUST be on scratch',required=True)
args = parser.parse_args()

#bash script
fn='%s_BB.sh' % args.brick
if os.path.exists(fn): bash('rm %s' % fn)
fin=open(fn,'w')
#header
ap('#!/bin/bash -l\n')
ap('#SBATCH -p regular') 
ap('#SBATCH -N 1')  
ap('#SBATCH -t 0%d:00:00' % args.hours)
ap('#SBATCH -J %s' % args.brick)      
ap('#SBATCH -o %s' % args.brick)
#use BB
ap('#DW jobdw capacity=50GB access_mode=striped type=scratch\n')
#stage in codebase
ap('#DW stage_in source=%s destination=$DW_JOB_STRIPED/legacypipe type=directory' % args.legacypipe_dir)
#stage in images
ap('#DW stage_in source=%s destination=$DW_JOB_STRIPED/images type=directory' % (args.images_dir))
#stage in decals_dir
ap('#DW stage_in source=%s destination=$DW_JOB_STRIPED/decals-dir type=directory' % args.decals_dir)
#finally, tell BB where to stage_out on SCRATCH
ap('#DW stage_out source=$DW_JOB_STRIPED/tractor destination=%s type=directory' % args.scratch_outdir)
#soft links not preserved, go into decals_dir on BB and recreate them
ap('cd $DW_JOB_STRIPED/decals-dir')
ap('ln -s $DW_JOB_STRIPED/images images')
#run tractor
ap('cd $SLURM_SUBMIT_DIR')
#sanity check, ls the BB directories
ap('echo DIRECTORIES ON BB ARE AS FOLLOWS')
ap('echo legacypipe:; ls $DW_JOB_STRIPED/legacypipe')
ap('echo images:; ls $DW_JOB_STRIPED/images')
ap('echo decals-dir:; ls $DW_JOB_STRIPED/decals-dir')
#phew, ready for tractor!
ap('export OMP_NUM_THREADS=%d' % (args.cores))
cmd= 'srun -n 1 -c %d python legacypipe/runbrick.py --brick %s --survey-dir $DW_JOB_STRIPED/decals-dir --outdir $DW_JOB_STRIPED/tractor --threads %d --force-all --no-write' % (args.cores,args.brick,args.cores)
ap(cmd)
print "wrote %s" % fn

