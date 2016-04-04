import os
import sys
import argparse
import glob
from astropy.io import fits
import numpy as np


def bash(cmd):
    ret= os.system('%s' % cmd)
    if ret:
        print 'command failed: %s' % cmd
        sys.exit() 


def_fn='BB_template.sh'
fn='submit.sh'
if os.path.exists(fn): bash('rm %s' % fn)
bash('cp %s %s' % (def_fn,fn))

def echo(cmd):
    ret= os.system('echo %s >> %s' % (cmd,fn))
    if ret:
        print 'command failed: %s' % cmd
        sys.exit() 


parser = argparse.ArgumentParser(description="test")
parser.add_argument("-cores",type=int,action="store",help='# of images to copy from /project to /scratch')
parser.add_argument("-nptf",type=int,choices=[100,200],action="store",help='which band')
args = parser.parse_args()

out='BB_tractor_800wh_nptf%d_c%d' % (args.nptf,args.cores)
scr_decals='/global/cscratch1/sd/kaylanb/desi/imaging/3523p002_mosaicTest_decamOverlap/ptf_gr_mosaic_z/decals-dir'
scr_legacypipe='/global/cscratch1/sd/kaylanb/desi/legacypipe/py/legacypipe'
scr_mosaic='/global/cscratch1/sd/kaylanb/desi/images/brick_3523p002/mosaicz_Test'
scr_ptf='/global/cscratch1/sd/kaylanb/desi/images/brick_3523p002/ptf'
scr_tractor='/global/cscratch1/sd/kaylanb/desi/imaging/3523p002_mosaicTest_decamOverlap/ptf_gr_mosaic_z/%s' % out

echo('#get images')
echo('#DW stage_in source=%s destination=$DW_JOB_STRIPED/mosaic type=directory' % (scr_mosaic))
echo('#DW stage_in source=%s destination=$DW_JOB_STRIPED/ptf type=directory' % (scr_ptf))
echo('#get decals-dir')
echo('#DW stage_in source=%s destination=$DW_JOB_STRIPED/decals-dir type=directory' % scr_decals)
echo('#soft link ccds.fits')
echo('cd $DW_JOB_STRIPED/decals-dir')
echo('ln -s ptf_gr-mosaic_z-ccds_SUBSET_%d.fits decals-ccds.fits' % args.nptf)
echo('#soft link images')
echo('cd images')
echo('ln -s $DW_JOB_STRIPED/mosaic mosaic')
echo('ln -s $DW_JOB_STRIPED/ptf ptf')
echo('#get tractor codebase')
echo('#DW stage_in source=scr_legacypipe destination=$DW_JOB_STRIPED/legacypipe type=directory')
echo('#RUN')
echo('cd $SLURM_SUBMIT_DIR')
echo('#set up file transfer from output on bb to scratch')
echo('#DW stage_out source=$DW_JOB_STRIPED/tractor destination=%s/%s type=directory' % (scr_tractor,out))
echo('echo START TIME:')
echo('date')
echo('export OMP_NUM_THREADS=%d' % (args.cores))
echo ('srun -n 1 -c %d python legacypipe/runbrick.py --brick 3523p002 --no-wise --force-all --no-sdss --width 800 --height 800 --decals-dir $DW_JOB_STRIPED/decals-dir --outdir $DW_JOB_STRIPED/tractor --pixpsf --splinesky --threads %d' % (args.cores,args.cores))
echo('echo END TIME:')
echo('date')
echo('echo DONE')

