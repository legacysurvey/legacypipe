import os
import sys
import argparse

def bash(cmd):
    ret= os.system('%s' % cmd)
    if ret:
        print 'command failed: %s' % cmd
        sys.exit() 


def_fn='BB_template.sh'
fn='submit.sh'
if os.path.exists(fn): bash('rm %s' % fn)
bash('cp %s %s' % (def_fn,fn))
fin=open(fn,'a')

def ap(cmd):
    fin.write('%s\n' % cmd)


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

ap('#get images')
ap('#DW stage_in source=%s destination=$DW_JOB_STRIPED/mosaic type=directory' % (scr_mosaic))
ap('#DW stage_in source=%s destination=$DW_JOB_STRIPED/ptf type=directory' % (scr_ptf))
ap('#get decals-dir')
ap('#DW stage_in source=%s destination=$DW_JOB_STRIPED/decals-dir type=directory' % scr_decals)
ap('#soft link ccds.fits')
ap('cd $DW_JOB_STRIPED/decals-dir')
ap('ln -s ptf_gr-mosaic_z-ccds_SUBSET_%d.fits decals-ccds.fits' % args.nptf)
ap('#soft link images')
ap('cd images')
ap('ln -s $DW_JOB_STRIPED/mosaic mosaic')
ap('ln -s $DW_JOB_STRIPED/ptf ptf')
ap('#get tractor codebase')
ap('#DW stage_in source=scr_legacypipe destination=$DW_JOB_STRIPED/legacypipe type=directory')
ap('#RUN')
ap('cd $SLURM_SUBMIT_DIR')
ap('#set up file transfer from output on bb to scratch')
ap('#DW stage_out source=$DW_JOB_STRIPED/tractor destination=%s/%s type=directory' % (scr_tractor,out))
ap('echo START TIME:')
ap('date')
ap('export OMP_NUM_THREADS=%d' % (args.cores))
ap('srun -n 1 -c %d python legacypipe/runbrick.py --brick 3523p002 --no-wise --force-all --no-sdss --width 800 --height 800 --decals-dir $DW_JOB_STRIPED/decals-dir --outdir $DW_JOB_STRIPED/tractor --pixpsf --splinesky --threads %d' % (args.cores,args.cores))
ap('echo END TIME:')
ap('date')
ap('echo DONE')

