#!/bin/bash -l

#SBATCH -p debug  
#SBATCH -N 1  
#SBATCH -t 00:20:00    
#SBATCH -J ptf_timing_BB.o%j      
#SBATCH -o ptf_timing_BB.o%j
#SBATCH --mail-user=kburleigh@lbl.gov
#SBATCH --mail-type=END,FAIL
#DW jobdw capacity=50GB access_mode=striped type=scratch

export mycores=8
export NPTF=100
export out=BB_tractor_800wh_nptf${NPTF}_c${mycores}
export OMP_NUM_THREADS=${mycores}

export scr_decals=/global/cscratch1/sd/kaylanb/desi/imaging/3523p002_mosaicTest_decamOverlap/ptf_gr_mosaic_z/decals-dir
export scr_legacypipe=/global/cscratch1/sd/kaylanb/desi/legacypipe/py/legacypipe
export scr_mosaic=/global/cscratch1/sd/kaylanb/desi/images/brick_3523p002/mosaicz_Test
export scr_ptf=/global/cscratch1/sd/kaylanb/desi/images/brick_3523p002/ptf
export scr_tractor=/global/cscratch1/sd/kaylanb/desi/imaging/3523p002_mosaicTest_decamOverlap/ptf_gr_mosaic_z/${out}

#get images
#DW stage_in source=/global/cscratch1/sd/kaylanb/desi/images/brick_3523p002/mosaicz_Test destination=$DW_JOB_STRIPED/mosaic type=directory
#DW stage_in source=/global/cscratch1/sd/kaylanb/desi/images/brick_3523p002/ptf destination=$DW_JOB_STRIPED/ptf type=directory
#get decals-dir
#DW stage_in source=/global/cscratch1/sd/kaylanb/desi/imaging/3523p002_mosaicTest_decamOverlap/ptf_gr_mosaic_z/decals-dir destination=$DW_JOB_STRIPED/decals-dir type=directory
#soft link ccds.fits
cd $DW_JOB_STRIPED/decals-dir
ln -s ptf_gr-mosaic_z-ccds_SUBSET_${NPTF}.fits decals-ccds.fits
#soft link images
cd images
ln -s $DW_JOB_STRIPED/mosaic mosaic
ln -s $DW_JOB_STRIPED/ptf ptf
#get tractor codebase
#DW stage_in source=/global/cscratch1/sd/kaylanb/desi/legacypipe/py/legacypipe destination=$DW_JOB_STRIPED/legacypipe type=directory
#RUN
cd $SLURM_SUBMIT_DIR
#set up file transfer from output on bb to scratch
#DW stage_out source=$DW_JOB_STRIPED/tractor destination=/global/cscratch1/sd/kaylanb/desi/imaging/3523p002_mosaicTest_decamOverlap/ptf_gr_mosaic_z/BB_tractor_800wh_nptf100_c8 type=directory
echo START TIME:
date
srun -n 1 -c ${mycores} python legacypipe/runbrick.py --brick 3523p002 --no-wise --force-all --no-sdss --width 800 --height 800 --decals-dir $DW_JOB_STRIPED/decals-dir --outdir $DW_JOB_STRIPED/tractor --pixpsf --splinesky --threads ${mycores}
echo END TIME:
date
echo DONE

#how does dir look?
#echo PWD IS:
#pwd
#echo LS IS:
#ls
#echo HERE IS: ls DW_JOB_STRIPED
#ls $DW_JOB_STRIPED
#echo AND: ls DW_JOB_STRIPED/decals-dir_NPTF
#ls $DW_JOB_STRIPED/decals-dir_${NPTF}

