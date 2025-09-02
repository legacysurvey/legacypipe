#! /bin/bash

brick=$1

#ODIN=/global/cfs/cdirs/cosmo/work/users/dstn/ODIN

export COSMO=/dvs_ro/cfs/cdirs/cosmo
export ODIN=/dvs_ro/cfs/cdirs/cosmo/work/users/dstn/ODIN

#export survey_dir=$COSMO/work/users/dstn/ODIN/cosmos-hsc-coadd-2
#export survey_dir=$SCRATCH/hsc-co-4
export survey_dir=$ODIN/2024-a/hsc-co-deep

#outdir=$CSCRATCH/odin-hsc-brickwise-v2
#outdir=$PSCRATCH/odin-hsc-xmm
#outdir=$CSCRATCH/odin-xmm
#outdir=$CSCRATCH/odin-hsc-deep23-v2
#outdir=$CSCRATCH/odin-xmm-v2
#outdir=$ODIN/cosmos
#coadd_dir=$ODIN/hsc-single-band-coadds-v2
#outdir=$SCRATCH/odin-cosmos-3
#outdir=$SCRATCH/odin-bricks-cosmos/forced-hsc-wide
#outdir=$SCRATCH/odin-bricks-xmm/forced-hsc-wide
outdir=$SCRATCH/odin-bricks-cosmos/forced-hsc-deep

#catdir=$SCRATCH/odin-bricks-cosmos
#catdir=$SCRATCH/odin-bricks-xmm
catdir=$ODIN/2024-a/odin-bricks-cosmos

export LARGEGALAXIES_CAT=$COSMO/staging/largegalaxies/v3.0/SGA-ellipse-v3.0.kd.fits

# Something weird going on with find_missing_sga
#export LARGEGALAXIES_CAT=$COSMO/staging/largegalaxies/v3.0/SGA-ellipse-v3.0.kd.fits

mkdir -p ${outdir}/logs-forced
echo Logging to ${outdir}/logs-forced/${brick}.log

# Don't add ~/.local/ to Python's sys.path
export PYTHONNOUSERSITE=1
# Force MKL single-threaded
# https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
# To avoid problems with MPI and Python multiprocessing
export MPICH_GNI_FORK_MODE=FULLCOPY
export KMP_AFFINITY=disabled

# # Config directory nonsense
export TMPCACHE=$(mktemp -d)
mkdir $TMPCACHE/cache
mkdir $TMPCACHE/config
# astropy
export XDG_CACHE_HOME=$TMPCACHE/cache
export XDG_CONFIG_HOME=$TMPCACHE/config
mkdir $XDG_CACHE_HOME/astropy
cp -r $HOME/.astropy/cache $XDG_CACHE_HOME/astropy
mkdir $XDG_CONFIG_HOME/astropy
cp -r $HOME/.astropy/config $XDG_CONFIG_HOME/astropy
# matplotlib
export MPLCONFIGDIR=$TMPCACHE/matplotlib
mkdir $MPLCONFIGDIR
cp -r $HOME/.config/matplotlib $MPLCONFIGDIR

#       --catalog-dir $ODIN/xmm-N673 \
       #--catalog-dir $ODIN/deep23-N419 \
#       --catalog-dir $ODIN/deep23-N419-2022-10 \
#       --catalog-dir $ODIN/cosmos \

# XMM    --bands g,r,i,i2,z,y \

# COSMOS --bands g,r2,i2,z,y \

# COSMOS D/UD --bands g,r,r2,i,i2,z,y \

python -O legacypipe/forced_photom_brickwise.py \
       --brick ${brick} \
       --bands g,r,r2,i,i2,z,y \
       --survey-dir ${survey_dir} \
       --catalog-dir ${catdir} \
       --outdir ${outdir} \
       --threads 32 \
       >> ${outdir}/logs-forced/${brick}.log 2>&1

#       >> ${outdir}/${band}/logs/forced-${brick}-${band}.log 2>&1
#       --bands ${band} \
#       --survey-dir $coadd_dir/odin-hsc-${band} \
#       --catalog-dir $ODIN/2band \

# Save the return value from the python command -- otherwise we
# exit 0 because the rm succeeds!
status=$?

# /Config directory nonsense
rm -R $TMPCACHE

exit $status
