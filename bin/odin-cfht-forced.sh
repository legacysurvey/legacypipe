#! /bin/bash

brick=$1

#ODIN=/global/cfs/cdirs/cosmo/work/users/dstn/ODIN

export COSMO=/dvs_ro/cfs/cdirs/cosmo
export ODIN=/dvs_ro/cfs/cdirs/cosmo/work/users/dstn/ODIN
export CFIS=/dvs_ro/cfs/cdirs/desi/users/dstn/cfis-dir

# export survey_dir=$SCRATCH/cfht-cosmos-u
# export out_dir=$SCRATCH/odin-bricks-cosmos/forced-cfht-u
# export cat_dir=$SCRATCH/odin-bricks-cosmos

# export survey_dir=$SCRATCH/cfht-xmm-u
# export out_dir=$SCRATCH/odin-bricks-xmm/forced-cfht-u
# export cat_dir=$SCRATCH/odin-bricks-xmm

export survey_dir=$CFIS/
export out_dir=$SCRATCH/odin-bricks-cosmos/forced-cfht-u-v2
export cat_dir=$ODIN/2024-a/odin-bricks-cosmos

export LARGEGALAXIES_CAT=$COSMO/staging/largegalaxies/v3.0/SGA-ellipse-v3.0.kd.fits

mkdir -p ${out_dir}/logs-forced
echo Logging to ${out_dir}/logs-forced/${brick}.log

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

python -O legacypipe/forced_photom_brickwise.py \
       --brick $brick \
       --survey-dir $survey_dir \
       --catalog-dir $cat_dir \
       --outdir $out_dir \
       --bands u \
       --threads 32 \
       >> ${out_dir}/logs-forced/${brick}.log 2>&1

# Save the return value from the python command -- otherwise we
# exit 0 because the rm succeeds!
status=$?

# /Config directory nonsense
rm -R $TMPCACHE

exit $status
