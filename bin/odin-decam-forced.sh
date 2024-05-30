#! /bin/bash

brick=$1

export COSMO=/dvs_ro/cfs/cdirs/cosmo

ODIN=$COSMO/work/users/dstn/ODIN

#out_dir=$ODIN/cosmos/forced-decam-deep
#cat_dir=$ODIN/cosmos/
#survey_dir=$COSMO/work/legacysurvey/dr10-deep/cosmos

cat_dir=$SCRATCH/odin-bricks-cosmos
out_dir=$SCRATCH/odin-bricks-cosmos/forced-decam-u
survey_dir=$ODIN

#unset LARGEGALAXIES_CAT
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

#--bands g,r,i,z \

python -u -O legacypipe/forced_photom_brickwise.py \
       --brick $brick \
       --bands u \
       --survey-dir ${survey_dir} \
       --catalog-dir ${cat_dir} \
       --outdir ${out_dir} \
       --threads 32 \
       >> ${out_dir}/logs-forced/${brick}.log 2>&1

# Save the return value from the python command -- otherwise we
# exit 0 because the rm succeeds!
status=$?

# /Config directory nonsense
rm -R $TMPCACHE

exit $status
