#! /bin/bash

brick=$1

ODIN=/dvs_ro/cfs/cdirs/cosmo/work/users/dstn/ODIN

export COSMO=/dvs_ro/cfs/cdirs/cosmo

#export survey_dir=$SCRATCH/hsc-co-4
#export survey_dir=$ODIN/2024-a/hsc-co-4
export survey_dir=$ODIN/2024-a/hsc-co-deep

#outdir=$SCRATCH/ibis4/forced-hsc-wide
outdir=$SCRATCH/ibis4/forced-hsc-deep
catdir=$SCRATCH/ibis4

#outdir=$SCRATCH/ibis3-bail/forced-hsc-wide-2
#catdir=$SCRATCH/ibis3-bail

#export LARGEGALAXIES_CAT=$COSMO/staging/largegalaxies/v3.0/SGA-ellipse-v3.0.kd.fits
unset LARGEGALAXIES_CAT

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


export PYTHONPATH=.:${PYTHONPATH}
#       --bands g,r,i,z,y \
    # -u
# COSMOS WIDE
#       --bands g,r2,i2,z,y \
    # COSMOS DEEP
#       --bands g,r,r2,i,i2,z,y \

python -u -O legacypipe/forced_photom_brickwise.py \
       --brick $brick \
       --bands g,r,r2,i,i2,z,y \
       --survey-dir $survey_dir \
       --catalog-dir ${catdir} \
       --outdir ${outdir} \
       >> ${outdir}/logs-forced/${brick}.log 2>&1
#       --threads 32 \

# Save the return value from the python command -- otherwise we
# exit 0 because the rm succeeds!
status=$?

# /Config directory nonsense
rm -R $TMPCACHE

exit $status
