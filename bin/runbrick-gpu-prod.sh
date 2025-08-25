#! /bin/bash

# Script for running the legacypipe code within a Shifter container at NERSC

export COSMO=/dvs_ro/cfs/cdirs/cosmo

## FIXME
#export LEGACY_SURVEY_DIR=$COSMO/work/legacysurvey/dr11
export LEGACY_SURVEY_DIR=$SCRATCH/dr11-sky

outdir=$SCRATCH/dr11-gpu

export GAIA_CAT_DIR=$COSMO/data/gaia/dr3/healpix
export GAIA_CAT_PREFIX=healpix
export GAIA_CAT_SCHEME=nested
export GAIA_CAT_VER=3

export DUST_DIR=$COSMO/data/dust/v0_1
export UNWISE_COADDS_DIR=$COSMO/data/unwise/neo7/unwise-coadds/fulldepth:$COSMO/data/unwise/allwise/unwise-coadds/fulldepth
export UNWISE_COADDS_TIMERESOLVED_DIR=$COSMO/work/wise/outputs/merge/neo7
export UNWISE_MODEL_SKY_DIR=$COSMO/data/unwise/neo7/unwise-catalog/mod

export TYCHO2_KD_DIR=$COSMO/staging/tycho2
export LARGEGALAXIES_CAT=$COSMO/staging/largegalaxies/v3.0/SGA-ellipse-v3.0.kd.fits
export SKY_TEMPLATE_DIR=$COSMO/work/legacysurvey/dr10/calib/sky_pattern

unset BLOB_MASK_DIR
unset PS1CAT_DIR
unset GALEX_DIR

# Don't add ~/.local/ to Python's sys.path
export PYTHONNOUSERSITE=1
# Force MKL single-threaded
# https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
# To avoid problems with MPI and Python multiprocessing
export MPICH_GNI_FORK_MODE=FULLCOPY
export KMP_AFFINITY=disabled

brick="$1"
bri=${brick:0:3}

mkdir -p "$outdir/logs/$bri"
mkdir -p "$outdir/metrics/$bri"
mkdir -p "$outdir/pickles/$bri"
log="$outdir/logs/$bri/$brick.log"
echo Logging to: "$log"

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

echo -e "\n\n\n" >> "$log"
echo "-----------------------------------------------------------------------------------------" >> "$log"
echo -e "\nStarting on $(hostname)\n" >> "$log"
echo "-----------------------------------------------------------------------------------------" >> "$log"

#export LEGACYPIPE_DIR=/src/legacypipe/py
#export PYTHONPATH=/src/unwise_psf/py:/src/legacypipe/py:/usr/local/lib/python

cd $LEGACYPIPE_DIR

echo "LEGACYPIPE_DIR: $LEGACYPIPE_DIR" >> "$log"
echo "PYTHONPATH: $PYTHONPATH" >> "$log"
python -c "import tractor; print(tractor.__file__)" >> "$log"
python -c "import legacypipe; print(legacypipe.__file__)" >> "$log"
python -c "import sys; print('\n'.join(sys.path))" >> "$log"

#       --skip-calibs \

# https://github.com/dmargala/desiscaleflow/blob/main/bin/desi_avoid_home
# https://developer.nvidia.com/blog/cuda-pro-tip-understand-fat-binaries-jit-caching/
# https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars
#export CUDA_CACHE_PATH=$SCRATCH/.nv/ComputeCache
export CUDA_CACHE_PATH=$TMPCACHE/.nv/ComputeCache
mkdir -p $CUDA_CACHE_PATH
#cp -r $HOME/.config/matplotlib $MPLCONFIGDIR

# # undocumented, see https://github.com/cupy/cupy/issues/3887
# export CUPY_CUDA_LIB_PATH=$SCRATCH/cupy/cuda_lib
# https://docs.cupy.dev/en/stable/reference/environment.html#envvar-CUPY_CACHE_DIR
#export CUPY_CACHE_DIR=/tmp/cupy/kernel_cache
export CUPY_CACHE_DIR=$TMPCACHE/.cupy/kernel_cache
mkdir -p $CUPY_CACHE_DIR

python -O $LEGACYPIPE_DIR/legacypipe/runbrick.py \
       --run south \
       --brick "$brick" \
       --use-gpu \
       --sub-blobs \
       --ngpu 2 \
       --threads 64 \
       --gpumode 2 \
       --threads-per-gpu 4 \
       --bands g,r,z \
       --rgb-stretch 1.5 \
       --nsatur 2 \
       --survey-dir "$LEGACY_SURVEY_DIR" \
       --outdir "$outdir" \
       --pickle "${outdir}/pickles/${bri}/runbrick-%(brick)s-%%(stage)s.pickle" \
       --checkpoint "${outdir}/checkpoints/${bri}/checkpoint-${brick}.pickle" \
       --checkpoint-period 120 \
       --write-stage srcs \
       --release 10099 \
       --no-wise \
       >> "$log" 2>&1

# Save the return value from the python command -- otherwise we
# exit 0 because the rm succeeds!
status=$?

# /Config directory nonsense
rm -R $TMPCACHE

exit $status
