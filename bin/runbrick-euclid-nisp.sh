#! /bin/bash

export LEGACY_SURVEY_DIR=/global/homes/d/dstn/legacypipe/py/euclid-dr1-dir

outdir=$SCRATCH/euclid-dr1-nisp

export COSMO=/dvs_ro/cfs/cdirs/cosmo

export DUST_DIR=$COSMO/data/dust/v0_1

export GAIA_CAT_DIR=$COSMO/data/gaia/dr3/healpix
export GAIA_CAT_PREFIX=healpix
export GAIA_CAT_SCHEME=nested
export GAIA_CAT_VER=3

export TYCHO2_KD_DIR=$COSMO/staging/tycho2
export PS1CAT_DIR=$COSMO/work/ps1/cats/chunks-qz-star-v3
unset LARGEGALAXIES_CAT
unset SKY_TEMPLATE_DIR
unset BLOB_MASK_DIR

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
mkdir "$TMPCACHE/cache"
mkdir "$TMPCACHE/config"
# astropy
export XDG_CACHE_HOME=$TMPCACHE/cache
export XDG_CONFIG_HOME=$TMPCACHE/config
mkdir "$XDG_CACHE_HOME/astropy"
cp -r "$HOME/.astropy/cache" "$XDG_CACHE_HOME/astropy"
mkdir "$XDG_CONFIG_HOME/astropy"
#cp -r "$HOME/.astropy/config" "$XDG_CONFIG_HOME/astropy"
# matplotlib
export MPLCONFIGDIR=$TMPCACHE/matplotlib
mkdir "$MPLCONFIGDIR"
cp -r "$HOME/.config/matplotlib" "$MPLCONFIGDIR"

echo -e "\n\n\n" >> "$log"
echo "-----------------------------------------------------------------------------------------" >> "$log"
echo -e "\nStarting on $(hostname)\n" >> "$log"
echo "-----------------------------------------------------------------------------------------" >> "$log"

# Local
export LEGACYPIPE_DIR=/global/homes/d/dstn/legacypipe/py
export PYTHONPATH=$LEGACYPIPE_DIR:${PYTHONPATH}

python -u "$LEGACYPIPE_DIR/legacypipe/runbrick.py"  \
       --brick ${brick} \
       --skip-calibs \
       --pixscale 0.3 \
       --width 3144 --height 3144 \
       --bands Y,J,H \
       --rgb-stretch 2 \
       --nsatur 1 \
       --survey-dir "$LEGACY_SURVEY_DIR" \
       --outdir "$outdir" \
       --checkpoint "${outdir}/checkpoints/${bri}/checkpoint-${brick}.pickle" \
       --checkpoint-period 600 \
       --pickle "${outdir}/pickles/${bri}/runbrick-%(brick)s-%%(stage)s.pickle" \
       --no-wise \
       --threads 64 \
       >> "$log" 2>&1

# Save the return value from the python command -- otherwise we
# exit 0 because the rm succeeds!
status=$?

# /Config directory nonsense
rm -R "$TMPCACHE"

exit $status
