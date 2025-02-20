#! /bin/bash

brick="$1"

outdir=$SCRATCH/ibis3-n395

unset BLOB_MASK_DIR
export COSMO=/dvs_ro/cfs/cdirs/cosmo

export LEGACY_SURVEY_DIR=$COSMO/work/legacysurvey/ibis

export LARGEGALAXIES_CAT=$COSMO/staging/largegalaxies/v3.0/SGA-ellipse-v3.0.kd.fits
export DUST_DIR=$COSMO/data/dust/v0_1

export GAIA_CAT_DIR=$COSMO/data/gaia/dr3/healpix
export GAIA_CAT_VER=3
export GAIA_CAT_PREFIX=healpix
export GAIA_CAT_SCHEME=nested
export TYCHO2_KD_DIR=$COSMO/staging/tycho2
export PS1CAT_DIR=$COSMO/work/ps1/cats/chunks-qz-star-v3

unset SKY_TEMPLATE_DIR

# PYTHONPATH is set in the container.
#export PYTHONPATH=/usr/local/lib/python:/usr/local/lib/python3.6/dist-packages:/src/unwise_psf/py:.

# Don't add ~/.local/ to Python's sys.path
export PYTHONNOUSERSITE=1

# Force MKL single-threaded
# https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

# To avoid problems with MPI and Python multiprocessing
export MPICH_GNI_FORK_MODE=FULLCOPY
export KMP_AFFINITY=disabled

bri=$(echo $brick | head -c 3)
mkdir -p $outdir/logs/$bri
log="$outdir/logs/$bri/$brick.log"

mkdir -p $outdir/metrics/$bri

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

echo Logging to: $log
echo Running on $(hostname)

echo -e "\n\n\n" >> $log
echo "-----------------------------------------------------------------------------------------" >> $log
#echo "PWD: $(pwd)" >> $log
#echo >> $log

echo -e "\nStarting on $(hostname)\n" >> $log
echo "-----------------------------------------------------------------------------------------" >> $log

export LEGACYPIPE_DIR=.
export PYTHONPATH=.:${PYTHONPATH}
#cd $LEGACYPIPE_DIR
#export LEGACYPIPE_DIR=/src/legacypipe/py
#       --bands M464 \
#       --bands $band \
#       --coadd-bw \
#       --skip-calibs \
#       --bands M411,M464 \

# Deep fields:
#       --blob-dilate 4 \

#       --plots \
# -O
#       --bands M411,M438,M464,M490,M517 \
#      --run ibis-special \
#       --skip \

python -u -O $LEGACYPIPE_DIR/legacypipe/runbrick.py \
       --blob-dilate 4 \
       --coadd-bw \
       --bands N395 \
       --nsatur 2 \
       --brick $brick \
       --rgb-stretch 1.5 \
       --sub-blobs \
       --no-wise \
       --skip-calibs \
       --checkpoint ${outdir}/checkpoints/${bri}/checkpoint-${brick}.pickle \
       --checkpoint-period 300 \
       --pickle "${outdir}/pickles/${bri}/runbrick-%(brick)s-%%(stage)s.pickle" \
       --outdir $outdir \
       --write-stage srcs \
       --threads 32 \
       >> $log 2>&1

#        --skip-coadd \
#        --stage image_coadds \
#        --blob-image \

# Save the return value from the python command -- otherwise we
# exit 0 because the rm succeeds!
status=$?
# /Config directory nonsense
rm -R $TMPCACHE
exit $status
