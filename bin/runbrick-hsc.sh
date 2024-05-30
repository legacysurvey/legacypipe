#! /bin/bash

# Script for running the legacypipe code within a Shifter container at NERSC

# Using HSC g-band data on COSMOS
#outdir=$SCRATCH/hsc-g-cosmos

# HSC/WIDE COSMOS
outdir=$SCRATCH/hsc-co-4

export COSMO=/dvs_ro/cfs/cdirs/cosmo

#export LEGACY_SURVEY_DIR=$COSMO/work/users/dstn/ODIN/
export LEGACY_SURVEY_DIR=$outdir

#export LARGEGALAXIES_CAT=$COSMO/staging/largegalaxies/v3.0/SGA-ellipse-v3.0.kd.fits
unset LARGEGALAXIES_CAT
unset BLOB_MASK_DIR
unset SKY_TEMPLATE_DIR

export DUST_DIR=$COSMO/data/dust/v0_1

export GAIA_CAT_DIR=$COSMO/data/gaia/edr3/healpix
export GAIA_CAT_PREFIX=healpix
export GAIA_CAT_SCHEME=nested
export GAIA_CAT_VER=E
export TYCHO2_KD_DIR=$COSMO/staging/tycho2
export PS1CAT_DIR=$COSMO/work/ps1/cats/chunks-qz-star-v3
#export SKY_TEMPLATE_DIR=/global/cfs/cdirs/cosmo/work/legacysurvey/sky-templates

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

brick="$1"

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

#export LEGACYPIPE_DIR=/src/legacypipe/py

#export PYTHONPATH=.:/global/homes/d/dstn/tractor:${PYTHONPATH}
#export LEGACYPIPE_DIR=$(pwd)

# hsc-g-cosmos:
#       --bands g \
#       --coadd-bw \

python -u -O $LEGACYPIPE_DIR/legacypipe/runbrick.py \
       --brick $brick \
       --bands g,r2,i2,z,y \
       --pixscale 0.168 \
       --width 5600 --height 5600 \
       --rgb-stretch 1.5 \
       --sub-blobs \
       --no-wise \
       --skip-calibs \
       --checkpoint ${outdir}/checkpoints/${bri}/checkpoint-${brick}.pickle \
       --checkpoint-period 300 \
       --pickle "${outdir}/pickles/${bri}/runbrick-%(brick)s-%%(stage)s.pickle" \
       --outdir $outdir \
       --threads 32 \
       --stage image_coadds --minimal-coadds \
       >> $log 2>&1

#       --plots \
#
#--bands N501,N673 \
#     --force-all \
#       --stage image_coadds --minimal-coadds \
#       --stage image_coadds \
#       --run hsc-calexp \

# Save the return value from the python command -- otherwise we
# exit 0 because the rm succeeds!
status=$?
# /Config directory nonsense
rm -R $TMPCACHE
exit $status
