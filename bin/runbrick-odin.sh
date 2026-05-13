#! /bin/bash

# Script for running the legacypipe code within a Shifter container at NERSC

#export LEGACY_SURVEY_DIR=/global/cfs/cdirs/cosmo/work/users/dstn/ODIN/

#outdir=$LEGACY_SURVEY_DIR/2band
#outdir=$LEGACY_SURVEY_DIR/xmm-N673
#outdir=$LEGACY_SURVEY_DIR/deep23-N419
#outdir=/global/cfs/cdirs/cosmo/work/users/dstn/ODIN/xmm-N419
#outdir=/global/cfs/cdirs/cosmo/work/users/dstn/ODIN/deep23-N419-2022-10

#field="$1"
#brick="$2"

#field=shela3456
field=shela
brick="$1"

outdir=$SCRATCH/odin-bricks-$field

unset BLOB_MASK_DIR
export COSMO=/dvs_ro/cfs/cdirs/cosmo

export LEGACY_SURVEY_DIR=$COSMO/work/users/dstn/ODIN/

#export UNWISE_COADDS_DIR=/global/cfs/cdirs/cosmo/work/wise/outputs/merge/neo6/fulldepth:/global/cfs/cdirs/cosmo/data/unwise/allwise/unwise-coadds/fulldepth
#export UNWISE_COADDS_TIMERESOLVED_DIR=/global/cfs/cdirs/cosmo/work/wise/outputs/merge/neo6
#export UNWISE_MODEL_SKY_DIR=/global/cfs/cdirs/cosmo/work/wise/unwise_catalog/dr3/mod

#export LARGEGALAXIES_CAT=$COSMO/staging/largegalaxies/v3.0/SGA-ellipse-v3.0.kd.fits
#export LARGEGALAXIES_CAT=$LEGACY_SURVEY_DIR/SGA-2020.kd.fits
export LARGEGALAXIES_CAT=$COSMO/work/legacysurvey/sga/2025/SGA2025-ellipse-v1.5-dr11-south.kd.fits

export DUST_DIR=$COSMO/data/dust/v0_1

#export GAIA_CAT_DIR=$COSMO/data/gaia/edr3/healpix
#export GAIA_CAT_VER=E
export GAIA_CAT_DIR=$COSMO/data/gaia/dr3/healpix
export GAIA_CAT_VER=3
export GAIA_CAT_PREFIX=healpix
export GAIA_CAT_SCHEME=nested
export TYCHO2_KD_DIR=$COSMO/staging/tycho2
export PS1CAT_DIR=$COSMO/work/ps1/cats/chunks-qz-star-v3
#export SKY_TEMPLATE_DIR=/global/cfs/cdirs/cosmo/work/legacysurvey/sky-templates

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

# CORI
#export PYTHONPATH=~/fitsio:${PYTHONPATH}
#python -c "import fitsio; print(fitsio.__file__)"

#--bands N501,N673 \
#     --force-all \

#export PYTHONPATH=.:${PYTHONPATH}
#cd $LEGACYPIPE_DIR

export LEGACYPIPE_DIR=/src/legacypipe/py

python -O $LEGACYPIPE_DIR/legacypipe/runbrick.py \
       --skip \
       --skip-calibs \
       --run odin \
       --brick $brick \
       --bands N419,N501,N673 \
       --forced-bands u,g,r,i,z \
       --nsatur 2 \
       --coadd-bw \
       --rgb-stretch 1.5 \
       --sub-blobs \
       --no-segmentation \
       --no-wise \
       --release 2605 \
       --threads 128 \
       --skip-calibs \
       --checkpoint ${outdir}/checkpoints/${bri}/checkpoint-${brick}.pickle \
       --checkpoint-period 300 \
       --pickle "${outdir}/pickles/${bri}/runbrick-%(brick)s-%%(stage)s.pickle" \
       --outdir $outdir \
       --write-stage srcs \
       >> $log 2>&1

#       --stage image_coadds \
#     --stage image_coadds --minimal-coadds \

# Save the return value from the python command -- otherwise we
# exit 0 because the rm succeeds!
status=$?
# /Config directory nonsense
rm -R $TMPCACHE
exit $status
