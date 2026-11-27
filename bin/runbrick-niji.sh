#! /bin/bash

outdir=/data/dstn/niji-cosmos-1

export LEGACY_SURVEY_DIR=/data/dstn/niji-dir

unset LARGEGALAXIES_CAT
unset BLOB_MASK_DIR
unset SKY_TEMPLATE_DIR

export GAIA_CAT_DIR=/data/dstn/gaia-dr3
export GAIA_CAT_PREFIX=healpix
export GAIA_CAT_SCHEME=nested
export GAIA_CAT_VER=3

export DUST_DIR=/data/dstn/dust-v0_1
export TYCHO2_KD_DIR=/data/dstn/tycho2
export PS1CAT_DIR=/data/dstn/ps1/chunks-qz-star-v3

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

echo Logging to: $log
echo Running on $(hostname)

echo -e "\n\n\n" >> $log
echo "-----------------------------------------------------------------------------------------" >> $log
echo -e "\nStarting on $(hostname)\n" >> $log
echo "-----------------------------------------------------------------------------------------" >> $log

export LEGACYPIPE_DIR=/home/dstn/legacypipe/py
#export PYTHONPATH=.:${PYTHONPATH}
#python -O $LEGACYPIPE_DIR/legacypipe/runbrick.py \

python -O legacypipe/runbrick.py \
       --brick $brick \
       --bands M413,M439,M465,M490 \
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
       --threads 56 \
       >> $log 2>&1

#       --stage image_coadds --minimal-coadds \
#       --write-stage outliers \

# Save the return value from the python command -- otherwise we
# exit 0 because the rm succeeds!
status=$?
# /Config directory nonsense
#rm -R $TMPCACHE
exit $status
