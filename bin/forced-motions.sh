#! /bin/bash

brick=$1

out_dir=$SCRATCH/forced-motions-dr10

# On Perlmutter, RO-CFS is available on /dvs_ro/cfs both inside and outside shifter containers
# (and on the login nodes too)
export COSMO=/dvs_ro/cfs/cdirs/cosmo

dr10=$COSMO/work/legacysurvey/dr10
cat_dir=$dr10
survey_dir=$dr10

export SKY_TEMPLATE_DIR=$COSMO/work/legacysurvey/dr10/calib/sky_pattern
export LARGEGALAXIES_CAT=$COSMO/staging/largegalaxies/v3.0/SGA-ellipse-v3.0.kd.fits
export GAIA_CAT_DIR=$COSMO/data/gaia/edr3/healpix
export GAIA_CAT_PREFIX=healpix
export GAIA_CAT_SCHEME=nested
export GAIA_CAT_VER=E

# Don't add ~/.local/ to Python's sys.path
export PYTHONNOUSERSITE=1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export KMP_AFFINITY=disabled
export MPICH_GNI_FORK_MODE=FULLCOPY

bri=${brick:0:3}
mkdir -p ${out_dir}/logs-forced/${bri}
echo Logging to ${out_dir}/logs-forced/${bri}/${brick}.log

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

#export PYTHONPATH=.:${PYTHONPATH}

python -O $LEGACYPIPE_DIR/legacypipe/forced_photom_brickwise.py \
       --brick $brick \
       --survey-dir ${survey_dir} \
       --catalog-dir ${cat_dir} \
       --outdir ${out_dir} \
       --bands g,r,i,z \
       --derivs \
       --threads 16 \
       >> ${out_dir}/logs-forced/${bri}/${brick}.log 2>&1

# Save the return value from the python command -- otherwise we
# exit 0 because the rm succeeds!
status=$?
# /Config directory nonsense
rm -R $TMPCACHE
exit $status
