#! /bin/bash
brick=$1
#python legacypipe/forced_photom_brickwise.py --catalog-dir $CSCRATCH/cfis-ls --survey-dir ~/cosmo/work/users/dstn/dr9/north -d $CSCRATCH/cfis-ls/forced-u -b $brick --threads 8

#export COSMO=/global/cfs/cdirs/cosmo
export COSMO=/dvs_ro/cfs/cdirs/cosmo

#out_dir=$SCRATCH/cfis-xmm-forced-from-dr9
#cat_dir=$COSMO/data/legacysurvey/dr9/south
#survey_dir=/global/cfs/cdirs/desi/users/dstn/cfis-dir

# out_dir=$SCRATCH/cfis-xmm-forced-on-dr9
# cat_dir=$SCRATCH/cfis-xmm
# survey_dir=$COSMO/work/legacysurvey/dr9

# out_dir=$SCRATCH/cfis-elais/forced-on-cfisu+dr9
# cat_dir=$SCRATCH/cfis-elais
# survey_dir=/global/cfs/cdirs/desi/users/dstn/cfis+dr9-dir

# out_dir=$SCRATCH/cfis-elais-u/forced-dr9
# cat_dir=$COSMO/data/legacysurvey/dr9/north
# survey_dir=/global/cfs/cdirs/desi/users/dstn/cfis-dir

out_dir=$SCRATCH/cfis-s82-u/forced-dr9
cat_dir=$COSMO/data/legacysurvey/dr9/north
survey_dir=/global/cfs/cdirs/desi/users/dstn/cfis-dir

mkdir -p ${out_dir}/logs-forced
echo Logging to ${out_dir}/logs-forced/${brick}.log

#export SKY_TEMPLATE_DIR=$COSMO/work/legacysurvey/dr10/calib/sky_pattern
unset SKY_TEMPLATE_DIR

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# Don't add ~/.local/ to Python's sys.path
export PYTHONNOUSERSITE=1
# To avoid problems with MPI and Python multiprocessing
export MPICH_GNI_FORK_MODE=FULLCOPY
export KMP_AFFINITY=disabled

unset LARGEGALAXIES_CAT

export PYTHONPATH=.:${PYTHONPATH}

python -u -O legacypipe/forced_photom_brickwise.py \
       --brick $brick \
       --survey-dir ${survey_dir} \
       --catalog-dir ${cat_dir} \
       --outdir ${out_dir} \
       --threads 8 \
       --bands u \
       >> ${out_dir}/logs-forced/${brick}.log 2>&1

#       --bands u,g,i,z \
#       --bands g,r,z \
#       --bands u \
