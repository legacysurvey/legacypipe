#! /bin/bash

# ####
# #SBATCH --qos=premium
# #SBATCH --nodes=3
# #SBATCH --ntasks-per-node=32
# #SBATCH --cpus-per-task=2
# #SBATCH --time=48:00:00
# #SBATCH --licenses=SCRATCH
# #SBATCH -C haswell
# nmpi=96

# Half-subscribed, 96 tasks
##SBATCH --qos=premium
##SBATCH --time=48:00:00
#SBATCH -p debug
#SBATCH --time=30:00
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=4
#SBATCH --licenses=SCRATCH
#SBATCH -C haswell
#SBATCH --job-name 0744m640
#SBATCH --image=docker:legacysurvey/legacypipe:mpi
#SBATCH --module=mpich-cle6
nmpi=48
brick=0744m640

#nmpi=4
#brick=0309p335

#brick=$1

# This seem to be the default at NERSC?
# module load cray-mpich

#export PYTHONPATH=$(pwd):${PYTHONPATH}

outdir=/global/cscratch1/sd/dstn/dr9m-mpi

BLOB_MASK_DIR=/global/cfs/cdirs/cosmo/work/legacysurvey/dr8/south

export LEGACY_SURVEY_DIR=/global/cfs/cdirs/cosmo/work/legacysurvey/dr9m

export DUST_DIR=/global/cfs/cdirs/cosmo/data/dust/v0_1
export UNWISE_COADDS_DIR=/global/cfs/cdirs/cosmo/work/wise/outputs/merge/neo6/fulldepth:/global/cfs/cdirs/cosmo/data/unwise/allwise/unwise-coadds/fulldepth
export UNWISE_COADDS_TIMERESOLVED_DIR=/global/cfs/cdirs/cosmo/work/wise/outputs/merge/neo6
export UNWISE_MODEL_SKY_DIR=/global/cfs/cdirs/cosmo/work/wise/unwise_catalog/dr3/mod
export GAIA_CAT_DIR=/global/cfs/cdirs/cosmo/work/gaia/chunks-gaia-dr2-astrom-2
export GAIA_CAT_VER=2
export TYCHO2_KD_DIR=/global/cfs/cdirs/cosmo/staging/tycho2
export LARGEGALAXIES_CAT=/global/cfs/cdirs/cosmo/staging/largegalaxies/v3.0/SGA-ellipse-v3.0.kd.fits
export PS1CAT_DIR=/global/cfs/cdirs/cosmo/work/ps1/cats/chunks-qz-star-v3
export SKY_TEMPLATE_DIR=/global/cfs/cdirs/cosmo/work/legacysurvey/sky-templates

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

echo Logging to: $log
echo Running on $(hostname)

echo -e "\n\n\n" >> $log
echo "-----------------------------------------------------------------------------------------" >> $log
echo "PWD: $(pwd)" >> $log
echo >> $log
#echo "Environment:" >> $log
#set | grep -v PASS >> $log
#echo >> $log
ulimit -a >> $log
echo >> $log

echo -e "\nStarting on $(hostname)\n" >> $log
echo "-----------------------------------------------------------------------------------------" >> $log

# When I was trying mpi4py compiled with openmpi...
#mpirun -n $nmpi --map-by core --rank-by node \

# cray-mpich doesn't support this kind of --distribution
#srun -n $nmpi --distribution cyclic:cyclic

# Cray-mpich does round-robin placement of ranks on nodes with this setting -- good for memory load balancing.
export MPICH_RANK_REORDER_METHOD=0

srun -n $nmpi \
     shifter \
     python -u -O -m mpi4py.futures \
     /src/legacypipe/py/legacypipe/mpi-runbrick.py \
       --no-wise-ceres \
       --run south \
       --brick $brick \
       --skip \
       --skip-calibs \
       --blob-mask-dir ${BLOB_MASK_DIR} \
       --checkpoint ${outdir}/checkpoints/${bri}/checkpoint-${brick}.pickle \
       --wise-checkpoint ${outdir}/checkpoints/${bri}/wise-${brick}.pickle \
       --pickle "${outdir}/pickles/${bri}/runbrick-%(brick)s-%%(stage)s.pickle" \
       --outdir $outdir \
       >> $log 2>&1

# QDO_BATCH_PROFILE=cori-shifter qdo launch -v tst 1 --cores_per_worker 8 --walltime=30:00 --batchqueue=debug --keep_env --batchopts "--image=docker:dstndstn/legacypipe:intel" --script "/src/legacypipe/bin/runbrick-shifter.sh"
