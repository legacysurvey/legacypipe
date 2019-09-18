#! /bin/bash

# To run the brick-summary code.  As of DR7, this couldn't be run
# on the command line.  Currently set up to be split into 360 tasks
# whose names are 000 to 359.  Code crashes if no bricks at that RA,
# but this failure is without consequence.

# Make sure this has the appropriate paths
#source $CSCRATCH/DRcode/legacypipe/bin/legacypipe-env
desiconda_version=20190311-1.2.7-img
module use /global/common/software/desi/cori/desiconda/$desiconda_version/modulefiles
module load desiconda
export PYTHONPATH=/global/cscratch1/sd/landriau/dr8/code/legacypipe/py:$PYTHONPATH

# Force MKL single-threaded
# https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

# To avoid problems with MPI and Python multiprocessing
export MPICH_GNI_FORK_MODE=FULLCOPY

# Try limiting memory to avoid killing the whole MPI job...
#ncores=1
# 128 GB / Cori Haswell node = 134217728 kbytes
#maxmem=134217728
#let usemem=${maxmem}*${ncores}/32
#ulimit -Sv $usemem

# Need to modify for future release and output directories
release=dr8
survey=south
drdir=/global/project/projectdirs/cosmo/work/legacysurvey/${release}/${survey}
outdir=/global/cscratch1/sd/landriau/${release}/${survey}/brick-summary
rundir=/global/cscratch1/sd/landriau/${release}/code

dirname="$1"

log="${outdir}/bs-${dirname}.log"

echo Logging to: $log
echo Running on ${NERSC_HOST} $(hostname)

echo -e "\n\n\n" >> $log
echo "-----------------------------------------------------------------------------------------" >> $log
echo "PWD: $(pwd)" >> $log
echo "Modules:" >> $log
module list >> $log 2>&1
echo >> $log
echo "Environment:" >> $log
set | grep -v PASS >> $log
echo >> $log
ulimit -a >> $log
echo >> $log

echo -e "\nStarting on ${NERSC_HOST} $(hostname)\n" >> $log
echo "-----------------------------------------------------------------------------------------" >> $log

python -u $rundir/legacypipe/py/legacyanalysis/brick-summary.py \
         -o ${outdir}/${release}-${survey}-bricks-summary-${dirname}.fits \
         ${drdir}/coadd/${dirname}/*/*-nexp-*.fits.fz \
         > $log 2>&1

# qdo launch bricksum 32 --cores_per_worker 1 --walltime=00:30:00 --script /global/cscratch1/sd/landriau/code/legacypipe/bin/qdo-brick-summary.sh --batchqueue debug --keep_env


