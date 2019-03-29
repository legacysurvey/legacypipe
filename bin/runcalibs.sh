#! /bin/bash

echo "runcalibs.sh: script is: $0"

starttime=$(date +%s.%N)

source $(dirname $0)/legacypipe-env
export LEGACY_SURVEY_DIR=/global/project/projectdirs/cosmo/work/legacysurvey/dr8b

# Don't add ~/.local/ to Python's sys.path
export PYTHONNOUSERSITE=1
# Force MKL single-threaded
# https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
# To avoid problems with MPI and Python multiprocessing
export MPICH_GNI_FORK_MODE=FULLCOPY
export KMP_AFFINITY=disabled

# Try limiting memory to avoid killing the whole MPI job...
ncores=8
if [ "$NERSC_HOST" = "edison" ]; then
    # 64 GB / Edison node = 67108864 kbytes
    maxmem=67108864
    let usemem=${maxmem}*${ncores}/24
else
    # 128 GB / Cori Haswell node = 134217728 kbytes
    maxmem=134217728
    let usemem=${maxmem}*${ncores}/32
fi
ulimit -Sv $usemem

#outdir=/global/project/projectdirs/cosmo/work/legacysurvey/dr8
outdir=cal

export camera=decam

image_fn=$1

log=${outdir}/$(basename -s .fits.fz $image_fn).log

export PYTHONPATH=${PYTHONPATH}:.

echo "python legacyzpts/legacy_zeropoints.py \
	   --camera ${camera} --image ${image_fn} --outdir ${outdir} \
       --image_dir $LEGACY_SURVEY_DIR/images \
       --overhead ${starttime} --threads ${ncores} >> $log"
       >> $log

#cd $zpts_code/legacyzpts/py
python legacyzpts/legacy_zeropoints.py \
	--camera ${camera} --image ${image_fn} --outdir ${outdir} \
    --image_dir $LEGACY_SURVEY_DIR/images \
    --threads ${ncores} \
    --overhead ${starttime} \
    >> $log 2>&1

