#! /bin/bash
# Set up the environment for using shifter in DR8 with any camera.

# Some NERSC-specific options to get MPI working properly.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export KMP_AFFINITY=disabled
export MPICH_GNI_FORK_MODE=FULLCOPY

# General environment variables
export DUST_DIR=/global/project/projectdirs/cosmo/data/dust/v0_1
export UNWISE_COADDS_DIR=/global/project/projectdirs/cosmo/work/wise/outputs/merge/neo4/fulldepth:/global/project/projectdirs/cosmo/data/unwise/allwise/unwise-coadds/fulldepth
export UNWISE_COADDS_TIMERESOLVED_DIR=/global/projecta/projectdirs/cosmo/work/wise/outputs/merge/neo4
export GAIA_CAT_DIR=/global/project/projectdirs/cosmo/work/gaia/chunks-gaia-dr2-astrom/
export GAIA_CAT_VER=2
export TYCHO2_KD_DIR=/global/project/projectdirs/cosmo/staging/tycho2
export LARGEGALAXIES_DIR=/global/project/projectdirs/cosmo/staging/largegalaxies/v2.0
export PS1CAT_DIR=/global/project/projectdirs/cosmo/work/ps1/cats/chunks-qz-star-v3/
UNWISE_PSF_DIR=/global/project/projectdirs/cosmo/staging/unwise_psf/2018-11-25
#UNWISE_PSF_DIR=/src/unwise_psf
export WISE_PSF_DIR=${UNWISE_PSF_DIR}/etc

unset PYTHONPATH
export PYTHONPATH=/usr/local/lib/python:/usr/local/lib/python3.6/dist-packages:.
export PYTHONPATH=$PYTHONPATH:$UNWISE_PSF_DIR/py

# Don't add ~/.local/ to Python's sys.path
export PYTHONNOUSERSITE=1
