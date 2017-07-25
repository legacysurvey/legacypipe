#! /bin/bash

if [[ -n $BASH_VERSION ]]; then
    _SCRIPT_LOCATION=${BASH_SOURCE[0]}
elif [[ -n $ZSH_VERSION ]]; then
    _SCRIPT_LOCATION=${funcstack[1]}
else
    echo "Only bash and zsh are supported"
    return 1
fi

if [ x"$SLURM_JOB_NUM_NODES" == x ]; then
    echo "The script is avalaible from a job script only."
    echo "Use with sbatch (for batch scripts) or salloc (for interactive)."
    return 1
fi

DIRNAME=`dirname ${_SCRIPT_LOCATION}`

#source /usr/common/contrib/bccp/python-mpi-bcast/activate.sh /dev/shm/local "srun -n $SLURM_JOB_NUM_NODES"
# bcast
source /usr/common/contrib/bccp/python-mpi-bcast/activate.sh /dev/shm/local "$QDO_MPIRUN"
FILES=`find $DIRNAME -name '*.tar.gz'`

#HPCPMODULES=`find_hpcp_modules`
HPCPMODULES=$FILES
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/composer_xe_2015.1.133/mkl/lib/intel64

echo "found hpcport packages: $HPCPMODULES"
bcast -v $HPCPMODULES
#bcast -v /scratch2/scratchdirs/kaylanb/yu-bcase/zzzzz-easy-install.tar.gz
bcast -v /scratch2/scratchdirs/kaylanb/yu-bcase/zzzzz-scikitimage-easy-install.tar.gz
