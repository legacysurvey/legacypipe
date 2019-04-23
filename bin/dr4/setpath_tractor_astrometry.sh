#!/bin/bash

# source this file after running either {install,update}_tractor_astrometry.sh
if [ "$NERSC_HOST" == "cori" ]; then
    export TOP_DIR=/global/cscratch1/sd/desiproc/dr5_kaylan
elif [ "$NERSC_HOST" == "edison" ]; then
    export TOP_DIR=/scratch1/scratchdirs/desiproc/dr5_kaylan
fi
export CODE_DIR=${TOP_DIR}/code
export PATH=${CODE_DIR}/bin:${PATH}
export CPATH=${CODE_DIR}/include:${CPATH}
export LIBRARY_PATH=${CODE_DIR}/lib:${LIBRARY_PATH}
export LD_LIBRARY_PATH=${CODE_DIR}/lib:${LD_LIBRARY_PATH}
export PYTHONPATH=${CODE_DIR}/lib/python2.7/site-packages:${PYTHONPATH}

echo Ready to use bleeding edge astrometry.net and tractor!
