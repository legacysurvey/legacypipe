#!/bin/bash

# How to use:
# new code will be installed in "TOP_DIR"
# so ONLY need to set "TOP_DIR" to your scratch space
if [ "$NERSC_HOST" == "cori" ]; then
    export TOP_DIR=/global/cscratch1/sd/desiproc/dr5_kaylan
elif [ "$NERSC_HOST" == "edison" ]; then
    export TOP_DIR=/scratch1/scratchdirs/desiproc/dr5_kaylan
fi
export CODE_DIR=${TOP_DIR}/code
export REPO_DIR=${TOP_DIR}/repos
mkdir -p ${CODE_DIR} ${REPO_DIR} ${CODE_DIR}/lib/python2.7/site-packages

# Ted's desiconda repo can set CC,CXX, etc. for each NERSC machine
cd $REPO_DIR
git clone https://github.com/desihub/desiconda.git
source ${REPO_DIR}/desiconda/conf/${NERSC_HOST}-gcc-py27.sh
# Existing Build
module use /global/common/$NERSC_HOST/contrib/desi/modulefiles
export TED_VER="20170613-1.1.4"
module load desiconda/${TED_VER}-imaging

export TED_CONDA=/global/common/${NERSC_HOST}/contrib/desi/code/desiconda/${TED_VER}-imaging_conda
export TED_AUX=/global/common/${NERSC_HOST}/contrib/desi/code/desiconda/${TED_VER}-imaging_aux

# Build bleeding edge Tractor,Astrometry
# Point lib/python to lib/python2.7/site-packages
# Point lib64 to lib
ln -s ${CODE_DIR}/lib/python2.7/site-packages ${CODE_DIR}/lib/python
ln -s ${CODE_DIR}/lib ${CODE_DIR}/lib64

export PATH=${CODE_DIR}/bin:${PATH}
export CPATH=${CODE_DIR}/include:${CPATH}
export LIBRARY_PATH=${CODE_DIR}/lib:${LIBRARY_PATH}
export LD_LIBRARY_PATH=${CODE_DIR}/lib:${LD_LIBRARY_PATH}
export PYTHONPATH=${CODE_DIR}/lib/python2.7/site-packages:${PYTHONPATH}

# astrometry.net
export LDFLAGS="-L${TED_AUX}/lib -lz" 
export WCSLIB_INC="-I${TED_AUX}/include/wcslib" 
export WCSLIB_LIB="-L${TED_AUX}/lib -lwcs"
export JPEG_INC="-I${TED_AUX}/include" 
export JPEG_LIB="-L${TED_AUX}/lib -ljpeg" 
export CFITS_INC="-I${TED_AUX}/include" 
export CFITS_LIB="-L${TED_AUX}/lib -lcfitsio -lm" 

cd $REPO_DIR \
 && git clone https://github.com/dstndstn/astrometry.net.git \
    && cd astrometry.net  \
    && make \
    && make extra \
    && make install INSTALL_DIR="${CODE_DIR}" \
    && cd .. 

# tractor
export SUITESPARSE_LIB_DIR="${TED_AUX}/lib" 
export BLAS_LIB="-L/opt/intel/compilers_and_libraries_2017.1.132/linux/mkl/lib/intel64 -lmkl_rt -fopenmp -lpthread -lm -ldl" 

# for ceres
# eigen3.pc (ligglog.pc in ${TED_AUX}/lib/pkgconfig but not needed)
export PKG_CONFIG_PATH=${TED_AUX}/share/pkgconfig

cd $REPO_DIR \
 && git clone https://github.com/dstndstn/tractor.git \
    && cd tractor \
    && make \
    && make install INSTALL_DIR=${CODE_DIR} \
    && cd .. 

echo installed astrometry.net and tractor!
