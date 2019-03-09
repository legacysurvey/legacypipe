#FROM registry.services.nersc.gov/nersc/intel_cxx_devel as builder
FROM registry.services.nersc.gov/nersc/intel_cxx_fort_mpi_mkl_devel as builder

RUN apt -y update && apt install -y apt-utils && echo yes

RUN DEBIAN_FRONTEND=noninteractive \
    apt install -y --no-install-recommends \
    make \
    git \
    file \
    pkg-config \
    wget \
    swig \
    netpbm \
    wcslib-dev \
    wcslib-tools \
    zlib1g-dev \
    libbz2-dev \
    libcairo2-dev \
    libcfitsio-dev \
    libcfitsio-bin \
    libgsl-dev \
    libjpeg-dev \
    libnetpbm10-dev \
    libpng-dev \
    libeigen3-dev \
    libgoogle-glog-dev \
    libceres-dev \
    python3 \
    python3-dev \
    python3-pip \
    python3-pil \
    python3-tk \
    # # Remove APT files
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN echo "../site-packages" > /usr/local/lib/python3.5/dist-packages/site-packages.pth

# Pip installs
RUN for x in \
    setuptools \
    wheel \
    intel-numpy \
    intel-scipy \
    psycopg2 \
    matplotlib \
    astropy \
    photutils \
    fitsio \
    zmq \
    ; do pip3 install $x; done \
    && echo 1

## NOTE -- intel compilers only set *after* pip installs --
# photutils didn't work with these cflags, and we don't care
# that much that python codes are compiled with icc.

# Astrometry.net
RUN mkdir -p /src \
    && export PYTHON=python3 \
    && export PYTHON_SCRIPT="/usr/bin/env python3" \
    && cd /src \
    && git clone http://github.com/dstndstn/astrometry.net.git astrometry \
    && cd astrometry \
    && make \
    && make py \
    && make extra \
    && make install INSTALL_DIR=/usr/local \
    && make clean && echo up

# The Tractor's mp_fourier module detects the $CC='icc' exactly (which is ugly!)
# /opt/intel/bin
ENV CC icc
ENV CXX icpc
ENV LDSHARED icc -shared
ENV CFLAGS -O3 -g -fPIC -xAVX -axMIC-AVX512 -std=gnu99 -pthread
ENV CXXFLAGS -O3 -g -fPIC -xAVX -axMIC-AVX512 -std=c++11 -pthread
ENV PYTHON python3
ENV PYTHON_CONFIG python3-config
ENV PYTHON_SCRIPT "/usr/bin/env python3"

# The Tractor
RUN cd /src \
    && git clone http://github.com/dstndstn/tractor.git tractor \
    && cd tractor \
    #&& export PYTHON=python3 \
    #&& export PYTHON_CONFIG=python3-config \
    && make \
    && make ceres \
    && make install INSTALL_DIR=/usr/local \
    && rm -R $(find . -name "*.o" -o -name "*.so")

#patch -p1 < ../rules/patch_sextractor
# CC="@CC@" CFLAGS="@CFLAGS@" CPPFLAGS="-I@BLAS_INCLUDE@" ./configure @CROSS@ \
# --with-lapacke="@LAPACK@ @BLAS@" --prefix="@AUX_PREFIX@" \

ENV BLAS = -L${MKLROOT}/lib/intel64 -lmkl_rt -fopenmp -lpthread -limf -lsvml -ldl
ENV CPPFLAGS -I${MKLROOT}/include

RUN mkdir -p /src \
    && cd /src \
    && git clone https://github.com/astromatic/sextractor.git sourceextractor \
    && cd sourceextractor \
    && sh autogen.sh

RUN cd /src/sourceextractor \
    && ./configure --enable-icc --enable-mkl \
    && make \
    && make install

RUN cd /src \
    && git clone https://github.com/astromatic/psfex.git psfex \
    && cd psfex \
    && sh autogen.sh

RUN cd /src/psfex \
    && ./configure --enable-icc --enable-mkl \
    && make \
    && make install




######## Stage 2 ########
# ... don't have to clean above, just copy over installation
#FROM registry.services.nersc.gov/nersc/intel_cxx_runtime
FROM registry.services.nersc.gov/nersc/intel_cxx_fort_mpi_mkl_runtime
#COPY --from=builder /opt/intel/ /opt/intel/
COPY --from=builder /usr/local /usr/local
#RUN echo '/opt/local/lib' > /etc/ld.so.conf.d/package-libs.conf && \
#    ldconfig

ENV PYTHON python3
ENV PYTHON_CONFIG python3-config
ENV PYTHON_SCRIPT "/usr/bin/env python3"

RUN rm /root/.profile

RUN apt -y update && apt install -y apt-utils && echo yes

RUN DEBIAN_FRONTEND=noninteractive \
    apt install -y --no-install-recommends \
    procps \
    make \
    git \
    file \
    pkg-config \
    wget \
    netpbm \
    wcslib-dev \
    wcslib-tools \
    libgsl-dev \
    libcfitsio-dev \
    libcfitsio-bin \
    libgoogle-glog-dev \
    libceres-dev \
    python3 \
    python3-dev \
    python3-pip \
    python3-pil \
    python3-tk \
    # # Remove APT files
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# QDO
RUN mkdir -p /src \
    && cd /src/ \
    && git clone https://bitbucket.org/berkeleylab/qdo.git qdo \
    && cd qdo \
    && python3 setup.py install

# python = python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# Legacypipe
RUN cd /src \
    && git clone http://github.com/legacysurvey/legacypipe.git legacypipe \
    && echo 1

# Legacyzpts
RUN cd /src \
    && git clone https://github.com/legacysurvey/legacyzpts.git
    #&& cd /src/legacypipe/py \
    #&& ln -s /src/legacyzpts/py/legacyzpts .

# unwise_psf
RUN cd /src \
    && git clone https://github.com/legacysurvey/unwise_psf.git

ENV PYTHONPATH /usr/local/lib/python:/src/unwise_psf/py:/src/legacyzpts/py:/src/legacypipe/py
ENV WISE_PSF_DIR /src/unwise_psf/etc

#RUN apt update && apt install -y procps && apt clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# up!
RUN cd /src/legacypipe && git pull && echo 3

