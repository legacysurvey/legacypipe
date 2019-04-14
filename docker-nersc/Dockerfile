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

# Pip installs -- binary
RUN for x in \
    setuptools \
    wheel \
    intel-numpy \
    intel-scipy \
    psycopg2-binary \
    matplotlib \
    astropy \
    zmq \
    ; do pip3 install $x; done \
    && rm -Rf /root/.cache/pip

# ugly: Tractor's mp_fourier module detects the $CC='icc' exactly
# These are in /opt/intel/bin, which is on the PATH.
ENV CC icc
ENV CXX icpc
ENV LDSHARED icc -shared
ENV CFLAGS -O3 -g -fPIC -std=gnu99 -pthread -x=ivybridge -ax=haswell,knl
ENV ARCH_FLAGS ""
#ENV CFLAGS -O3 -g -fPIC -xAVX -axMIC-AVX512 -std=gnu99 -pthread
#ENV CXXFLAGS -O3 -g -fPIC -xAVX -axMIC-AVX512 -std=c++11 -pthread

# Pip installs -- built from source
RUN for x in \
    photutils \
    fitsio \
    ; do pip3 install $x; done \
    && rm -Rf /root/.cache/pip

ENV PYTHON python3
ENV PYTHON_CONFIG python3-config
ENV PYTHON_SCRIPT "/usr/bin/env python3"

RUN mkdir -p /src
WORKDIR /src

ENV BLAS -L${MKLROOT}/lib/intel64 -lmkl_rt -fopenmp -lpthread -limf -lsvml -ldl
ENV CPPFLAGS -I${MKLROOT}/include

RUN git clone https://github.com/astromatic/sextractor.git sourceextractor \
    && cd sourceextractor \
    && sh autogen.sh \
    && ./configure --enable-icc --enable-mkl \
    && make \
    && make install \
    && make clean

RUN git clone https://github.com/astromatic/psfex.git psfex \
    && cd psfex \
    && sh autogen.sh \
    && ./configure --enable-icc --enable-mkl \
    && make \
    && make install \
    && make clean

# Astrometry.net
RUN git clone http://github.com/dstndstn/astrometry.net.git astrometry \
    && cd astrometry \
    && make -j4 && make \
    && make py \
    && make extra \
    && make install INSTALL_DIR=/usr/local \
    && make clean \
    && echo 2

# The Tractor
RUN git clone http://github.com/dstndstn/tractor.git tractor \
    && cd tractor \
    && make \
    && make ceres \
    && make install INSTALL_DIR=/usr/local \
    && rm -R $(find . -name "*.o" -o -name "*.so")

######## Stage 2 ########
FROM registry.services.nersc.gov/nersc/intel_cxx_fort_mpi_mkl_runtime

ENV PYTHON python3
ENV PYTHON_CONFIG python3-config
ENV PYTHON_SCRIPT "/usr/bin/env python3"

RUN rm /root/.profile

RUN apt -y update && apt install -y apt-utils && echo yes

RUN DEBIAN_FRONTEND=noninteractive \
    apt install -y --no-install-recommends \
    procps \
    htop \
    make \
    git \
    file \
    pkg-config \
    gdb \
    vim \
    less \
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

RUN pip3 install cython

COPY --from=builder /usr/local /usr/local

WORKDIR /src

# QDO
RUN git clone https://bitbucket.org/berkeleylab/qdo.git qdo \
    && cd qdo \
    && python3 setup.py install

# python = python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# unwise_psf
RUN git clone https://github.com/legacysurvey/unwise_psf.git \
    && cd unwise_psf && git checkout dr8

# Legacypipe
RUN git clone http://github.com/legacysurvey/legacypipe.git legacypipe

ENV PYTHONPATH /usr/local/lib/python:/src/unwise_psf/py:/src/legacypipe/py
ENV WISE_PSF_DIR /src/unwise_psf/etc

# set prompt and default shell
SHELL ["/bin/bash", "-c"]
RUN echo "export PS1='[container] \\u@\\h:\\w$ '" >> /root/.bashrc
# Create config files in /root
RUN mkdir /root/.qdo && echo "[qdo]" > /root/.qdo/qdorc
RUN python -c "import astropy"

RUN    python    -m compileall /src/unwise_psf/py/unwise_psf.py \
    && python -O -m compileall /src/unwise_psf/py/unwise_psf.py

RUN python -O -m compileall /usr/lib/python3.5 /usr/lib/python3/dist-packages /usr/local/lib/python /usr/local/lib/python3.5/site-packages /usr/local/lib/python3.5/dist-packages && \
    python    -m compileall /usr/lib/python3.5 /usr/lib/python3/dist-packages /usr/local/lib/python /usr/local/lib/python3.5/site-packages /usr/local/lib/python3.5/dist-packages
    
RUN cd /src/legacypipe/py && git pull && git describe && echo 4

RUN    python    -m compileall /src/legacypipe/py/{legacypipe,legacyzpts} \
    && python -O -m compileall /src/legacypipe/py/{legacypipe,legacyzpts}

