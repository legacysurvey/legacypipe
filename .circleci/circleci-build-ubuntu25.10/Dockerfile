FROM ubuntu:25.10

ENV DEBIAN_FRONTEND=noninteractive

RUN apt install -y gpgv && \
    apt -y update && apt install -y apt-utils gpgv && \
    apt install -y --no-install-recommends \
    build-essential \
    make \
    gcc \
    g++ \
    gfortran \
    git \
    ssh \
    file \
    pkg-config \
    wget \
    curl \
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
    libnetpbm-dev \
    libpng-dev \
    libeigen3-dev \
    libgoogle-glog-dev \
    libceres-dev \
    python3-pip \
    python3.13 \
    python3.13-dev \
    python3.14 \
    python3.14-dev \
    source-extractor \
    psfex \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Pip installs
RUN for x in \
    cython \
    numpy \
    scipy \
    fitsio \
    matplotlib \
    astropy \
    photutils \
    zmq \
    ; do \
    python3.13 -m pip install --no-cache-dir $x --break-system-packages; \
    python3.14 -m pip install --no-cache-dir $x --break-system-packages; \
    done

RUN mkdir /src
WORKDIR /src

# compile for a fairly old standard (for Perlmutter)
ENV ARCH_FLAGS=-march=x86-64-v3

RUN python3.13 -m pip install --no-cache-dir --break-system-packages setuptools \
 && python3.14 -m pip install --no-cache-dir --break-system-packages setuptools

# Astrometry.net
RUN git clone https://github.com/dstndstn/astrometry.net.git astrometry \
    && cd astrometry \
    && make \
    && make extra \
    && make py PYTHON=python3.13 PYTHON_CONFIG=python3.13-config \
    && make py PYTHON=python3.14 PYTHON_CONFIG=python3.14-config \
    && make install INSTALL_DIR=/usr/local PYTHON=python3.13 PYTHON_CONFIG=python3.13-config \
    && make install INSTALL_DIR=/usr/local PYTHON=python3.14 PYTHON_CONFIG=python3.14-config \
    && make clean

# python = python3
ENV PYTHON=python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# DUST maps
RUN mkdir -p dust/maps \
  && cd dust/maps \
  && wget -c https://portal.nersc.gov/project/cosmo/temp/dstn/travis-ci/maps/SFD_dust_4096_ngp.fits \
  && wget -c https://portal.nersc.gov/project/cosmo/temp/dstn/travis-ci/maps/SFD_dust_4096_sgp.fits
ENV DUST_DIR=/src/dust

# Unwise PSF
RUN git clone https://github.com/legacysurvey/unwise_psf.git

# The Tractor
RUN git clone https://github.com/dstndstn/tractor.git tractor \
    && cd tractor \
    && python3.13 setup.py build_ext --with-ceres --with-cython --inplace \
    && python3.14 setup.py build_ext --with-ceres --with-cython --inplace \
    && python3.13 -m pip install . --break-system-packages \
    && python3.14 -m pip install . --break-system-packages \
    && rm -R $(find . -name "*.o" -o -name "*.so") \
    && echo 1

ENV PYTHONPATH=/usr/local/lib/python:/src/unwise_psf/py:/src/legacypipe/py

RUN python3.14 -m pip install --break-system-packages --no-cache-dir coverage coveralls
