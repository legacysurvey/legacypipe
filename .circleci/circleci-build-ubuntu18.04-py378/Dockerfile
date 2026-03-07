FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt -y update && apt install -y apt-utils && \
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
    libnetpbm10-dev \
    libpng-dev \
    libeigen3-dev \
    libgoogle-glog-dev \
    libceres-dev \
    python3 \
    python3-dev \
    python3-pip \
    python3-pil \
    python3-zmq \
    python3-tk \
    python3-wheel \
    python3.8 \
    python3.8-dev \
    sextractor \
    psfex \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

#RUN apt remove -y python-pip-whl

RUN python3.8 -m pip install --no-cache-dir --upgrade setuptools \
 && python3.8 -m pip install --no-cache-dir --upgrade pip \
 && python3.8 -m pip install --no-cache-dir wheel wheelfile

# Pip installs
RUN for x in \
    cython \
    pybind11 \
    extension_helpers \
    numpy \
    pythran \
    scipy \
    matplotlib \
    fitsio \
    astropy \
    photutils \
    coverage \
    coveralls \
    ; do \
    python3.8 -m pip install --no-cache-dir $x; \
    done;

RUN mkdir /src
WORKDIR /src

# Astrometry.net
RUN git clone http://github.com/dstndstn/astrometry.net.git astrometry \
    && cd astrometry \
    && make \
    && make extra \
    && make py PYTHON=python3.8 \
    && make install INSTALL_DIR=/usr/local PYTHON=python3.8 \
    && make clean

ENV PYTHON=python3.8
RUN ln -s /usr/bin/python3.8 /usr/bin/python

RUN ln -s /usr/bin/sextractor /usr/bin/source-extractor

# https://github.com/pypa/setuptools/issues/4501
#RUN python3.8 -m pip install setuptools==70.3.0
#RUN python3.8 -m pip install fitsio

# DUST maps
RUN mkdir -p dust/maps \
  && cd dust/maps \
  && wget -nv http://portal.nersc.gov/project/cosmo/temp/dstn/travis-ci/maps/SFD_dust_4096_ngp.fits \
  && wget -nv http://portal.nersc.gov/project/cosmo/temp/dstn/travis-ci/maps/SFD_dust_4096_sgp.fits
ENV DUST_DIR=/src/dust

# Unwise PSF
RUN git clone https://github.com/legacysurvey/unwise_psf.git \
    && cd unwise_psf \
    && git checkout dr10.0

ENV PYTHONPATH=/usr/local/lib/python:/src/unwise_psf/py:/src/legacypipe/py

# The Tractor
RUN git clone http://github.com/dstndstn/tractor.git tractor \
    && cd tractor \
    && python3.8 setup.py build_ext --with-ceres --with-cython --inplace \
    && python3.8 -m pip install . --break-system-packages \
    && rm -R $(find . -name "*.o" -o -name "*.so") && echo 1
