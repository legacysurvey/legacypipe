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
    pybind11-dev \
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
    python3-pybind11 \
    python3-tk \
    python3-setuptools \
    python3-wheel \
    sextractor \
    psfex \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

#    python3-numpy \
#    python3-scipy \
#    python3-matplotlib \

# multiprocessing.pool from official python 3.6.8 release
# The python 3.6.7 version shipped in current Ubuntu 18.04 breaks the
# Astrometry.net "timingpool" class.  This changed from 3.6.6 to 3.6.7 and
# then was reverted in 3.6.8.
COPY pool.py /usr/lib/python3.6/multiprocessing/pool.py

# Python related stuff
RUN echo "../site-packages" > /usr/local/lib/python3.6/dist-packages/site-packages.pth

# Pip installs
RUN for x in \
    cython \
    numpy \
    scipy \
    matplotlib \
    fitsio \
    astropy \
    photutils \
    zmq \
    ; do \
    python3.6 -m pip install $x; \
    done; \
    rm -R /root/.cache/pip

RUN mkdir /src
WORKDIR /src

# Astrometry.net
RUN git clone http://github.com/dstndstn/astrometry.net.git astrometry \
    && cd astrometry \
    && make \
    && make extra \
    && make py PYTHON=python3.6 \
    && make install INSTALL_DIR=/usr/local PYTHON=python3.6 \
    && make clean

ENV PYTHON=python3

# The Tractor
RUN git clone http://github.com/dstndstn/tractor.git tractor \
    && cd tractor \
    && python3 setup.py build_ext --with-ceres --with-cython \
    && python3 setup.py install --with-ceres --with-cython \
    && rm -R $(find . -name "*.o" -o -name "*.so")

# python = python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# DUST maps
RUN mkdir -p dust/maps \
  && cd dust/maps \
  && wget -c http://portal.nersc.gov/project/cosmo/temp/dstn/travis-ci/maps/SFD_dust_4096_ngp.fits \
  && wget -c http://portal.nersc.gov/project/cosmo/temp/dstn/travis-ci/maps/SFD_dust_4096_sgp.fits
ENV DUST_DIR=/src/dust

# Unwise PSF
RUN git clone https://github.com/legacysurvey/unwise_psf.git \
    && cd unwise_psf \
    && git checkout dr9.3

ENV PYTHONPATH=/usr/local/lib/python:/src/unwise_psf/py:/src/legacypipe/py

RUN ln -s /usr/bin/sextractor /usr/bin/source-extractor
