FROM ubuntu:20.04

RUN apt -y update && apt install -y apt-utils

RUN DEBIAN_FRONTEND=noninteractive \
    apt install -y --no-install-recommends \
    gcc \
    g++ \
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
    postgresql-common \
    libpq-dev \
    python3.8 \
    python3.8-dev \
    python3.8-tk \
    python3-pip \
    # Remove APT files
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# # multiprocessing.pool from official python 3.6.8 release
# # The python 3.6.7 version shipped in current Ubuntu 18.04 breaks the
# # Astrometry.net "timingpool" class.  This changed from 3.6.6 to 3.6.7 and
# # then was reverted in 3.6.8.
# COPY pool.py /usr/lib/python3.6/multiprocessing/pool.py

# Python related stuff
#RUN echo "../site-packages" > /usr/local/lib/python3.6/dist-packages/site-packages.pth

# Pip installs
RUN for x in \
    setuptools \
    wheel \
    cython \
    numpy \
    scipy \
    pillow \
    psycopg2 \
    fitsio \
    matplotlib \
    astropy \
    photutils \
    zmq \
    ; do pip3 install $x; done \
    && rm -Rf /root/.cache/pip

RUN ln -s /usr/bin/python3.8-config /usr/local/bin/python3-config
# python = python3
RUN ln -s /usr/bin/python3 /usr/bin/python

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
    && make clean

# The Tractor
RUN cd /src \
    && git clone http://github.com/dstndstn/tractor.git tractor \
    && cd tractor \
    && export PYTHON=python3 \
    && export PYTHON_CONFIG=python3-config \
    && python3 setup-cython.py --with-ceres install --home /usr/local \
    && make version && cp tractor/version.py /usr/local/lib/python/tractor/ \
    && rm -R $(find . -name "*.o" -o -name "*.so")

# QDO
RUN cd /src/ \
    && git clone https://bitbucket.org/berkeleylab/qdo.git qdo \
    && cd qdo \
    && python3 setup.py install

# Legacypipe
RUN cd /src \
    && git clone http://github.com/legacysurvey/legacypipe.git legacypipe
