FROM ubuntu:18.04

RUN apt -y update && apt install -y apt-utils

RUN DEBIAN_FRONTEND=noninteractive \
    apt install -y --no-install-recommends \
    make \
    gcc \
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
    g++ \
    python3 \
    python3-dev \
    python3-pip \
    python3-pil \
    # # Remove APT files
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# multiprocessing.pool from official python 3.6.8 release
COPY pool.py /usr/lib/python3.6/multiprocessing/pool.py

# Python related stuff
RUN echo "../site-packages" > /usr/local/lib/python3.6/dist-packages/site-packages.pth
RUN pip3 install --upgrade\
    pip
RUN for x in \
    setuptools \
    wheel \
    intel-numpy \
    intel-scipy \
    psycopg2 \
    fitsio \
    matplotlib \
    astropy \
    photutils \
    ; do pip3 install $x; done \
    && rm -Rf /root/.cache/pip

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

# Demo
# RUN cd /tmp && \
#    build-astrometry-index -d 3 -o index-9918.fits -P 18 -S mag -B 0.1 -s 0 -r 1 -I 9918 -M -i /usr/local/examples/tycho2-mag6.fits && \
#    echo "add_path .\ninparallel\nindex index-9918.fits" > 99.cfg && \
#    cat 99.cfg && \
#    solve-field --config 99.cfg /usr/local/examples/apod4.jpg  --continue

# The Tractor
RUN cd /src \
    && git clone http://github.com/dstndstn/tractor.git tractor \
    && cd tractor \
    && export PYTHON=python3 \
    && export PYTHON_CONFIG=python3-config \
    && make \
    && make ceres \
    && make install INSTALL_DIR=/usr/local \
    && rm -R $(find . -name "*.o" -o -name "*.so")

# Legacypipe
RUN cd /src \
    && git clone http://github.com/legacysurvey/legacypipe.git legacypipe

# Yeah no, this needs some work still
#    && python3 setup.py install

# Demo
#RUN cd /src/legacypipe/py \
#    && export PYTHONPATH=${PYTHONPATH}:/usr/local/lib/python:. \
#    && python3 test/runbrick_test.py travis

# QDO
RUN cd /src/ \
    && git clone https://bitbucket.org/berkeleylab/qdo.git qdo \
    && cd qdo \
    && python3 setup.py install

