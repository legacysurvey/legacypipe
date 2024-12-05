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
    python3.7 \
    python3.7-dev \
    python3.8 \
    python3.8-dev \
    sextractor \
    psfex \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN python3.7 -m pip install setuptools \
 && python3.7 -m pip install pip --upgrade \
 && python3.8 -m pip install setuptools \
 && python3.8 -m pip install pip --upgrade

RUN apt remove -y python-pip-whl

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
    ; do \
    python3.7 -m pip install $x; \
    python3.8 -m pip install $x; \
    done; \
    rm -R /root/.cache/pip

RUN python3.7 -m pip install --no-cache-dir coverage coveralls

RUN mkdir /src
WORKDIR /src

# Astrometry.net
RUN git clone http://github.com/dstndstn/astrometry.net.git astrometry \
    && cd astrometry \
    && make \
    && make extra \
    && make py PYTHON=python3.7 \
    && make py PYTHON=python3.8 \
    && make install INSTALL_DIR=/usr/local PYTHON=python3.7 \
    && make install INSTALL_DIR=/usr/local PYTHON=python3.8 \
    && make clean \
    && echo 2

ENV PYTHON=python3.7
# python = python3.7
RUN ln -s /usr/bin/python3.7 /usr/bin/python

ENV PYTHONPATH=/usr/local/lib/python:/src/unwise_psf/py:/src/legacypipe/py

# The Tractor
RUN git clone http://github.com/dstndstn/tractor.git tractor \
    && cd tractor \
    && python3.7 setup.py build_ext --with-ceres --with-cython \
    && python3.7 setup.py install --with-ceres --with-cython \
    && python3.8 setup.py build_ext --with-ceres --with-cython \
    && python3.8 setup.py install --with-ceres --with-cython \
    && rm -R $(find . -name "*.o" -o -name "*.so")

# DUST maps
RUN mkdir -p dust/maps \
  && cd dust/maps \
  && wget -nv http://portal.nersc.gov/project/cosmo/temp/dstn/travis-ci/maps/SFD_dust_4096_ngp.fits \
  && wget -nv http://portal.nersc.gov/project/cosmo/temp/dstn/travis-ci/maps/SFD_dust_4096_sgp.fits
ENV DUST_DIR=/src/dust

# Unwise PSF
RUN git clone https://github.com/legacysurvey/unwise_psf.git \
    && cd unwise_psf \
    && git checkout dr9.3


