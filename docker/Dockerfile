FROM ubuntu:24.04 as working

ENV DEBIAN_FRONTEND noninteractive

RUN apt -y update && apt install -y apt-utils \
 && apt install -y --no-install-recommends \
    gcc \
    g++ \
    gfortran \
    make \
    patch \
    git \
    file \
    pkg-config \
    autoconf \
    automake \
    libtool \
    less \
    procps \
    htop \
    gdb \
    strace \
    valgrind \
    vim \
    less \
    nano \
    imagemagick \
    wget \
    swig \
    netpbm \
    wcslib-dev \
    wcslib-tools \
    zlib1g-dev \
    libatlas-base-dev \
    libbz2-dev \
    libcairo2-dev \
    libcfitsio-dev \
    libcfitsio-bin \
    libcurl4-openssl-dev \
    libfftw3-dev \
    libgsl-dev \
    libjpeg-dev \
    libnetpbm10-dev \
    libpng-dev \
    libeigen3-dev \
    libgoogle-glog-dev \
    libceres-dev \
    postgresql-common \
    libpq-dev \
    python3.12 \
    python3.12-dev \
    python3-pip \
    plplot-driver-cairo \
    libplplot-dev \
    source-extractor \
    psfex \
    # Remove APT files
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

#    scamp \ -- Ubuntu currently (2025-04-14) shipping v 2.10; we need 2.14.0

ENV FFLAGS=-fallow-argument-mismatch
ENV FCFLAGS=-fallow-argument-mismatch

RUN wget -nv https://www.mpich.org/static/downloads/4.0.2/mpich-4.0.2.tar.gz \
  && tar xzf mpich-4.0.2.tar.gz \
  && cd mpich-4.0.2 \
  && ./configure \
  && make -j \
  && make install \
  && make clean \
  && cd .. && rm -Rf mpich-4.0.2
# defaults to installing in /usr/local

# Pip installs
RUN for x in \
    setuptools \
    wheel \
    cython \
    numpy \
    scipy \
    pillow \
    psycopg2 \
    matplotlib \
    lacosmic \
    astropy \
    photutils \
    zmq \
    mpi4py \
    ; do pip3 install --break-system-packages $x; done \
    && rm -Rf /root/.cache/pip

#    fitsio \ -- We need commit 273a00c (Feb 23/2025)

RUN ln -s /usr/bin/python3.12-config /usr/local/bin/python3-config
RUN ln -s /usr/bin/python3 /usr/bin/python
ENV PYTHON=python3
ENV PYTHON_CONFIG=python3-config
ENV PYTHON_SCRIPT="/usr/bin/env python3"

RUN mkdir -p /src
WORKDIR /src

# to help astrometry.net find netpbm (yuck)
RUN ln -s /usr/include /usr/local/include/netpbm

# Astrometry.net
ENV ARCH_FLAGS=-march=x86-64-v3
RUN git clone https://github.com/dstndstn/astrometry.net.git astrometry \
    && cd astrometry \
    && git checkout 0.97 \
    && make \
    && make py \
    && make install INSTALL_DIR=/usr/local

#FROM working

# The Tractor
RUN git clone https://github.com/dstndstn/tractor.git tractor \
    && cd tractor \
    && git checkout dr10.6 \
    && python setup.py build_ext --with-ceres --with-cython --inplace \
    && python setup.py install

# unwise_psf
RUN git clone https://github.com/legacysurvey/unwise_psf.git \
    && cd unwise_psf \
    && git checkout dr10.0

# QDO
RUN git clone https://bitbucket.org/berkeleylab/qdo.git qdo \
    && cd qdo \
    && python3 setup.py install

ENV PYTHONPATH /src/unwise_psf/py:/src/legacypipe/py:/usr/local/lib/python

SHELL ["/bin/bash", "-c"]
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["/bin/bash"]

# ?
RUN mkdir /homedir && chmod 777 /homedir
ENV HOME /homedir

# otherwise "fatal: detected dubious ownership in repository" upon 'git describe'
RUN git config --global --add safe.directory /src/legacypipe \
 && git config --global --add safe.directory /src/unwise_psf

RUN echo "export PS1='[container] \\u@\\h:\\w$ '" >> $HOME/.bashrc \
  # Create config files in $HOME
  && mkdir $HOME/.qdo && echo "[qdo]" > $HOME/.qdo/qdorc \
  && python -c "import astropy" \
  && python -c "import matplotlib.font_manager as fm; f = fm.FontManager()" \
  # Download astropy site locations and USNO sky model
  && python -c "from astropy.coordinates import EarthLocation; EarthLocation._get_site_registry(force_download=True)" \
  && python -c "from astropy.coordinates import EarthLocation, SkyCoord, AltAz; from astropy.time import Time; print(EarthLocation.of_site('ctio')); print(SkyCoord(180.,-45.,unit='deg').transform_to(AltAz(obstime=Time(56806.0, format='mjd'), location=EarthLocation.of_site('ctio'))))" \
  # Download astropy IERS leap-second list
  && python -c "from astropy.time import Time; Time.now()" \
  # Make astropy cache files readable!?!!
  && chmod -R a+rwX $HOME/.astropy

# Compile a local version of Scamp to pick up adding EXTNAME to the results:
# https://github.com/astromatic/scamp/commit/d936c73d7dd08382b70cb32ceb1a535ffca81732
RUN git clone https://github.com/astromatic/scamp.git && \
    cd scamp && \
    ./autogen.sh && \
    ./configure --enable-plplot && \
    make && \
    make install && \
    make clean

# Compile a local version of fitsio to pick up the ability to set the FPACK compression seed:
# https://github.com/esheldon/fitsio/commit/273a00cf2770c989d35babc8b95d44dbbc4d2096
RUN git clone https://github.com/esheldon/fitsio.git \
    && cd fitsio \
    && pip install -e . --break-system-packages

# Legacypipe
RUN git clone https://github.com/legacysurvey/legacypipe.git legacypipe

RUN python -O -m compileall \
    -x lib2to3 \
    /usr/local/lib/python3.12/dist-packages \
    /usr/lib/python3.12 \
    /usr/local/lib/python \
    /src/unwise_psf/py \
    /src/legacypipe/py/legacypipe \
    /src/legacypipe/py/legacyzpts && \
    python    -m compileall \
    -x lib2to3 \
    /usr/local/lib/python3.12/dist-packages \
    /usr/lib/python3.12 \
    /usr/local/lib/python \
    /src/unwise_psf/py \
    /src/legacypipe/py/legacypipe \
    /src/legacypipe/py/legacyzpts

# # 
# # update legacypipe
# #  git checkout DR10.3.4 && git describe && \
# RUN cd /src/legacypipe && git pull && \
#   python    -m compileall /src/legacypipe/py/{legacypipe,legacyzpts} && \
#   python -O -m compileall /src/legacypipe/py/{legacypipe,legacyzpts}
