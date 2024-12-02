FROM ubuntu:22.04 as working

ENV DEBIAN_FRONTEND noninteractive

RUN apt -y update && apt install -y apt-utils \
 && apt install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    patch \
    git \
    file \
    pkg-config \
    less \
    procps \
    htop \
    gdb \
    strace \
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
    python3.10 \
    python3.10-dev \
    python3-pip \
    plplot-driver-cairo \
    source-extractor \
    psfex \
    scamp \
    # Remove APT files
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN apt -y update \
  && apt install -y --no-install-recommends \
      gfortran
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
    fitsio \
    matplotlib \
    astropy \
    photutils \
    zmq \
    mpi4py \
    ; do pip3 install $x; done \
    && rm -Rf /root/.cache/pip

RUN ln -s /usr/bin/python3.10-config /usr/local/bin/python3-config
RUN ln -s /usr/bin/python3 /usr/bin/python
ENV PYTHON=python3
ENV PYTHON_CONFIG=python3-config
ENV PYTHON_SCRIPT="/usr/bin/env python3"

RUN mkdir -p /src
WORKDIR /src

RUN ln -s /usr/include /usr/local/include/netpbm

# Astrometry.net
RUN git clone https://github.com/dstndstn/astrometry.net.git astrometry \
    && cd astrometry \
    && git checkout 0.94 \
    && make install INSTALL_DIR=/usr/local

#FROM working

# The Tractor
RUN git clone https://github.com/dstndstn/tractor.git tractor \
    && cd tractor \
    && pip install --install-option="--with-ceres" --install-option="--with-cython" -v .
#    && git checkout dr10.4 \

# unwise_psf
RUN git clone https://github.com/legacysurvey/unwise_psf.git \
    && cd unwise_psf \
    && git checkout dr10.0

# QDO
RUN git clone https://bitbucket.org/berkeleylab/qdo.git qdo \
    && cd qdo \
    && python3 setup.py install

# Legacypipe
RUN git clone https://github.com/legacysurvey/legacypipe.git legacypipe

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

RUN pip3 install --no-cache lacosmic

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

RUN python -O -m compileall \
    -x lib2to3 \
    /usr/local/lib/python3.10/dist-packages \
    /usr/lib/python3.10 \
    /usr/local/lib/python \
    /src/unwise_psf/py \
    /src/legacypipe/py/legacypipe \
    /src/legacypipe/py/legacyzpts && \
    python    -m compileall \
    -x lib2to3 \
    /usr/local/lib/python3.10/dist-packages \
    /usr/lib/python3.10 \
    /usr/local/lib/python \
    /src/unwise_psf/py \
    /src/legacypipe/py/legacypipe \
    /src/legacypipe/py/legacyzpts

# ugh gross
RUN ln -s /usr/bin/source-extractor /usr/bin/sex

# update legacypipe
RUN cd /src/legacypipe && git pull && \
  git checkout main && git pull && git describe && \
  python    -m compileall /src/legacypipe/py/{legacypipe,legacyzpts} && \
  python -O -m compileall /src/legacypipe/py/{legacypipe,legacyzpts}

RUN apt remove -y scamp && \
    apt install -y autoconf libtool libfftw3-dev libatlas-base-dev libcurl4-openssl-dev && \
    apt clean

RUN git clone https://github.com/dstndstn/scamp.git && \
    cd scamp && \
    git checkout add-extname && \
    ./autogen.sh && \
    ./configure && \
    make && \
    make install && \
    make clean

# update legacypipe
RUN cd /src/legacypipe && git pull && \
  git checkout DR10.3.2 && git describe && \
  python    -m compileall /src/legacypipe/py/{legacypipe,legacyzpts} && \
  python -O -m compileall /src/legacypipe/py/{legacypipe,legacyzpts}
