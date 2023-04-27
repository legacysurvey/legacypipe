FROM ubuntu:22.04

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
    python3.10 \
    python3.10-dev \
    python3-pip \
    source-extractor \
    psfex \
    # Remove APT files
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

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

RUN ln -s /usr/bin/python3.10-config /usr/local/bin/python3-config
# python = python3
RUN ln -s /usr/bin/python3 /usr/bin/python
ENV PYTHON=python3
ENV PYTHON_CONFIG=python3-config
ENV PYTHON_SCRIPT="/usr/bin/env python3"

RUN mkdir -p /src
WORKDIR /src

# Astrometry.net
RUN git clone https://github.com/dstndstn/astrometry.net.git astrometry \
    && cd astrometry \
    && pip install -v . && echo 3

# The Tractor
RUN git clone https://github.com/dstndstn/tractor.git tractor \
    && cd tractor \
    && pip install --install-option="--with-ceres" --install-option="--with-cython" -v .

# unwise_psf
RUN git clone https://github.com/legacysurvey/unwise_psf.git \
    && cd unwise_psf && git checkout dr9.3

# QDO
RUN git clone https://bitbucket.org/berkeleylab/qdo.git qdo \
    && cd qdo \
    && python3 setup.py install

# Legacypipe
RUN git clone https://github.com/legacysurvey/legacypipe.git legacypipe

ENV PYTHONPATH /src/unwise_psf/py:/src/legacypipe/py

# ?
# RUN mkdir /homedir && chmod 777 /homedir
# ENV HOME /homedir

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
    /src/unwise_psf/py \
    /src/legacypipe/py/legacypipe \
    /src/legacypipe/py/legacyzpts && \
    python    -m compileall \
    -x lib2to3 \
    /usr/local/lib/python3.10/dist-packages \
    /usr/lib/python3.10 \
    /src/unwise_psf/py \
    /src/legacypipe/py/legacypipe \
    /src/legacypipe/py/legacyzpts
