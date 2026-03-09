FROM ubuntu:24.04 AS working

ENV DEBIAN_FRONTEND=noninteractive

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

# patchelf to fix NERSC stuff; Cuda
RUN apt -y update \
 && apt install -y patchelf \
 && apt install -y libnvrtc12 libcudart12 libcurand10 libcusolver11 libnccl2 \
    libcufft11 libcusparse12 \
 && apt clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

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
    fitsio \
    requests \
    django \
    cupy-cuda12x \
; do pip3 install --break-system-packages --no-cache $x; done

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
    && git checkout 0.98 \
    && make \
    && make py \
    && make install INSTALL_DIR=/usr/local

#FROM working

# The Tractor
RUN git clone https://github.com/dstndstn/tractor.git tractor \
    && cd tractor \
    && python setup.py build_ext --with-ceres --with-cython --inplace \
    && pip install . --break-system-packages

# unwise_psf
RUN git clone https://github.com/legacysurvey/unwise_psf.git \
    && cd unwise_psf \
    && git checkout dr10.0

# QDO - dustin's version
RUN git clone https://bitbucket.org/dstn/qdo.git \
    && cd qdo \
    && git checkout ssh-srun \
    && python3 setup.py install

# ?
RUN mkdir /homedir && chmod 777 /homedir
ENV HOME=/homedir

# otherwise "fatal: detected dubious ownership in repository" upon 'git describe'
RUN git config --global --add safe.directory /src/legacypipe \
 && git config --global --add safe.directory /src/unwise_psf \
 && git config --global --add safe.directory /src/tractor

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
# (that's in 2.14.0; ubuntu currently has 2.10.0 (as of 2026-03-07))
RUN git clone https://github.com/astromatic/scamp.git && \
    cd scamp && \
    ./autogen.sh && \
    ./configure --enable-plplot && \
    make && \
    make install && \
    make clean

# NERSC's shifter "mpich" module includes a bunch of BROKEN libraries that interfere
# with executables we have installed.
RUN patchelf --force-rpath --set-rpath /lib/x86_64-linux-gnu/ /usr/local/bin/listhead
RUN patchelf --force-rpath --set-rpath /lib/x86_64-linux-gnu/ /usr/local/bin/modhead

# Add the stuff that John needs from the "cutouts" container
RUN git clone https://github.com/legacysurvey/imagine.git \
 && cd imagine/viewer \
 && ln -s settings_test.py settings.py \
 && cd /src/imagine/ \
 && mv data data-orig \
 && ln -s /dvs_ro/cfs/cdirs/cosmo/webapp/viewer/data . \
 && cp /src/imagine/docker/cutouts/cutout /usr/local/bin/ \
 && chmod 755 /usr/local/bin/cutout

ENV PYTHONPATH=/src/unwise_psf/py:/src/legacypipe/py:/usr/local/lib/python:/src/imagine
SHELL ["/bin/bash", "-c"]
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["/bin/bash"]

ENV LEGACYPIPE_DIR=/src/legacypipe/py

RUN python -O -m compileall \
    -x lib2to3 \
    /usr/local/lib/python3.12/dist-packages \
    /usr/lib/python3.12 \
    /usr/local/lib/python \
    /src/unwise_psf/py && \
    python    -m compileall \
    -x lib2to3 \
    /usr/local/lib/python3.12/dist-packages \
    /usr/lib/python3.12 \
    /usr/local/lib/python \
    /src/unwise_psf/py

RUN cd tractor \
  && git fetch \
  && git pull \
  && python setup.py build_ext --with-ceres --with-cython --inplace \
  && pip install . --break-system-packages \
  && echo dr10.9 && git describe

RUN git clone https://github.com/legacysurvey/legacypipe.git \
  && cd legacypipe \
  && git pull \
  && python    -m compileall /src/legacypipe/py/{legacypipe,legacyzpts} \
  && python -O -m compileall /src/legacypipe/py/{legacypipe,legacyzpts} \
  && git describe \
  && echo 3
