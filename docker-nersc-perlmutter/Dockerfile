FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt -y update && apt install -y apt-utils && \
    apt install -y --no-install-recommends ca-certificates && \
    apt install -y --no-install-recommends \
    make \
    git \
    file \
    patch \
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
    procps \
    htop \
    gdb \
    strace \
    vim \
    less \
    imagemagick \
    python3.10-dev \
    libpython3.10-dev \
    python3-pip \
    libopenblas64-0-serial \
    libopenblas64-serial-dev \
    libopenblas64-0 \
    python3-numpy \
    python3-scipy \
    python3-fitsio \
    python3-astropy \
    python3-matplotlib \
    python3-zmq \
    python3-psycopg2 \
    python3-photutils \
    python3-ipython \
    python3-healpy \
    python3-seaborn \
    python3-ipykernel \
    source-extractor \
    psfex \
    gfortran \
    # Remove APT files
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Perlmutter CPU-only nodes: 2 x AMD Milan EPYC 7763
# https://developer.amd.com/wordpress/media/2020/04/Compiler%20Options%20Quick%20Ref%20Guide%20for%20AMD%20EPYC%207xx2%20Series%20Processors.pdf

ENV BASE_CFLAGS="-O3 -g -fPIC -std=gnu99 -pthread" \
    REAL_CFLAGS="$BASE_CFLAGS -march=znver2" \
    CFLAGS="$REAL_CFLAGS" \
    ARCH_FLAGS="" \
    FFLAGS="-fallow-argument-mismatch -O2"

# FFLAGS is for mpich

# -L/libm-install-dir/lib -lamdlibm

ENV PYTHON="python3" \
    PYTHON_CONFIG="python3-config" \
    PYTHON_SCRIPT="/usr/bin/env python3"

RUN mkdir -p /src
WORKDIR /src

# ENV BLAS="-L${MKLROOT}/lib/intel64 -lmkl_rt -fopenmp -lpthread -limf -lsvml -ldl" \
#     CPPFLAGS="-I${MKLROOT}/include"

RUN wget -nv https://www.mpich.org/static/downloads/3.3/mpich-3.3.tar.gz \
    && tar xzf mpich-3.3.tar.gz \
    && cd mpich-3.3 \
    && ./configure \
    && make -j4 \
    && make install \
    && make clean \
    && cd .. && rm -Rf mpich-3.3
# defaults to installing in /usr/local

RUN pip --no-cache-dir install cython

# mpi4py / mpicc can't handle -ax=knl or -x=haswell (it compiles using gcc?)
ENV CFLAGS $BASE_CFLAGS
RUN git clone https://github.com/mpi4py/mpi4py.git \
         && (cd mpi4py \
         && python3 setup.py build \
         && python3 setup.py install) \
         && rm -Rf mpi4py
ENV CFLAGS $REAL_CFLAGS

# Astrometry.net
RUN git clone http://github.com/dstndstn/astrometry.net.git astrometry \
    && cd astrometry \
    && make -j4 && make \
    && make py \
    && make extra \
    && make install INSTALL_DIR=/usr/local \
    && make clean \
    && (cd && PYTHONPATH=/usr/local/lib/python python3 -c "import astrometry; print(astrometry.__version__)") \
    && git describe && echo 0.90

# The Tractor
RUN git clone http://github.com/dstndstn/tractor.git tractor \
    && cd tractor \
    && git checkout dr10.1 \
    && python3 setup-cython.py --with-ceres install --home /usr/local \
    && make version && cp tractor/version.py /usr/local/lib/python/tractor/ \
    && rm -R $(find . -name "*.o" -o -name "*.so") \
    && (cd && PYTHONPATH=/usr/local/lib/python python3 -c "import tractor; print(tractor.__version__)")

RUN /sbin/ldconfig

RUN rm /root/.profile

RUN pip --no-cache-dir install Pillow

# QDO
RUN git clone https://bitbucket.org/berkeleylab/qdo.git qdo \
    && cd qdo \
    && python3 setup.py install

# python = python3
RUN ln -s "$(which python3)" /usr/bin/python

# unwise_psf
RUN git clone https://github.com/legacysurvey/unwise_psf.git \
    && cd unwise_psf && git describe && echo dr10.0 #git checkout dr9.3

# Legacypipe
RUN git clone http://github.com/legacysurvey/legacypipe.git legacypipe \
    && cd legacypipe && git describe

ENV PYTHONPATH=/usr/local/lib/python:/src/unwise_psf/py:/src/legacypipe/py \
    PATH=/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN mkdir /homedir && chmod 777 /homedir
ENV HOME /homedir

# set prompt and default shell
SHELL ["/bin/bash", "-c"]
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["/bin/bash"]

RUN apt -y update && apt install -y ipython3 && \
  apt clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN echo "export PS1='[container] \\u@\\h:\\w$ '" >> $HOME/.bashrc \
  # Create config files in $HOME
  && mkdir $HOME/.qdo && echo "[qdo]" > $HOME/.qdo/qdorc \
  && python -c "import astropy" \
  && python -c "import matplotlib.font_manager as fm; f = fm.FontManager()" \
  && ipython3 -c "print('hello')" \
  # Download astropy site locations and USNO sky model
  && python -c "from astropy.coordinates import EarthLocation; EarthLocation._get_site_registry(force_download=True)" \
  && python -c "from astropy.coordinates import EarthLocation, SkyCoord, AltAz; from astropy.time import Time; print(EarthLocation.of_site('ctio')); print(SkyCoord(180.,-45.,unit='deg').transform_to(AltAz(obstime=Time(56806.0, format='mjd'), location=EarthLocation.of_site('ctio'))))" \
  # Download astropy IERS leap-second list
  && python -c "from astropy.time import Time; Time.now()" \
  # Make astropy cache files readable!?!!
  && chmod -R a+rwX $HOME/.astropy \
  # Make ipython config files readable!?!!
  && chmod -R a+rwX $HOME/.ipython

# Curses upon you, astropy
RUN wget -O /usr/lib/python3/dist-packages/astropy/utils/iers/data/Leap_Second.dat \
    https://raw.githubusercontent.com/astropy/astropy/master/astropy/utils/iers/data/Leap_Second.dat \
    && rm $HOME/.wget-hsts

# NOTE, for Shifter (where the whole container filesystem is read-only)
# you probably want to add something like this to your scripts to set up
# config and cache directories:
#  # Config directory nonsense
#  export TMPCACHE=$(mktemp -d)
#  mkdir $TMPCACHE/cache
#  mkdir $TMPCACHE/config
#  # astropy
#  export XDG_CACHE_HOME=$TMPCACHE/cache
#  export XDG_CONFIG_HOME=$TMPCACHE/config
#  mkdir $XDG_CACHE_HOME/astropy
#  cp -r $HOME/.astropy/cache $XDG_CACHE_HOME/astropy
#  mkdir $XDG_CONFIG_HOME/astropy
#  cp -r $HOME/.astropy/config $XDG_CONFIG_HOME/astropy
#  # matplotlib
#  export MPLCONFIGDIR=$TMPCACHE/matplotlib
#  mkdir $MPLCONFIGDIR
#  cp -r $HOME/.config/matplotlib $MPLCONFIGDIR
#  # ipython
#  export IPYTHONDIR=$TMPCACHE/ipython
#  mkdir $IPYTHONDIR
#  cp -r $HOME/.ipython $IPYTHONDIR  

RUN python -O -m compileall \
    -x lib2to3\|debugpy \
    /usr/local/lib/python \
    /usr/lib/python3.10 \
    /usr/lib/python3/dist-packages \
    /src/unwise_psf/py \
    /src/legacypipe/py/{legacypipe,legacyzpts} && \
    python    -m compileall \
    -x lib2to3\|debugpy \
    /usr/local/lib/python \
    /usr/lib/python3.10 \
    /usr/lib/python3/dist-packages \
    /src/unwise_psf/py \
    /src/legacypipe/py/{legacypipe,legacyzpts}

# Bug in pip 22.0.2 prevents uninstallation from working!
RUN pip3 --no-cache-dir install --upgrade pip

# QDO
RUN cd qdo && git remote add dstn https://bitbucket.org/dstn/qdo.git \
        && git fetch dstn && git checkout handle-oom \
	&& pip install .

ENV LEGACYPIPE_DIR /src/legacypipe/py

# update legacypipe
RUN cd /src/legacypipe && git pull \   
    && git describe && echo DR10.0.0 && \
  python    -m compileall /src/legacypipe/py/{legacypipe,legacyzpts} && \
  python -O -m compileall /src/legacypipe/py/{legacypipe,legacyzpts}

