FROM registry.services.nersc.gov/nersc/intel_cxx_fort_mpi_mkl_devel as builder

RUN apt -y update && apt install -y apt-utils

RUN DEBIAN_FRONTEND=noninteractive \
    apt install -y --no-install-recommends \
    build-essential \
    make \
    git \
    file \
    pkg-config \
    wget \
    swig \
    netpbm \
    sed \
    grep \
    less \
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
    # Remove APT files
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN echo 'PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh \
    && chmod 755 /etc/profile.d/conda.sh

RUN wget -nv https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    conda config --add channels intel && \
    conda config --set always_yes yes && \
    conda update conda && \
    conda create -n myenv python=3.8 pip numpy scipy && \
    conda clean -a -y

    #conda install python=3.8 pip numpy scipy && \

# LSST
FROM builder as lsst

RUN mkdir lsst_stack
WORKDIR lsst_stack

# Tell bash to run this startup file for all shells (interactive, non-interactive, etc)
# This already seems to be the default.
ENV BASH_ENV=/etc/bash.bashrc

COPY myenv.sh /etc/profile.d
RUN chmod 755 /etc/profile.d/myenv.sh

RUN echo $PATH && which sed

RUN conda config --add channels conda-forge && \
    conda install rubin-env=0.6.0

RUN conda install eups

# ENV EUPS_PATH=/opt/conda/share/eups
# ENV PATH=${PATH}:/opt/conda/eups/bin
# ENV EUPS_SHELL=/bin/bash
# ENV EUPS_PKGROOT=https://eups.lsst.codes/stack/src
# ENV PYTHONPATH=${PYTHONPATH}:/opt/conda/eups/python
# RUN mkdir /opt/lsst
# ENV EUPS_DIR=/opt/lsst
# RUN mkdir $EUPS_DIR/bin && cp /opt/conda/eups/bin/setups.sh $EUPS_DIR/bin
# 
# RUN eups distrib install -t v22_0_1 afw

#ENV LSST_CONDA_ENV_NAME=base

ENV LSST_CONDA_ENV_NAME=myenv
ENV LSST_CONDA_CHANNELS="intel conda-forge"

RUN conda list && \
    curl -OL https://raw.githubusercontent.com/lsst/lsst/tickets/DM-31235/scripts/newinstall.sh && \
    chmod 755 newinstall.sh && \
    bash newinstall.sh -b -T -P /opt/conda

#curl -OL https://raw.githubusercontent.com/lsst/lsst/master/scripts/newinstall.sh && \

#RUN mv /etc/profile.d/myenv.sh /etc/profile.d/10_myenv.sh

# RUN rm /etc/profile.d/myenv.sh /etc/profile.d/conda.sh && \
#     echo "source /root/lsst_stack/loadLSST.bash" > /etc/profile.d/20_lsst.sh && \
#     chmod 755 /etc/profile.d/20_lsst.sh

ENV EUPS_SHELL=bash

RUN rm /etc/profile.d/myenv.sh /etc/profile.d/conda.sh

RUN source /root/lsst_stack/loadLSST.bash && eups distrib install -t v22_0_1 sconsUtils
RUN source /root/lsst_stack/loadLSST.bash && eups distrib install -t v22_0_1 astshim
RUN source /root/lsst_stack/loadLSST.bash && eups distrib install -t v22_0_1 base
RUN source /root/lsst_stack/loadLSST.bash && eups distrib install -t v22_0_1 sphgeom
RUN source /root/lsst_stack/loadLSST.bash && eups distrib install -t v22_0_1 pex_exceptions
RUN source /root/lsst_stack/loadLSST.bash && eups distrib install -t v22_0_1 utils

#RUN source /root/lsst_stack/loadLSST.bash && eups distrib install -t v22_0_1 afw


#ENV LSST_CONDA_CHANNELS=
#COPY newinstall.sh .
#RUN chmod 755 newinstall.sh && \
#   ./newinstall.sh -n -b -t -P /opt/conda
#RUN echo "source $HOME/lsst_stack/loadLSST.bash" > /etc/profile.d/lsst.sh



FROM builder as legacypipe

# ugly: Tractor's mp_fourier module detects the $CC='icc' exactly
# These are in /opt/intel/bin, which is on the PATH.
ENV CC icc
ENV CXX icpc
ENV LDSHARED icc -shared
# Edison & Cori
#ENV CFLAGS -O3 -g -fPIC -std=gnu99 -pthread -x=ivybridge -ax=haswell,knl
# Just Cori
ENV CFLAGS -O3 -g -fPIC -std=gnu99 -pthread -x=haswell -ax=knl
ENV ARCH_FLAGS ""

ENV PYTHON python3
ENV PYTHON_CONFIG python3-config
ENV PYTHON_SCRIPT "/usr/bin/env python3"

RUN mkdir -p /src
WORKDIR /src

ENV BLAS -L${MKLROOT}/lib/intel64 -lmkl_rt -fopenmp -lpthread -limf -lsvml -ldl
ENV CPPFLAGS -I${MKLROOT}/include

RUN wget -nv https://www.mpich.org/static/downloads/3.3/mpich-3.3.tar.gz \
    && tar xvzf mpich-3.3.tar.gz \
    && cd mpich-3.3 \
    && ./configure \
    && make -j4 \
    && make install \
    && make clean \
    && cd .. && rm -Rf mpich-3.3
# defaults to installing in /usr/local

# mpi4py / mpicc can't handle -ax=knl
ENV CFLAGS -O3 -g -fPIC -std=gnu99 -pthread
RUN git clone https://github.com/mpi4py/mpi4py.git \
         && (cd mpi4py \
         && python setup.py build \
         && python setup.py install) \
         && rm -Rf mpi4py
ENV CFLAGS -O3 -g -fPIC -std=gnu99 -pthread -x=haswell -ax=knl

RUN git clone https://github.com/astromatic/sextractor.git sourceextractor \
    && cd sourceextractor \
    && sh autogen.sh \
    && ./configure --enable-icc --enable-mkl \
    && make \
    && make install \
    && make clean

RUN git clone https://github.com/astromatic/psfex.git psfex \
    && cd psfex \
    && sh autogen.sh \
    && ./configure --enable-icc --enable-mkl \
    && make \
    && make install \
    && make clean

# Astrometry.net
RUN git clone http://github.com/dstndstn/astrometry.net.git astrometry \
    && cd astrometry \
    && make -j4 && make \
    && make py \
    && make extra \
    && make install INSTALL_DIR=/usr/local \
    && make clean \
    && (cd && PYTHONPATH=/usr/local/lib/python python3 -c "import astrometry; print(astrometry.__version__)") \
    && git describe && echo 0.85

# Pip installs -- binary.  We need cython for the tractor.
RUN for x in \
    setuptools \
    wheel \
    psycopg2-binary \
    matplotlib \
    astropy \
    zmq \
    cython \
    ; do pip --no-cache-dir install $x; done

# The Tractor
RUN git clone http://github.com/dstndstn/tractor.git tractor \
    && cd tractor \
    && git checkout dr9.6 \
    && python3 setup-cython.py --with-ceres install --home /usr/local \
    && make version && cp tractor/version.py /usr/local/lib/python/tractor/ \
    && rm -R $(find . -name "*.o" -o -name "*.so") \
    && (cd && PYTHONPATH=/usr/local/lib/python python3 -c "import tractor; print(tractor.__version__)")

# Pip installs -- built from source
RUN pip --no-cache-dir install photutils
RUN pip --no-cache-dir install fitsio==1.1.4

RUN /sbin/ldconfig

######## Stage 2 ########
FROM registry.services.nersc.gov/nersc/intel_cxx_fort_mpi_mkl_runtime

ENV PYTHON python3
ENV PYTHON_CONFIG python3-config
ENV PYTHON_SCRIPT "/usr/bin/env python3"
RUN echo 'PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh

RUN rm /root/.profile

RUN apt -y update && apt install -y apt-utils \
 && DEBIAN_FRONTEND=noninteractive \
    apt install -y --no-install-recommends \
    procps \
    htop \
    make \
    git \
    file \
    pkg-config \
    swig \
    gdb \
    strace \
    vim \
    less \
    wget \
    netpbm \
    wcslib-dev \
    wcslib-tools \
    libgsl-dev \
    libcfitsio-dev \
    libcfitsio-bin \
    libgoogle-glog-dev \
    libceres-dev \
    imagemagick \
    # Remove APT files
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY --from=builder /usr/local /usr/local
COPY --from=builder /opt/conda /opt/conda

RUN conda update --no-deps certifi
RUN pip --no-cache-dir install ipython Pillow healpy seaborn ipykernel

WORKDIR /src

# QDO
RUN git clone https://bitbucket.org/berkeleylab/qdo.git qdo \
    && cd qdo \
    && python3 setup.py install

# python = python3
RUN ln -s "$(which python3)" /usr/bin/python

# unwise_psf
RUN git clone https://github.com/legacysurvey/unwise_psf.git \
    && cd unwise_psf && git checkout dr9.3

# Legacypipe
RUN git clone http://github.com/legacysurvey/legacypipe.git legacypipe \
    && cd legacypipe && git describe

ENV PYTHONPATH /usr/local/lib/python:/src/unwise_psf/py:/src/legacypipe/py
ENV PATH /opt/conda/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN mkdir /homedir && chmod 777 /homedir
ENV HOME /homedir

# set prompt and default shell
SHELL ["/bin/bash", "-c"]
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["/bin/bash"]

RUN echo "export PS1='[container] \\u@\\h:\\w$ '" >> $HOME/.bashrc \
  # Create config files in $HOME
  && mkdir $HOME/.qdo && echo "[qdo]" > $HOME/.qdo/qdorc \
  && python -c "import astropy" \
  && python -c "import matplotlib.font_manager as fm; f = fm.FontManager()" \
  && ipython -c "print('hello')" \
  # Download astropy site locations and USNO sky model
  && python -c "from astropy.coordinates import EarthLocation; EarthLocation._get_site_registry(force_download=True)" \
  && python -c "from astropy.coordinates import EarthLocation, SkyCoord, AltAz; from astropy.time import Time; print(EarthLocation.of_site('ctio')); print(SkyCoord(180.,-45.,unit='deg').transform_to(AltAz(obstime=Time(56806.0, format='mjd'), location=EarthLocation.of_site('ctio'))))" \
  # Download astropy IERS leap-second list
  && python -c "from astropy.time import Time; Time.now()" \
  # Make astropy cache files readable!?!!
  && chmod -R a+rwX $HOME/.astropy \
  # Make ipython config files readable!?!!
  && chmod -R a+rwX $HOME/.ipython \
  && echo 1

# Curses upon you, astropy
RUN wget -O /opt/conda/lib/python3.7/site-packages/astropy/utils/iers/data/Leap_Second.dat \
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
    -x lib2to3 \
    /usr/local/lib/python \
    /opt/conda/lib/python3.7 \
    /src/unwise_psf/py \
    /src/legacypipe/py/{legacypipe,legacyzpts} && \
    python    -m compileall \
    -x lib2to3 \
    /usr/local/lib/python \
    /opt/conda/lib/python3.7 \
    /src/unwise_psf/py \
    /src/legacypipe/py/{legacypipe,legacyzpts}

ENV LEGACYPIPE_DIR /src/legacypipe/py

# QDO
RUN cd qdo && git remote add dstn https://bitbucket.org/dstn/qdo.git \
        && git fetch dstn && git checkout handle-oom \
	&& pip install .

# update legacypipe
RUN cd /src/legacypipe && git pull \   
    && git describe && echo DR9.8.2b && \
  python    -m compileall /src/legacypipe/py/{legacypipe,legacyzpts} && \
  python -O -m compileall /src/legacypipe/py/{legacypipe,legacyzpts}
