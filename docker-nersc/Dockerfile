FROM registry.services.nersc.gov/nersc/intel_cxx_fort_mpi_mkl_devel as builder

RUN apt -y update && apt install -y apt-utils && echo yes

RUN DEBIAN_FRONTEND=noninteractive \
    apt install -y --no-install-recommends \
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
    python3 \
    python3-dev \
    python3-pip \
    python3-pil \
    python3-tk \
    # # Remove APT files
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN echo "../site-packages" > /usr/local/lib/python3.5/dist-packages/site-packages.pth

# Pip installs -- binary
RUN for x in \
    setuptools \
    wheel \
    intel-numpy \
    intel-scipy \
    psycopg2-binary \
    matplotlib \
    astropy \
    zmq \
    ; do pip3 --no-cache-dir install $x; done

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

# Pip installs -- built from source
RUN for x in \
    photutils \
    fitsio==0.9.12 \
    ; do pip3 --no-cache-dir install $x; done

ENV PYTHON python3
ENV PYTHON_CONFIG python3-config
ENV PYTHON_SCRIPT "/usr/bin/env python3"

RUN mkdir -p /src
WORKDIR /src

ENV BLAS -L${MKLROOT}/lib/intel64 -lmkl_rt -fopenmp -lpthread -limf -lsvml -ldl
ENV CPPFLAGS -I${MKLROOT}/include

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
    && (cd && PYTHONPATH=/usr/local/lib/python python3 -c "import astrometry; print(astrometry.__version__)")

RUN pip3 --no-cache-dir install cython

# The Tractor
RUN git clone http://github.com/dstndstn/tractor.git tractor \
    && cd tractor \
    && python3 setup-cython.py --with-ceres install --home /usr/local \
    && make version && cp tractor/version.py /usr/local/lib/python/tractor/ \
    && rm -R $(find . -name "*.o" -o -name "*.so") \
    && (cd && PYTHONPATH=/usr/local/lib/python python3 -c "import tractor; print(tractor.__version__)")
#    && make ceres && make install-ceres INSTALL_DIR=/usr/local \

######## Stage 2 ########
FROM registry.services.nersc.gov/nersc/intel_cxx_fort_mpi_mkl_runtime

ENV PYTHON python3
ENV PYTHON_CONFIG python3-config
ENV PYTHON_SCRIPT "/usr/bin/env python3"

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
    python3 \
    python3-dev \
    python3-pip \
    python3-pil \
    python3-tk \
    # # Remove APT files
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY --from=builder /usr/local /usr/local

RUN pip3 --no-cache-dir install cython ipython

WORKDIR /src

# QDO
RUN git clone https://bitbucket.org/berkeleylab/qdo.git qdo \
    && cd qdo \
    && python3 setup.py install

# python = python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# unwise_psf
RUN git clone https://github.com/legacysurvey/unwise_psf.git \
    && cd unwise_psf && git checkout dr9.2

# Legacypipe
RUN git clone http://github.com/legacysurvey/legacypipe.git legacypipe \
    && cd legacypipe && git describe
# && git checkout dr9-dev && echo 6

ENV PYTHONPATH /usr/local/lib/python:/src/unwise_psf/py:/src/legacypipe/py

ENV HOME /root

# set prompt and default shell
SHELL ["/bin/bash", "-c"]

RUN echo "export PS1='[container] \\u@\\h:\\w$ '" >> /root/.bashrc \
  # Create config files in /root
  && mkdir /root/.qdo && echo "[qdo]" > /root/.qdo/qdorc \
  && python -c "import astropy" \
  && python -c "import matplotlib.font_manager as fm; f = fm.FontManager()" \
  && ipython -c "print('hello')" \
  # Download astropy site locations and USNO sky model
  && python -c "from astropy.coordinates import EarthLocation; EarthLocation._get_site_registry(force_download=True)" \
  && python -c "from astropy.coordinates import EarthLocation, SkyCoord, AltAz; from astropy.time import Time; print(EarthLocation.of_site('ctio')); print(SkyCoord(180.,-45.,unit='deg').transform_to(AltAz(obstime=Time(56806.0, format='mjd'), location=EarthLocation.of_site('ctio'))))"

# Patch astropy:
ADD urlmap-readonly.patch /tmp/
RUN cd /usr/local/lib/python3.5/dist-packages && \
    patch -p1 < /tmp/urlmap-readonly.patch

# Alternatively, could retrieve
#   http://data.astropy.org/coordinates/sites.json
# to
#   /usr/local/lib/python3.5/dist-packages/astropy/coordinates/data/sites.json

ENTRYPOINT ["/bin/bash", "-c"]
CMD ["/bin/bash"]

# # update
# RUN git clone http://github.com/dstndstn/tractor.git tractor \
#     && cd tractor \
#     && git checkout cython && echo 3 \
#     && rm /usr/local/lib/python/tractor/dense_optimizer.cpython-35m-x86_64-linux-gnu.so \
#     && cp tractor/dense_optimizer.py /usr/local/lib/python/tractor

RUN python -O -m compileall /usr/lib/python3.5 /usr/lib/python3/dist-packages \
           /usr/local/lib/python /usr/local/lib/python3.5/site-packages \
           /usr/local/lib/python3.5/dist-packages \
           /src/unwise_psf/py \
           /src/legacypipe/py/{legacypipe,legacyzpts} && \
    python    -m compileall /usr/lib/python3.5 /usr/lib/python3/dist-packages \
           /usr/local/lib/python /usr/local/lib/python3.5/site-packages \
           /usr/local/lib/python3.5/dist-packages \
           /src/unwise_psf/py \
           /src/legacypipe/py/{legacypipe,legacyzpts}

# update legacypipe
RUN cd /src/legacypipe && git pull && git describe && echo 1 \
    && python    -m compileall /src/legacypipe/py/{legacypipe,legacyzpts} \
    && python -O -m compileall /src/legacypipe/py/{legacypipe,legacyzpts}
