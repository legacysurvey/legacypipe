FROM legacysurvey/legacypipe:DR10.0.0

# /global/common/software/nersc/pm-2022.05.0/extra_modulefiles/fast-mkl-amd/fast-mkl-amd.lua
# /global/common/software/nersc/pm-2021q4/sw/fast-mkl-amd/libfakeintel.so

COPY libfakeintel.so /usr/local/lib
ENV LD_PRELOAD=/usr/local/lib/libfakeintel.so

# docker build -t legacysurvey/legacypipe:perlmutter-hack .
