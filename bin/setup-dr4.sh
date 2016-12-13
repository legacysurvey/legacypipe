#! /bin/bash

# Copy input files required for running the Tractor on Edison to $SCRATCH.


export basedir=$SCRATCH/cp-images
mkdir -p $basedir
# rsync -a is equivalent to rsync -rlptgoD, want that without -t since $SCRATCH files older than 12 weeks are purged
export RSYNC_ARGS="-rlpgoD --size-only"

# Bootes
# Make file that lists all ooi images then do
# for fn in `cat bootes-90prime-abspath.txt`;do fns=`find $(echo $fn|sed s/ooi/oo\[idw\]/g)`;rsync -Riv -rlpgoD --size-only $fns /scratch1/scratchdirs/desiproc/DRs/cp-images/bootes-bokTPV/;done
# OR do get everyting
# for fn in `find /project/projectdirs/cosmo/staging/bok/BOK_CP/*/ksb_*_oo[idw]_*_v1.fits.fz`;do rsync -Riv -rlpgoD --size-only $fn /scratch1/scratchdirs/desiproc/DRs/cp-images/bootes/;done 

# MZLS v2 -- 2.8T
#rsync -Riv $RSYNC_ARGS /project/projectdirs/cosmo/staging/mosaicz/MZLS_CP/*v2/k4m_*_oo[iwd]_zd_v2.fits.fz ${basedir}/
# MZLS v3
#rsync -Riv $RSYNC_ARGS /project/projectdirs/cosmo/staging/mosaicz/MZLS_CP/*v3/k4m_*_oo[iwd]_zd_v3.fits.fz ${basedir}/
# MZLS v1
#rsync -Riv $RSYNC_ARGS /project/projectdirs/cosmo/staging/mosaicz/MZLS_CP/*[0-9][0-9]/k4m_*_oo[idw]_zd_v1.fits.fz ${basedir}/

# CP BASS -- 38G
#rsync -Riv $RSYNC_ARGS /project/projectdirs/cosmo/staging/bok/BOK_CP/CP20160703V0/ksb_*_oo[idw]_[gr]_v0.fits.fz ${basedir}/

# NAOC BASS -- 6.3T
#rsync -Riv $RSYNC_ARGS /global/projecta/projectdirs/cosmo/staging/bok/reduced/*/*.fits ${basedir}/


# CALIB
# MZLS v1
export newdir=${basedir}/calib/mzls_v1
mkdir -p $newdir
#rsync -v $RSYNC_ARGS /global/cscratch1/sd/desiproc/dr3-mzls/calib/* ${newdir}/


# Code
# mkdir -p $SCRATCH/code;
# for x in $(ls COPY-TO-SCRATCH); do
#   echo $x;
#   #cp -r COPY-TO-SCRATCH/$x $SCRATCH/code/;
#   rsync -arv COPY-TO-SCRATCH/$x $SCRATCH/code/;
#   ln -s $SCRATCH/code/$x .;
# done

# Calibration products
#mkdir -p $SCRATCH/calib/decam
#rsync -v $RSYNC_ARGS ~/cosmo/work/decam/calib/{astrom-pv,psfex,sextractor,sky} $SCRATCH/calib/decam

# SDSS photoObj slice
#./legacypipe/copy-sdss-slice.py

# unWISE images
#mkdir -p $SCRATCH/unwise-coadds
#UNW=/project/projectdirs/cosmo/data/unwise/unwise-coadds/
#cp $UNW/allsky-atlas.fits $SCRATCH/unwise-coadds
#rsync -Rv $RSYNC_ARGS $UNW/./*/*/*-w{3,4}-{img-m.fits,invvar-m.fits.gz,n-m.fits.gz,n-u.fits.gz} $SCRATCH/unwise-coadds
