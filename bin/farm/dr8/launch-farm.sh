#!/bin/bash

###Dependencies
source /global/cscratch1/sd/ziyaoz/farm/mpi_bugfix.sh
source /global/cscratch1/sd/ziyaoz/farm/qdo_login.sh
###

###OPTIONS
FARM_SCRIPT='/global/cscratch1/sd/ziyaoz/farm/legacypipe.bac/py/legacypipe/farm.py'
FARM_OUTDIR='/global/cscratch1/sd/ziyaoz/farm-checkpoints-prod/checkpoint-%(brick)s.pickle'
FARM_INDIR='/global/cscratch1/sd/landriau/dr8/decam/pickles/%(brick).3s/runbrick-%(brick)s-srcs.pickle'
#QNAME=ziyao-big-blobs
#QNAME=ziyao-dr8-south-1000
QNAME=ziyao-dr8-pt3
###

###Run farm
cd /src/legacypipe/py
python -u ${FARM_SCRIPT} --pickle ${FARM_INDIR} --big drop --inthreads 4 --checkpoint $FARM_OUTDIR $QNAME
###

###Remove Queue
#qdo delete $QNAME --force
###
