#! /bin/bash

# Merge the zeropoint files to make the survey-ccds and annotated-ccds files.

source dr8-env.sh

outdir=$LEGACY_SURVEY_DIR
zptsdir=$LEGACY_SURVEY_DIR/zpts

for prefix in annotated survey; do
    # DECam
    find $zptsdir/decam/*/ -name "*-"$prefix".fits" | grep -v debug > $outdir/$prefix-ccds-dr8b-decam.txt

    python $LEGACYZPTS_DIR/py/legacyzpts/legacy_zeropoints_merge.py \
        --file_list $outdir/$prefix-ccds-dr8b-decam.txt \
        --outname $outdir/$prefix-ccds-dr8b-decam-nocuts.fits
    
    python $LEGACYPIPE_DIR/py/legacypipe/create_kdtrees.py --no-cut \
        $outdir/$prefix-ccds-dr8b-decam-nocuts.fits \
        $outdir/$prefix-ccds-dr8b-decam-nocuts.kd.fits

    /usr/bin/rm -f $outdir/$prefix-ccds-dr8b-decam.txt
    
    # 90prime and mosaic
    find $zptsdir/90prime/*/ -name "*-"$prefix".fits" | grep -v debug > $outdir/$prefix-ccds-dr8b-90prime.txt
    find $zptsdir/mosaic/*/ -name "*-"$prefix".fits" | grep -v debug > $outdir/$prefix-ccds-dr8b-mosaic.txt
    cat $outdir/$prefix-ccds-dr8b-90prime.txt $outdir/$prefix-ccds-dr8b-mosaic.txt > $outdir/$prefix-ccds-dr8b-90prime-mosaic.txt
    
    python $LEGACYZPTS_DIR/py/legacyzpts/legacy_zeropoints_merge.py \
        --file_list $outdir/$prefix-ccds-dr8b-90prime-mosaic.txt \
        --outname $outdir/$prefix-ccds-dr8b-90prime-mosaic-nocuts.fits
    
    python $LEGACYPIPE_DIR/py/legacypipe/create_kdtrees.py --no-cut \
        $outdir/$prefix-ccds-dr8b-90prime-mosaic-nocuts.fits \
        $outdir/$prefix-ccds-dr8b-90prime-mosaic-nocuts.kd.fits
    
    /usr/bin/rm -f $outdir/$prefix-ccds-dr8b-90prime.txt $outdir/$prefix-ccds-dr8b-mosaic.txt $outdir/$prefix-ccds-dr8b-90prime-mosaic.txt 

done
 
