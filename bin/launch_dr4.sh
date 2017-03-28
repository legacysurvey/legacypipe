#!/bin/bash 

# DR4 fixes runs from
# $LEGACY_SURVEY_DIR/../dr4_fixes/legacypipe-dir which is
#     /scratch1/scratchdirs/desiproc/DRs/dr4-bootes/dr4_fixes/legacypipe-dir
# $CODE_DIR/../dr4_fixes/legacypipe which is 
#     /scratch1/scratchdirs/desiproc/DRs/code/dr4_fixes/legacypipe

export LEGACY_SURVEY_DIR=/scratch1/scratchdirs/desiproc/DRs/dr4-bootes/dr4_fixes/legacypipe-dir
export UNWISE_COADDS_DIR=/scratch1/scratchdirs/desiproc/unwise-coadds/fulldepth:/scratch1/scratchdirs/desiproc/unwise-coadds/w3w4
export UNWISE_COADDS_TIMERESOLVED_DIR=/scratch1/scratchdirs/desiproc/unwise-coadds/time_resolved_neo2
export UNWISE_COADDS_TIMERESOLVED_INDEX=/scratch1/scratchdirs/desiproc/unwise-coadds/time_resolved_neo2/time_resolved_neo2-atlas.fits

export CODE_DIR=/scratch1/scratchdirs/desiproc/DRs/code/dr4_fixes/legacypipe
export outdir=/scratch1/scratchdirs/desiproc/DRs/data-releases/dr4_fixes
#export DUST_DIR=adfa
#export unwise_dir=lakdjf

export overwrite_tractor=no
export full_stacktrace=no
export early_coadds=no
export just_calibs=no
export bad_astrom=yes

#bricklist=${LEGACY_SURVEY_DIR}/bricks_bootes_W3_deep2_BOSS_5017.txt
#bricklist=${LEGACY_SURVEY_DIR}/bricks_dr4b_oom.txt
#bricklist=${LEGACY_SURVEY_DIR}/bricks_dr4b_psf.txt
#bricklist=${LEGACY_SURVEY_DIR}/bricks_dr4b_all.txt
#bricklist=${LEGACY_SURVEY_DIR}/bricks_dr4b_olapdr3_grz1.txt
#bricklist=${LEGACY_SURVEY_DIR}/bricks_dr4b_need_calibs.txt
bricklist=${LEGACY_SURVEY_DIR}/bricks_dr4b_rerun.txt


#bricklist=${LEGACY_SURVEY_DIR}/bricks-dr4-${NERSC_HOST}.txt
#bricklist=${LEGACY_SURVEY_DIR}/bricks-dr4-notdone-${NERSC_HOST}.txt
#bricklist=${LEGACY_SURVEY_DIR}/bricks-dr4-${NERSC_HOST}-oom.txt
#bricklist=${LEGACY_SURVEY_DIR}/bricks-dr4-rerunpsferr-${NERSC_HOST}.txt
#bricklist=${LEGACY_SURVEY_DIR}/bricks-dr4-${NERSC_HOST}-asserterr.txt
#bricklist=${LEGACY_SURVEY_DIR}/bricks-dr4-${NERSC_HOST}-hascutccds.txt
if [ "$overwrite_tractor" = "yes" ]; then
    #bricklist=${LEGACY_SURVEY_DIR}/bricks-dr4-nowise-${NERSC_HOST}.txt
    #bricklist=${LEGACY_SURVEY_DIR}/bricks-dr4-nowiseflux-${NERSC_HOST}.txt
    #bricklist=${LEGACY_SURVEY_DIR}/bricks-dr4-${NERSC_HOST}-nolc.txt
    bricklist=${LEGACY_SURVEY_DIR}/bricks_bootes_W3_deep2_BOSS_5017.txt
elif [ "$full_stacktrace" = "yes" ]; then
    bricklist=${LEGACY_SURVEY_DIR}/bricks_bootes_W3_deep2_BOSS_5017.txt
elif [ "$bad_astrom" = "yes" ]; then
    bricklist=${LEGACY_SURVEY_DIR}/bricks_dr4b_badastrom.txt
    export outdir=/scratch1/scratchdirs/desiproc/DRs/data-releases/dr4_fixes_badastrom
fi 
echo bricklist=$bricklist
if [ ! -e "$bricklist" ]; then
    echo file=$bricklist does not exist, quitting
    exit 999
fi

export statdir="${outdir}/progress"
mkdir -p $statdir 

# Loop over bricks
start_brick=1
end_brick=9000
cnt=0
while read aline; do
    export brick=`echo $aline|awk '{print $1}'`
    if [ "$full_stacktrace" = "yes" ];then
        stat_file=$statdir/stacktrace_$brick.txt
    else
        stat_file=$statdir/submitted_$brick.txt
    fi
    bri=$(echo $brick | head -c 3)
    tractor_fits=$outdir/tractor/$bri/tractor-$brick.fits
    if [ -e "$tractor_fits" ]; then
        if [ "$overwrite_tractor" = "yes" ]; then
            echo ignoring existing tractor.fits
        else
            continue
        fi
    fi
    if [ -e "$stat_file" ]; then
        continue
    fi
    if [ -e "${outdir}/tractor/${bri}/tractor-${brick}.fits" ]; then
        continue
    fi
    sbatch ../bin/job_dr4.sh --export brick,outdir,overwrite_tractor,full_stacktrace,early_coadds,just_calibs
    touch $stat_file
    let cnt=${cnt}+1
done <<< "$(sed -n ${start_brick},${end_brick}p $bricklist)"
echo submitted $cnt bricks
