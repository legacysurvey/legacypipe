#!/bin/bash 

# DR4 fixes runs from
# $LEGACY_SURVEY_DIR/../dr4_fixes/legacypipe-dir which is
#     /scratch1/scratchdirs/desiproc/DRs/dr4-bootes/dr4_fixes/legacypipe-dir
# $CODE_DIR/../dr4_fixes/legacypipe which is 
#     /scratch1/scratchdirs/desiproc/DRs/code/dr4_fixes/legacypipe

export LEGACY_SURVEY_DIR=/global/cscratch1/sd/desiproc/dr4/legacypipe-dir/../dr4_fixes/legacypipe-dir
#/global/cscratch1/sd/desiproc/dr4/master_wdr4fixes/legacypipe-dir
export UNWISE_COADDS_DIR=/global/cscratch1/sd/desiproc/dr4/unwise-coadds/fulldepth:/global/cscratch1/sd/desiproc/dr4/unwise-coadds/w3w4
export UNWISE_COADDS_TIMERESOLVED_DIR=/global/cscratch1/sd/desiproc/dr4/unwise-coadds/time_resolved_neo2
export UNWISE_COADDS_TIMERESOLVED_INDEX=/global/cscratch1/sd/desiproc/dr4/unwise-coadds/time_resolved_neo2/time_resolved_neo2-atlas.fits
export CODE_DIR=/global/cscratch1/sd/desiproc/code
export outdir=/global/cscratch1/sd/desiproc/dr4/data_release/dr4_fixes
#export DUST_DIR=adfa
#export unwise_dir=lakdjf

export overwrite_tractor=no
export full_stacktrace=no
export early_coadds=no
export just_calibs=no
export force_all=no

#bricklist=${LEGACY_SURVEY_DIR}/bricks_bootes_W3_deep2_BOSS_5017.txt
#bricklist=${LEGACY_SURVEY_DIR}/bricks_dr4b_oom.txt
#bricklist=${LEGACY_SURVEY_DIR}/bricks_dr4b_psf.txt
#bricklist=${LEGACY_SURVEY_DIR}/bricks_dr4b_all.txt
#bricklist=${LEGACY_SURVEY_DIR}/bricks_dr4b_olapdr3_grz1.txt
#bricklist=${LEGACY_SURVEY_DIR}/bricks_dr4b_need_calibs.txt
#bricklist=${LEGACY_SURVEY_DIR}/bricks_dr4b_rerun.txt
#bricklist=${LEGACY_SURVEY_DIR}/bricks_tocheckmaster.txt
#bricklist=${LEGACY_SURVEY_DIR}/bricks_dr4b_badastrom.txt
#bricklist=${LEGACY_SURVEY_DIR}/bricks_dr4b_badastromStill.txt
#bricklist=${LEGACY_SURVEY_DIR}/bricks_dr4b_remain.txt
#bricklist=${LEGACY_SURVEY_DIR}/remain_mem_oot.txt
#bricklist=${LEGACY_SURVEY_DIR}/remain_other_wshotbad.txt 
#bricklist=${LEGACY_SURVEY_DIR}/remain_other_oldckpt.txt 
#bricklist=${LEGACY_SURVEY_DIR}/remain_other_skynotinhdr.txt
#bricklist=${LEGACY_SURVEY_DIR}/remain_other_astromgitpull.txt
#bricklist=${LEGACY_SURVEY_DIR}/remain_resubmit.txt
#bricklist=${LEGACY_SURVEY_DIR}/remain_resubmit2.txt
#bricklist=${LEGACY_SURVEY_DIR}/remain_other_hugefwhm.txt
#bricklist=${LEGACY_SURVEY_DIR}/remain_resubmit3.txt
#bricklist=${LEGACY_SURVEY_DIR}/bricks_dr4b_moretime.txt
#bricklist=${LEGACY_SURVEY_DIR}/bricks_dr4b_remain.txt
#bricklist=${LEGACY_SURVEY_DIR}/bricks_dr4b_gitpullastrom.txt
#bricklist=${LEGACY_SURVEY_DIR}/bricks_dr4b_oldckpt.txt
bricklist=${LEGACY_SURVEY_DIR}/bricks_dr4b_other.txt

#bricklist=${LEGACY_SURVEY_DIR}/bricks-dr4-${NERSC_HOST}.txt
#bricklist=${LEGACY_SURVEY_DIR}/bricks-dr4-notdone-${NERSC_HOST}.txt
#bricklist=${LEGACY_SURVEY_DIR}/bricks-dr4-${NERSC_HOST}-oom.txt
#bricklist=${LEGACY_SURVEY_DIR}/bricks-dr4-rerunpsferr-${NERSC_HOST}.txt
#bricklist=${LEGACY_SURVEY_DIR}/bricks-dr4-${NERSC_HOST}-asserterr.txt
#bricklist=${LEGACY_SURVEY_DIR}/bricks-dr4-${NERSC_HOST}-hascutccds.txt
echo bricklist=$bricklist
if [ ! -e "$bricklist" ]; then
    echo file=$bricklist does not exist, quitting
    exit 999
fi

export statdir="${outdir}/progress"
mkdir -p $statdir 

# Loop over bricks
start_brick=1
end_brick=300
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
    sbatch ../bin/job_dr4.sh --export brick,outdir,overwrite_tractor,full_stacktrace,early_coadds,just_calibs,force_all
    touch $stat_file
    let cnt=${cnt}+1
done <<< "$(sed -n ${start_brick},${end_brick}p $bricklist)"
echo submitted $cnt bricks
