#!/bin/bash

bricklist=dr4_bricks.txt
dr=/global/cscratch1/sd/desiproc/dr4/data_release/dr4_fixes
# Write problems to a file
output_fn=wrong_number_files.txt
if [ -e "$output_fn" ];then rm $output_fn;fi
# Correct number of files
n_metric=2
n_chkpt=1
n_tractor=2
n_tractor_i=1
n_coadd_g=12
n_coadd_gr=19
n_coadd_grz=26
# Loop
#set -x
n_bricks=`wc -l $bricklist |awk '{print $1}'`
cnt=0
for brick in `cat $bricklist`;do
    let cnt=$cnt+1
    ith=$(($cnt % 100))
    if [ "$ith" == "0" ];then echo $cnt/$n_bricks;fi
    bri=`echo $brick|head -c 3`
    metric_dr=${dr}/metrics/$bri
    chkpt_dr=${dr}/checkpoints
    tractor_dr=${dr}/tractor/$bri
    tractor_i_dr="${dr}/tractor-i/$bri"
    coadd_dr=${dr}/coadd/$bri/$brick

    cnt_metric=`ls ${metric_dr}/*${brick}*|wc -l`
    cnt_chkpt=`ls ${chkpt_dr}/*${brick}*|wc -l`
    cnt_tractor=`ls ${tractor_dr}/*${brick}*|wc -l`
    cnt_tractor_i=`ls ${tractor_i_dr}/*${brick}*|wc -l`
    cnt_coadd=`ls ${coadd_dr}/*${brick}*|wc -l`

    cnt_bands=`ls ${coadd_dr}/legacysurvey-${brick}-image-*.fits|wc -l`

    if [ "$cnt_metric" != "$n_metric" ]; then echo $brick metric $cnt_metric $n_metric >> $output_fn;fi
    if [ "$cnt_chkpt" != "$n_chkpt" ]; then echo $brick chkpt $cnt_chkpt $n_chkpt >> $output_fn;fi
    if [ "$cnt_tractor" != "$n_tractor" ]; then echo $brick tractor $cnt_tractor $n_tractor >> $output_fn;fi
    if [ "$cnt_tractor_i" != "$n_tractor_i" ]; then echo $brick tractor_i $cnt_tractor_i $n_tractor_i >> $output_fn;fi
    # Coadds depend on number bands
    if [ "$cnt_bands" == "1" ]; then
        n_coadd=${n_coadd_g} 
    elif [ "$cnt_bands" == "2" ]; then 
        n_coadd=${n_coadd_gr} 
    elif [ "$cnt_bands" == "3" ]; then 
        n_coadd=${n_coadd_grz}
    fi 
    if [ "$cnt_coadd" != "$n_coadd" ]; then echo $brick coadd $cnt_coadd $n_coadd >> $output_fn;fi
    #break
done
