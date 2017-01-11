#!/bin/bash

# This code finds files like: /scratch2/scratchdirs/kaylanb/obiwan/eboss_ngc_good/input_sample/bybrick/eboss_ngcsample_1346p152.fits
# For each one, it removes: /scratch2/scratchdirs/kaylanb/obiwan/eboss_ngc_good/input_sample/bybrick/eboss_ngcsample_1346p152_*.fits
echo finding samples
don=done_samples.txt
#find /scratch2/scratchdirs/kaylanb/obiwan/eboss_ngc_good/input_sample/bybrick/eboss_ngcsample_*[pm][0-9][0-9][0-9].fits > $donecho removing samples
wc -l $don
for fn in `cat $don`;do 
    if [ -e $fn ];then
        echo fn exists: $fn 
        # Combined brick file exists, remove the 240 pieces
        name=`echo $fn|sed s/.fits//g`
        newname=${name}_200.fits
        if [ -e $newname ]; then
            echo removing pieces for $fn  
            # 240 pieces haven't been removed yet
            #fils=`ls ${name}_*.fits |egrep ${name}_[0-2][0-9]\{0,2\}.fits`
            #echo fils=$fils
            #rm $fils
            rm ${name}_*.fits 
            #for num in `seq 0 239`;do
            #    newname=${name}_${num}.fits
            #    echo removing $newname
            #    rm $newname
            #done
        fi
    fi
done

