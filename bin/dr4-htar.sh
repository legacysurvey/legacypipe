#!/bin/bash

# Htars 1000 brick files to hpss archive
export backup_dir=htar_backups
# Find all completed bricks
if [ "$NERSC_HOST" == "edison" ]; then
    outdir=/scratch1/scratchdirs/desiproc/DRs/data-releases/dr4
else
    outdir=/global/cscratch1/sd/desiproc/dr4/data_release/dr4
fi
# Use finished brick list from job_accounting
don=dr4_bricks_done.tmp
# Backing up all bricks existing on Today's date
year=`date|awk '{print $NF}'`
today=`date|awk '{print $3}'`
month=`date +"%F"|awk -F "-" '{print $2}'`

#ontape=bricks_ontape.txt
#cat $backup_dir/fortape_[0-9][0-9][0-9][0-9][mp][0-9][0-9][0-9].txt > $ontape
# Remove from list all bricks already on tape
#echo New bricks
#don_new=don_new.txt
#rm $don_new
#python diff_list.py --completed $don --ontape $ontape --outfn $don_new
#don_new=don.txt

# Every 1000 bricks to new file
nbricks=`wc -l $don |awk '{print $1}'`
let chunks=$nbricks/1000
echo Every 1000 bricks
junk=fortape.txt
rm $junk
# loop over chunks each of 1000
echo Splitting $nbricks finished bricks into $chunks files of 1000 each
for i in `seq 1 $chunks`;do 
    let j=$i-1
    let en=1000*$j+1000
    let st=1000*$j+1
    echo rows $st,$en of $don
    sed -n ${st},${en}p $don > $junk
    # Give a unique name
    unique=`head -n 1 $junk`
    fortape=fortape_${year}_${month}_${today}_$unique.txt
    if [ ! -e "$fortape" ];then
        mv $junk $fortape
        # Replace whitespaces with newlines (one file per line)
        sed -i -e 's/\s\+/\n/g' $fortape
    fi
done

# List all files that exists for each completed brick
echo Looping over files having 1000 bricks, populating with all files to backup
for fn in `ls fortape_${year}_${month}_${today}_*[mp][0-9][0-9][0-9].txt`;do
    backup=`echo $fn|sed s/.txt/_allfiles.txt/g`
    echo Writing $backup
    if [ ! -e "$backup" ]; then
        for brick in `cat $fn`;do
            bri="$(echo $brick | head -c 3)"
            echo $outdir/checkpoints/$bri/$brick.pickle >> $backup
            echo $outdir/coadd/$bri/$brick >> $backup
            echo $outdir/logs/$bri/$brick >> $backup
            echo $outdir/metrics/$bri/*${brick}* >> $backup
            echo $outdir/tractor/$bri/*${brick}* >> $backup
            echo $outdir/tractor-i/$bri/*${brick}* >> $backup
        done
        # Replace whitespaces with newlines (one file per line)
        sed -i -e 's/\s\+/\n/g' $backup
    fi
done

# Write htar commands to file
# need to run them from command line NOT script 
cmds=htar_cmds_${year}_${month}_${today}.txt
rm $cmds
# e.g. for cmd in `cat $cmds`;do $cmd;done 
echo Htar-ing everything listed in fortape_${year}_${month}_${today}...txt
for fn in `ls fortape_${year}_${month}_${today}_*allfiles.txt`;do
    nam=`echo $fn | sed s/.txt/.tar/g`
    sout=`echo $fn | sed s/.txt/.out/g`
    # If sout exists but htar not successful, rm sout and re-htar
    if [ -e "$sout" ];then
        good=`tail ${sout}|grep "HTAR SUCCESSFUL"|wc -c`
        if [ "$good" -eq 0 ]; then
            # Htar didn't work, rm sout and run htar again
            rm $sout
            echo Rerunning htar for $sout
        fi
    fi
    # Just write core htar command to file, anything else confuses unix!
    if [ ! -e "$sout" ];then
        echo "${nam} -L ${fn} > ${sout}" >> $cmds
    fi
done
echo Htar commands written to: $cmds
echo htar can interact with login nodes at most 8 times simultaneously
echo Submit 7 htars at one time, leaving extra to use hsi
echo when those are successful submit next 7, etc
echo See dr4-qdo-htar.sh for running htar
# sed -n 1,7p ..., sed -n 8,14p..., sed -n 15,21p
#sed -n 1,7p htar_cmds_2017_01_30.txt| while read line;do a=`echo $line|awk -F ">" '{print $1}'`;b=`echo $line|awk -F ">" '{print $2}'`;echo $b;nohup htar -Hcrc -cf $a > $b & done


#echo Execute them from command line with:
#echo echo "for cmd in `cat $cmds`;do $cmd;done"
# HTAR, store checksums
# Confirm HTAR SUCESSFUL then:
#for fn in `find fortape_*_allfiles.txt`;do sout=`echo $fn|sed s/.txt/.out/g`;blist=`echo $fn|sed s/_allfiles//g`;mv $fn $sout $blist htar_backups/;done
