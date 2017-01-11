#!/bin/bash

# Htars 1000 brick files to hpss archive
# Find all completed bricks
export outdir=/scratch1/scratchdirs/desiproc/DRs/data-releases/dr4
echo Writing finished bricks
don=don.txt
rm $don
for fn in `find ${outdir}/tractor -name "tractor*.fits"`;do echo $(basename $fn)|sed s/tractor-//g |sed s/.fits//g >> $don;done

# Bricks already on tape
echo Bricks already on tape
ontape=bricks_ontape.txt
export backup_dir=htar_backups
cat $backup_dir/fortape_[0-9][0-9][0-9][0-9][mp][0-9][0-9][0-9].txt > $ontape

# Remove from list all bricks already on tape
echo New bricks
don_new=don_new.txt
rm $don_new
python diff_list.py --completed $don --ontape $ontape --outfn $don_new
#don_new=don.txt

# Every 1000 bricks to new file
nbricks=`wc -l $don_new |cut -d ' ' -f 1`
let chunks=$nbricks/1000
echo Every 1000 bricks
junk=fortape.txt
rm $junk
# loop over chunks each of 1000
echo $chunks new chunks of 1000
for i in `seq 1 $chunks`;do 
    let j=$i-1
    let en=1000*$j+1000
    let st=1000*$j+1
    echo rows $st,$en of $don_new
    sed -n ${st},${en}p $don_new > $junk
    unique=`head -n 1 $junk`
    fortape=fortape_$unique.txt
    mv $junk $fortape
    # Replace whitespaces with newlines (one file per line)
    sed -i -e 's/\s\+/\n/g' $fortape
#done
    # List all files that exists for each completed brick
    echo Writing backup
    backup=`echo $fortape|sed s/.txt/_allfiles.txt/g`
    rm $backup
    for brick in `cat $fortape`;do
        bri="$(echo $brick | head -c 3)"
        echo $outdir/checkpoints/$bri/$brick.pickle >> $backup
        echo $outdir/coadd/$bri/$brick >> $backup
        echo $outdir/logs/$bri/$brick >> $backup
        echo $outdir/metrics/$bri/*${brick}* >> $backup
        echo $outdir/tractor/$bri/*${brick}* >> $backup
        echo $outdir/tractor-i/$bri/*${brick}* >> $backup
    done
done

# HTAR, store checksums
#for i in `find fortape_*_allfiles.txt`;do nam=`echo $i | sed s/.txt/.tar/g`;sout=`echo $i | sed s/.txt/.out/g`;echo $nam $i $sout; nohup htar -Hcrc -cf $nam -L $i > $sout & done
# Confirm HTAR SUCESSFUL then:
#for fn in `find fortape_*_allfiles.txt`;do sout=`echo $fn|sed s/.txt/.out/g`;blist=`echo $fn|sed s/_allfiles//g`;mv $fn $sout $blist htar_backups/;done
