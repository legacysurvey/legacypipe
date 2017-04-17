#!/bin/bash

# Input: list of bricks
# Output: $fn, all files associated with those bricks

bricklist="$1"
if [ ! -e "$bricklist" ]; then
    exit 999
fi

fn=files_to_remove.txt
if [ -e "$fn" ]; then
    rm $fn
fi

#outdir=/global/cscratch1/sd/desiproc/dr4/data_release/dr4_fixes
outdir=/scratch1/scratchdirs/desiproc/DRs/data-releases/dr4
for brick in `cat $bricklist`;do
    bri=`echo $brick | head -c 3`
    echo $outdir/checkpoints/$brick.pickle >> $fn
    echo $outdir/checkpoints/$bri/$brick.pickle >> $fn
    echo $outdir/coadd/$bri/$brick/* >> $fn
    echo $outdir/metrics/$bri/*${brick}* >> $fn
    echo $outdir/tractor/$bri/*${brick}* >> $fn
    echo $outdir/tractor-i/$bri/*${brick}* >> $fn
done
# Replace whitespaces with newlines (one file per line)
sed -i -e 's/\s\+/\n/g' $fn
# "*" will be printed if files don't exist, remove these
junk=junk.txt
grep "*" -v $fn > $junk
mv $junk $fn
echo Wrote $fn
#
nbricks=`wc $bricklist|awk '{print $1}'`
files_per_brick=18
let expect=$nbricks*$files_per_brick
actual=`wc $fn|awk '{print $1}'`
echo Given $nbricks bricks, Expect $files_per_brick files per brick
echo So $expect files to remove 
echo Found $actual files in $fn
echo OK if above numbers are close b/c not all bricks have full grz
