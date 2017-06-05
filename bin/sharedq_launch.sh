#!/bin/bash 

#export bricklist=$SCRATCH/test/legacypipe/py/coadd_dirs.txt
#export bricklist=/global/cscratch1/sd/kaylanb/test/legacypipe/py/dr4_bricks.txt
#export bricklist=/global/cscratch1/sd/desiproc/test/legacypipe/py/dr4c_notcoadd_dirs.txt
#export bricklist=/global/cscratch1/sd/desiproc/test/legacypipe/py/dr4c_coadd_dirs.txt


#export camera=mosaic
export camera=90prime
#export cal=psfex
#export cal=sextractor
export cal=splinesky

export bricklist=/global/cscratch1/sd/desiproc/test/legacypipe/py/calib_${camera}_${cal}.txt
#export bricklist=/global/cscratch1/sd/desiproc/test/legacypipe/py/calib_${camera}_${cal}_timedout.txt

echo bricklist=$bricklist
if [ ! -e "$bricklist" ]; then
    echo file=$bricklist does not exist, quitting
    exit 999
fi

#
export dr4c_dir=/global/cscratch1/sd/desiproc/dr4/data_release/dr4c
export proj_dir=/global/projecta/projectdirs/cosmo/work/dr4c
export walltime=01:00:00
export jobname=tarcalibs
export usecores=1
export threads=1
let hypercores=${usecores}*2
export hypercores=${hypercores}
# Loop over a star and end
step=1
beg=1
max=`wc -l ${bricklist}|awk '{print $1}'`
###
cnt=0
# Make 1 script for each i of loop
for star in `seq ${beg} ${step} ${max}`;do 
    export start_brick=${star}
    let en=${star}+${step}-1
    export end_brick=${en}
    # Make job script with N = step srun commands
    export fn=../bin/sharedq_job_${start_brick}_${end_brick}.sh
    cp ../bin/sharedq_job_template.sh $fn
    # 2 Options
    # 1) Multiple sruns, 1 per line in bricklist
    while read aline; do
        brick=`echo $aline|awk '{print $1}'`
        #cmd="srun -n 1 -c $usecores python legacypipe/format_headers.py --brick $brick --dr4c_dir ${dr4c_dir}"
        #cmd="rsync -av ${dr4c_dir}/$brick ${proj_dir}/"
        #cmd="rsync -av ${dr4c_dir}/coadd/$brick ${proj_dir}/coadd/"
        #echo $cmd >> $fn 
        echo "cd /global/cscratch1/sd/desiproc/dr4/dr4_fixes/legacypipe-dir/calib/${camera}/${cal}" >> $fn
        echo "tar -zcvf legacysurvey_dr4_calib_${camera}_${cal}_${brick}.tar.gz ${brick}" >> $fn
    done <<< "$(sed -n ${start_brick},${end_brick}p $bricklist)"
    # 2) One srun, using args start and end
    #cmd="srun -n 1 -c $usecores python ../bin/job_accounting.py --dowhat dr4c_vs_dr4b --fn ${bricklist} --line_start ${start_brick} --line_end ${end_brick}"
    #echo $cmd >> $fn 
    # Modify N cores, nodes, walltime, etc
    sed -i "s#SBATCH\ -n\ 2#SBATCH\ -n\ ${hypercores}#g" $fn
    sed -i "s#SBATCH\ -t\ 00:10:00#SBATCH\ -t\ ${walltime}#g" $fn
    sed -i "s#SBATCH\ -J\ repack#SBATCH\ -J\ ${jobname}#g" $fn
    sed -i "s#usecores=1#usecores=${usecores}#g" $fn 
    # Submit script created above
    sbatch $fn 
    let cnt=${cnt}+1
    #--export bricklist,start_brick,end_brick
done
echo submitted $cnt bricks
