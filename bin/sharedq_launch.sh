#!/bin/bash 

#export bricklist=$SCRATCH/test/legacypipe/py/coadd_dirs.txt
#export bricklist=/global/cscratch1/sd/kaylanb/test/legacypipe/py/dr4_bricks.txt
export bricklist=/global/cscratch1/sd/kaylanb/test/legacypipe/py/tractor_fns.txt

echo bricklist=$bricklist
if [ ! -e "$bricklist" ]; then
    echo file=$bricklist does not exist, quitting
    exit 999
fi

#
export dr4c_dir=/global/cscratch1/sd/desiproc/dr4/data_release/dr4c
export walltime=03:00:00
export usecores=1
export threads=1
let hypercores=${usecores}*2
export hypercores=${hypercores}
# Loop over a star and end
step=200
beg=10001
max=70000
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
    # Write multiple sruns to 1 job script
    while read aline; do
        brick=`echo $aline|awk '{print $1}'`
        cmd="srun -n 1 -c $usecores python legacypipe/format_headers.py --brick $brick --dr4c_dir ${dr4c_dir}"
        echo $cmd >> $fn 
    done <<< "$(sed -n ${start_brick},${end_brick}p $bricklist)"
    # Modify N cores, nodes, walltime, etc
    sed -i "s#SBATCH\ -n\ 2#SBATCH\ -n\ ${hypercores}#g" $fn
    sed -i "s#SBATCH\ -t\ 00:10:00#SBATCH\ -t\ ${walltime}#g" $fn
    sed -i "s#usecores=1#usecores=${usecores}#g" $fn 
    # Submit script created above
    sbatch $fn 
    let cnt=${cnt}+1
    #--export bricklist,start_brick,end_brick
done
echo submitted $cnt bricks
