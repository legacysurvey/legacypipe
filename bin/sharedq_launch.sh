#!/bin/bash 

#export bricklist=$SCRATCH/test/legacypipe/py/coadd_dirs.txt
#export bricklist=/global/cscratch1/sd/kaylanb/test/legacypipe/py/dr4_bricks.txt
#export bricklist=/global/cscratch1/sd/kaylanb/test/legacypipe/py/tractor_fns.txt
#export bricklist=/global/cscratch1/sd/kaylanb/test/legacypipe/py/tractor_i_fns.txt
#export bricklist=/global/cscratch1/sd/kaylanb/test/legacypipe/py/tractors_proja_dr4c.txt
#export bricklist=/global/cscratch1/sd/kaylanb/test/legacypipe/py/dr3_bricks.txt
#export bricklist=/global/projecta/projectdirs/cosmo/work/dr4/dr4_bricks.txt
#export bricklist=/global/cscratch1/sd/kaylanb/test/legacypipe/py/fns_for_25bricks.txt
#export bricklist=/global/cscratch1/sd/kaylanb/test/legacypipe/py/fns_for_25bricks_rerun.txt
#export bricklist=/global/cscratch1/sd/kaylanb/test/legacypipe/py/tractor_fns.txt
export bricklist=/global/cscratch1/sd/kaylanb/test/legacypipe/py/tractor_i_fns.txt


echo bricklist=$bricklist
if [ ! -e "$bricklist" ]; then
    echo file=$bricklist does not exist, quitting
    exit 999
fi

#
export need_project=yes
export need_projecta=no

export proj_dir=/global/projecta/projectdirs/cosmo/work/dr4
export dr4c_dir=/global/cscratch1/sd/desiproc/dr4/data_release/dr4c
export run=dr3
#export run=legacy
export outdir=/global/cscratch1/sd/kaylanb/dr5_zpts
export walltime=00:30:00
export jobname=blobs
export usecores=1
export threads=1
let hypercores=${usecores}*2
export hypercores=${hypercores}
# Loop over a star and end
step=2000
beg=1
max=`wc -l ${bricklist} |awk '{print $1}'`
#max=10
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
    #while read aline; do
        #brick=`echo $aline|awk '{print $1}'`
        ####
        #cmd="srun -n 1 -c $usecores python legacypipe/format_headers.py --brick $brick --dr4c_dir ${dr4c_dir}"
        #echo $cmd >> $fn 
        #echo "source /global/homes/k/kaylanb/.bashrc_desiconda" >> $fn 
        ####
        #echo "srun -n 1 -c $usecores python legacypipe/runbrick.py --run ${run} --brick ${brick} --hybrid-psf --outdir ${outdir} --nsigma 6 --no-write --threads ${threads} --skip-calibs" >> $fn
        #### 
        #export bri=`echo $brick|head -c 3`
        #echo "mkdir -p ${proj_dir}/logs2/${bri}" >> $fn
        #echo "export logfn=\`find ${proj_dir}/logs/ -name \"log.${brick}*\"|xargs ls -l|tail -n 1|awk '{print \$NF}'\`" >> $fn
        #echo "echo $brick \${logfn}" >> $fn
        #echo "cp \${logfn} ${proj_dir}/logs2/${bri}/" >> $fn
        ###
        #cmd="srun -n 1 -c $usecores python legacyccds/legacy_zeropoints.py --camera decam --image ${brick} --outdir ${outdir}"
        #echo $cmd >> $fn 
    #done <<< "$(sed -n ${start_brick},${end_brick}p $bricklist)"
    # 2) One srun, using args start and end
    cmd="srun -n 1 -c $usecores python ../bin/job_accounting.py --dowhat blobs --fn ${bricklist} --line_start ${start_brick} --line_end ${end_brick}"
    echo $cmd >> $fn 
    # Modify N cores, nodes, walltime, etc
    sed -i "s#SBATCH\ -n\ 2#SBATCH\ -n\ ${hypercores}#g" $fn
    sed -i "s#SBATCH\ -t\ 00:10:00#SBATCH\ -t\ ${walltime}#g" $fn
    sed -i "s#SBATCH\ -J\ repack#SBATCH\ -J\ ${jobname}#g" $fn
    sed -i "s#usecores=1#usecores=${usecores}#g" $fn 
    sed -i "s#threads=1#threads=${threads}#g" $fn 
    if [ "$NERSC_HOST" = "edison" ]; then
        sed -i "s?SBATCH\ -C\ haswell?#SBATCH\ -C\ haswell?g" $fn
    fi
    if [ ${need_project} == "yes" ];then
        sed -i "s#SBATCH\ -L\ SCRATCH#SBATCH\ -L\ SCRATCH,project#g" $fn
    elif [ ${need_projecta} == "yes" ];then
        sed -i "s#SBATCH\ -L\ SCRATCH#SBATCH\ -L\ SCRATCH,projecta#g" $fn
    fi
    # Submit script created above
    echo Created batch script: $fn 
    sbatch $fn 
    let cnt=${cnt}+1
    #--export bricklist,start_brick,end_brick
done
echo submitted $cnt bricks
