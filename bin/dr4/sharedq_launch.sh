#!/bin/bash 

# Python stack, env vars, etc
setEnv() {
    local fn=$1
    echo "#Environment" >> $fn
    echo "source ~/.bashrc_desiconda" >> $fn
    echo "export LEGACY_SURVEY_DIR=/global/cscratch1/sd/kaylanb/dr5_bricks/dr5-new" >> $fn
}

getLogDir() {
    local outdir=$1
    local dmy=`date|awk '{print $3$2$NF}'`
    echo ${outdir}/logs/${NERSC_HOST}_${dmy}
}

getLogName() {
    local logdir=$1
    local brick=$2
    local hm=`date|awk '{print $4}'|awk -F ":" '{print $1$2}'`
    echo "${logdir}/log.${brick}_${hm}"
}

# configures MPI/OpenMP for cori haswell
setOMP() {
    local fn=$1
    local pythreads=$2
    # number of OpenMP threads per MPI task
    echo "#Configure MPI/OpenMP for cori haswell" >> $fn
    echo "export OMP_PLACES=cores" >> $fn
    echo "export OMP_NUM_THREADS=$pythreads" >> $fn
    # stays 1 no matter what
    # https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
    echo "export MKL_NUM_THREADS=1" >> $fn
    # check mpi/openmp settings
    echo "export KMP_AFFINITY=verbose" >> $fn
}

# 0 for true, 1 for false
isShared() {
    local part=$1
    if [ ${part} == "shared" ];then
        return 0
    else
        return 1
    fi
}

isCori() {
    if [ "$NERSC_HOST" = "cori" ]; then
        return 0
    else
        return 1
    fi
}

# Limit memory to avoid 1 srun killing whole node
# ...Shared queue only
limMemory() {
    local fn=$1
    local part=$2
    local usecores=$3
    if isShared $part; then 
        if isCori;then
            # 128 GB / node = 1345000000 kbytes
            local maxmem=134000000
            let usemem=${maxmem}*${usecores}/32
        else
            # 62 GB / node = 65000000 kbytes
            local maxmem=65000000
            let usemem=${maxmem}*${usecores}/24
        fi
        echo "#Limit memory for Shared Queue jobs" >> $fn
        echo "ulimit -S -v $usemem" >> $fn
        echo "ulimit -a" >> $fn
    fi
}


getSrun() {
    local part=$1
    local brick=$2
    local outdir=$3 
    local hypercores=$4 
    local pythreads=$5 
    local logfn=$6
    local cmd="python legacypipe/runbrick.py --brick ${brick} --outdir ${outdir} --threads ${pythreads} --run dr5 --no-write --skip --skip-calibs >> ${logfn} 2>&1"
    if isShared ${part};then
        echo "srun -n 1 -c ${hypercores} ${cmd}"
    else
        echo "srun -n 1 -c ${hypercores} --cpu_bind=cores ${cmd} &"
    fi
}

getJobsPerNode() {
    local part=$1
    local usecores=$2
    if isShared ${part};then
        echo 1
    else
        let a=32/${usecores}
        echo $a
    fi
}

export bricklist=/global/cscratch1/sd/kaylanb/dr5_bricks/legacypipe/py/dr5_bricks25.txt

echo bricklist=$bricklist
if [ ! -e "$bricklist" ]; then
    echo file=$bricklist does not exist, quitting
    exit 999
fi

#export part=shared
export part=debug
export outdir=/global/cscratch1/sd/kaylanb/dr5_bricks/runs
export logdir=$(getLogDir ${outdir})
mkdir -p $logdir
export walltime=00:30:00
export jobname=dr5
#export need_project=no
#export need_projecta=yes
# MPI taks must be divisible by 64!!
# if isn't set --cpu_bind=cores in srun
# and -c to (32/4)*2 in srun
export usecores=4
export pythreads=4
let hypercores=${usecores}*2
export hypercores=${hypercores}
# Loop over a star and end
step=$(getJobsPerNode $part $usecores)
beg=1
max=`wc -l ${bricklist} |awk '{print $1}'`
max=1
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
    # KEY
    setEnv $fn
    setOMP $fn $pythreads
    limMemory $fn $part $usecores
    # 2 Options
    # 1) Multiple sruns, 1 per line in bricklist
    while read aline; do
        brick=`echo $aline|awk '{print $1}'`
        echo "#Run" >> $fn
        logfn=$(getLogName ${logdir} ${brick})
        echo "echo Logging to: $logfn" >> $fn
        cmd=$(getSrun $part $brick $outdir $hypercores $pythreads $logfn)
        echo $cmd >> $fn
    #    #export bri=`echo $brick|head -c 3`
    #    #echo "mkdir -p ${proj_dir}/logs2/${bri}" >> $fn
    #    #echo "export logfn=\`find ${proj_dir}/logs/ -name \"log.${brick}*\"|xargs ls -l|tail -n 1|awk '{print \$NF}'\`" >> $fn
    done <<< "$(sed -n ${start_brick},${end_brick}p $bricklist)"
    if ! isShared $part; then
        echo "wait" >> $fn
    fi
    # 2) One srun, using args start and end
    #cmd="srun -n 1 -c $usecores python ../bin/job_accounting.py --dowhat blobs --fn ${bricklist} --line_start ${start_brick} --line_end ${end_brick}"
    #echo $cmd >> $fn 
    # Modify N cores, nodes, walltime, etc
    if isShared $part;then
        sed -i "s#SBATCH\ -n\ 2#SBATCH\ -n\ ${hypercores}#g" $fn
    else
        sed -i "s#SBATCH\ -p\ shared#SBATCH\ -p\ ${part}#g" $fn
        sed -i "s#SBATCH\ -n\ 2#SBATCH\ -N\ 1#g" $fn
    fi
    sed -i "s#SBATCH\ -t\ 00:10:00#SBATCH\ -t\ ${walltime}#g" $fn
    sed -i "s#SBATCH\ -J\ repack#SBATCH\ -J\ ${jobname}#g" $fn
    sed -i "s#usecores=1#usecores=${usecores}#g" $fn 
    if [ "$NERSC_HOST" = "edison" ]; then
        sed -i "s?SBATCH\ -C\ haswell?#SBATCH\ -C\ haswell?g" $fn
    fi
    #if [ ${need_project} == "yes" ];then
    #    sed -i "s#SBATCH\ -L\ SCRATCH#SBATCH\ -L\ SCRATCH,project#g" $fn
    #elif [ ${need_projecta} == "yes" ];then
    #    sed -i "s#SBATCH\ -L\ SCRATCH#SBATCH\ -L\ SCRATCH,projecta#g" $fn
    #fi
    # Submit script created above
    echo Created batch script: $fn 
    sbatch $fn 
    let cnt=${cnt}+1
    #--export bricklist,start_brick,end_brick
done
echo submitted $cnt bricks
