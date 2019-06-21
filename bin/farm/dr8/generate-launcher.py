import sys
if sys.version_info[0] < 3:
    print('Please use Python 3')
    exit(1)

def get_param(msg, default):
    s = input(msg + '(Default: ' + str(default) + ') ')
    if s:
        return s
    return default

nodes = get_param('How many worker nodes do you need?', 16)
minutes = get_param('How many minutes do you want the job to run?', 60)
qos = get_param('QOS?', 'debug' if int(minutes) <= 30 else 'regular')
image_tag = get_param('Image tag?', 'nersc-dr8.3.1')
working_dir = get_param('Where is your working directory? (i.e. where launch-worker.sh and launch-farm.sh are located)', '/global/cscratch1/sd/ziyaoz/farm')
head_arch = get_param('Head node architecture?', 'haswell')
assert head_arch == 'knl' or head_arch == 'haswell', 'Architecture not found'
head_core_count = 272 if head_arch == 'knl' else 64

script = '#!/bin/bash\n\
#SBATCH --qos={2}\n\
#SBATCH --time={1}:00\n\
#SBATCH --nodes=1\n\
#SBATCH --tasks-per-node=1\n\
#SBATCH --cpus-per-task={6}\n\
#SBATCH --constraint={5}\n\
#SBATCH --image=docker:legacysurvey/legacypipe:{3}\n\
#\n\
#SBATCH packjob\n\
#SBATCH --qos={2}\n\
#SBATCH --time={1}:00\n\
#SBATCH --nodes={0}\n\
#SBATCH --tasks-per-node=1\n\
#SBATCH --cpus-per-task=272\n\
#SBATCH --constraint=knl\n\
#SBATCH --image=docker:legacysurvey/legacypipe:{3}\n\
\n\
cd {4}\n\
srun --pack-group=0 shifter ./launch-farm.sh &\n\
CURR_HOST=$(hostname)\n\
srun --pack-group=1 -N {0} -n {0} --cpus-per-task 272 shifter ./launch-worker.sh tcp://${{CURR_HOST}}:5555 &\n\
wait\n'

filename = 'launch-combo.sh'
f = open(filename, 'w')
f.write(script.format(nodes, minutes, qos, image_tag, working_dir, head_arch, head_core_count))
print('Script generated as ' + filename)
f.close()
