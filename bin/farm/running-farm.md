#### Preparation

1.  Create a directory named farm, or any other name you want

```bash
mkdir farm
cd farm
```

2.  Clone the legacypipe repo and checkout the correct version

```bash
git clone https://github.com/legacysurvey/legacypipe
cd legacypipe
git fetch && git fetch --tags
git checkout <tag>
```

3.  Copy the relevant scripts from the legacypipe repo

```bash
# Copy all sh files from <legacypipe directory> to <the farm directory you created at step 1>
cd ..
cp bin/farm/<data release>/*.sh .
```

4.  Create qdo_login.sh containing qdo database credential environmental variables.
5.  Run generate-launcher.py

```bash
module load python3
python generate-launcher.py
```

6.  Modify the options within launch-farm.sh

```bash
###Dependencies
# Paths to mpi_bugfix.sh and qdo_login.sh
###

###OPTIONS
FARM_SCRIPT # Path to farm.py
FARM_OUTDIR # Path to farm's output files, should be formatted as
						# /your/path/checkpoint-%(brick)s.pickle
FARM_INDIR  # Path to farm's input files, should be formatted as
						# /your/path/%(brick).3s/runbrick-%(brick)s-srcs.pickle
QNAME				# qdo queue containing the bricks you want to process
###
```

7.  Modify the options within launch-worker.sh

```bash
###Dependencies
# Path to mpi_bugfix.sh
###

###Options
WORKER_SCRIPT # Path to worker.py
###
```

#### Launch farm and worker combo

This is probably how you should run farm.py in production.

```bash
sbatch launch-combo.sh
```

To change any of the settings, run

```bash
python generate-launcher.py
```

#### Launch farm and workers separately

This might be useful while debugging using interactive nodes. Details coming soon.