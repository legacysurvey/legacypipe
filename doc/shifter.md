# Notes on using Shifter at NERSC


## Docker setup for legacypipe:

`docker-intel/Dockerfile` contains a build script that uses the Intel Numpy and Scipy libraries.

```
docker build docker-intel/ -t dstndstn/legacypipe:intel
docker push dstndstn/legacypipe:intel
```

DockerHub page: <https://cloud.docker.com/repository/docker/dstndstn/legacypipe/general>

## Shifter:

```
shifterimg pull docker:dstndstn/legacypipe:intel
```

## Launching jobs:

To run `qdo` inside the container:

```
QDO_BATCH_PROFILE=cori-shifter qdo launch -v tst 1 --cores_per_worker 8 --walltime=30:00 --batchqueue=debug --keep_env --batchopts "--image=docker:dstndstn/legacypipe:intel" --script "/src/legacypipe/bin/runbrick-shifter.sh"
```

So you tell qdo you're using cori-shifter, and you tell slurm which shifter image to use, and the script specified here is within the container's filesystem.

The resulting turducken of processes looks like this:
```
slurm -> qdo-in-shifter -> sh-in-shifter -> bash-in-shifter -> python3-in-shifter
```
