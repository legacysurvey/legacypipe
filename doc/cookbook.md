------------------------------------------------------------------------

How to run a Legacy Survey data release

Martin Landriau

July 2017 - January 2018

------------------------------------------------------------------------

Introduction
============

This report documents the procedure to run a data release from the
community pipeline (CP) processed images to the final data products at
NERSC. Instructions assume code is being run as desiproc.

Environment
===========

The environment is set in the desiproc .bashrc.ext file.

Dust maps, WISE coadds, etc
---------------------------

    module use /global/cscratch1/sd/desiproc/modulefiles
    module load legacypipe/legacysurvey
    module load legacypipe/unwise_coadds
    module load legacypipe/unwise_coadds_timeresolved
    module load legacypipe/dust

The above module files need to be modified to contain the correct
information for the current DR. Unless the location of relevant
quantities has changed, the only thing to update in these module files
between DRs is `legacysurvey` to indicate the location of the DR
directory.

desiconda
---------

Desiconda provides a consistent build for all the dependencies required
for the Legacy Survey processing. To ensure we use the most up-to-date
version, we use explicitely the version number instead of the default.

    module use /global/common/${NERSC_HOST}/contrib/desi/desiconda/20170818-1.1.12-img/modulefiles
    module load desiconda

### Alternate builds

It sometimes has been necessary to have a newer build of some desiconda
packages due to some bug(s). Here I detail the installation procedure
for the three most common: `fitsio`, `astrometry.net` and `tractor`, on
Edison scratch. Note that these codes need to be built on both Edison
and Cori, because of compiled components.

    git clone https://github.com/esheldon/fitsio.git
    cd fitsio/
    python setup.py install --prefix=/scratch1/scratchdirs/desiproc/DRcode/build/
    export PYTHONPATH=$SCRATCH/DRcode/build/lib/python3.5/site-packages:$PYTHONPATH
    cd ..
    git clone https://github.com/dstndstn/astrometry.net.git
    cd astrometry.net/
    make
    make py
    make extra
    make install INSTALL_DIR=/scratch1/scratchdirs/desiproc/DRcode/build/
    export PYTHONPATH=$SCRATCH/DRcode/build/lib/python:$PYTHONPATH
    export PATH=$SCRATCH/DRcode/build/bin:$PATH
    cd ..
    git clone https://github.com/dstndstn/tractor.git
    cd tractor/
    make
    python setup.py install --prefix=/scratch1/scratchdirs/desiproc/DRcode/build/
    cd ..

In the above, the export commands need to be included in any script
passed to qdo, described in the next section.

qdo
---

In order to facilitate the launching of jobs, we use qdo.

    export PATH=${PATH}:/global/cscratch1/sd/desiproc/code/qdo/bin
    export PYTHONPATH=${PYTHONPATH}:/global/cscratch1/sd/desiproc/code/qdo/
    export QDO_BACKEND=postgres
    export QDO_DB_NAME=desirun
    export QDO_DB_HOST=nerscdb03.nersc.gov
    export QDO_DB_USER=desirun_admin
    export QDO_DB_PASS= password redacted

The easiest way to set up a queue is to pass to qdo a list of task as an
ASCII file:

    qdo load queuename tasklist.txt

The queue doesn't need to exist prior to loading the task list. To
launch a job:

    qdo launch queuename 48 --cores_per_worker 8 --walltime=36:00:00 \
                            --script script.sh \
                            --batchqueue regular --keep_env --batchopts "-C haswell"

Most of the arguments are self-explanatory, but a few notes. The number
of cores per node (24 for Edison and 32 for Cori); in the example above
the, the job is meant to run on Cori because of the last option; the
number of nodes requested will be 48 / (32/8) = 12. The fifth argument
is a script that contain several variables to be passed to the code
actually doing the work. The seventh argument is required for everything
to work, no one is entirely sure why; the last one is required for
running on Cori (it should not be there if running on Edison). To
monitor the progress and manage a queue, the following commands are
usefull

    qdo list
    qdo recover queuename
    qdo retry queuename

which list all the active queues and the number of tasks pending,
running, completed and failed; re-queue all tasks that are listed as
running but are in fact inactive (must NOT be run when some tasks are
actually running); and re-queue failed tasks, after a bug-fix.

Legacy Survey codes
-------------------

Most of the code to run a DR are in two repository, which can be
checked-out from GitHub.

    cd /global/cscratch1/sd/desiproc/DRcode
    git clone https://github.com/legacysurvey/legacyzpts.git
    git clone https://github.com/legacysurvey/legacypipe.git
    export PYTHONPATH=/global/cscratch1/sd/desiproc/DRcode/legacypipe/py/:$PYTHONPATH
    export PYTHONPATH=/global/cscratch1/sd/desiproc/DRcode/legacyzpts/py/:$PYTHONPATH

Pre-tractor computations
========================

Up to DR5, we have been using aperture zeropoints. For DR6, we have used
instead PSF normalised zeropoints, computed at the same time as the
calibration files.

NOTE: I haven't run most of this myself. --ML

Aperture zeropoints
-------------------

As of DR6, these are no longer used for tractor processing, although
they were still computed.

The first step in generating the zero-points is to make a list of images
for each camera. This can be done

    find /project/projectdirs/cosmo/staging/mosaicz/MZLS_CP/CP*/k4m*ooi*.fits.fz
         > mosaic_images.txt
    find /project/projectdirs/cosmo/staging/bok/BOK_CP/CP*/ksb*ooi*.fits.fz
         > 90prime_images.txt
    find /project/projectdirs/cosmo/staging/decam/*/c4d*ooi*.fits.fz
         > decam_images

The image list needs to be trimmed for dates. NOTE: Kaylan wrote an
ipython notebook, showing how he did this for DR6, but it's broken --ML.
The image lists can now be converted into a qdo task list:

    for camera in 90prime mosaic decam; do
        python legacyzpts/py/legacyzpts/runmanager/qdo_tasks.py \
               --camera \
               --imagelist ${camera}_images.txt
        qdo load zpts_dr6 ${camera}_tasks.txt
    done

To launch, use a local copy of the script
`zpts_code/legacyzpts/bin/qdo_job.sh` and edit the `name_for_run`
environment variable (this should be the directory from which jobs are
launched) and then:

    cd rundirectory
    qdo launch zpts_dr6 320 --cores_per_worker 1 --batchqueue debug --walltime 00:30:00 \
                            --script fullpathtorundirectory/qdo_job.sh --keep_env

Once all the zeropoints have been computed, the tables must be merged:

    find rundirectory/decam -name "*-zpt.fits" > done_zpt.txt
    python legacyzpts/py/legacyzpts/legacy_zeropoints_merge.py --file_list done_zpt.txt \
          --nproc 1 --outname merged_zpt.fits

The final merging of the thousands of tables could take a while, so do
is better to do as a batch job, e.g. with an appropriately edited
version of legacyzpts `/bin/slurm_job_merge.sh`.

NOTE: the QA python script described in the documentation no longer
exists. --ML

For DR5, we imposed a depth cut in order to process some bricks with a
very high number of exposure. The following procedure was used:

    python legacyanalysis/depth-cut.py
    python legacyanalysis/check-depth-cut.py
    python legacyanalysis/dr5-cut-ccds.py

The first step is run for each brick and generates
`depthcut/*/ccds-*.fits` tables of CCDs that pass the depth cut for that
brick. The second reads the `per-brick ccds-*` tables and cut to the
union of all CCDs that pass depth cut in some brick and generates the
`depth-cut-kept-ccds.fits` file. The third step reads
`depth-cut-kept-ccds.fits` and cut the (already-created) annotated-ccds
table and create the `.kd.fits` version of the CCDs table. We have found
that, for DR6, a depth cut was not necessary.

Calibration files
-----------------

Calibration files can be computed on the fly, but it is more efficient
to compute them in advance. Furthermore, if we want to use PSF
zeropoints, then we must compute the calibrations in advance because
they are needed to compute the PSF zeropoints.

The cut on images to be used was performed by the code
`legacyzpts/py/legacyzpts/compare-psf-ap.py`

The procedure is to run the two following scripts:

    python legacypipe/run-calib.py
    python legacypipe/merge-calibs.py

To create the kd-tree version of the zeropoint files,

    cp survey-ccds-90prime-g.fits /tmp/dr6.fits
    tabmerge survey-ccds-90prime-r.fits+1 /tmp/dr6.fits+1
    tabmerge survey-ccds-mosaic-z.fits+1 /tmp/dr6.fits+1
    startree -i /tmp/dr6.fits -o /tmp/dr6.kd -P -k -n ccds
    fitsgetext -i /tmp/dr6.kd -o /tmp/survey-ccds-dr6.kd.fits -e 0 -e 6 -e \
           1 -e 2 -e 3 -e 4 -e 5

Running tractor
===============

Staging of input data
---------------------

For DR1-4, input images were copied from project to Cori or Edison
scratch; for DR5, this was not done and things worked just fine.

Setting up the task list
------------------------

We need to create a database with a list of tasks. For our purpose, a
task is simply the brick name. The code that generates the brick list
from the CCDs files outputs some diagnostic information which must be
edited out before being fed to qdo:

    python legacypipe/queue-calibs.py --touching --region mzls > bricks.txt >> log
    vi bricks.txt # Edit out everything that isn't a brick name, e.g. 1234p567
    qdo load dr6 bricks.txt

Launching tractor
-----------------

    qdo launch dr6 48 --cores_per_worker 8 --walltime=36:00:00 \
                      --script pb.sh \
                      --batchqueue regular --keep_env --batchopts "-C haswell"

The script in the example is a local copy of
`legacypipe/bin/pipebrick-checkpoint.sh`. Some lines of the script need
to be modified for everything to run:

    export PYTHONPATH=$CSCRATCH/DRcode/legacypipe/py:$PYTHONPATH
    outdir=$CSCRATCH/dr6-out
    rundir=$CSCRATCH/DRcode

The arguments to runbrick can be modified in this script. Here is a
generic call:

    python ${rundir}/legacypipe/py/legacypipe/runbrick.py \ 
          --psf-normalize \
          --skip \ 
          --threads 16 \ 
          --skip-calibs \ 
          --checkpoint ${rundir}/checkpoints/${bri}/checkpoint-${brick}.pickle \ 
          --pickle "${rundir}/pickles/${bri}/runbrick-\%(brick)s-\%\%(stage)s.pickle" \ 
          --brick $brick --outdir $outdir --nsigma 6 \  
          >> $log 2>&1 

The flag `psf-normalize` is necessary to use PSF zeropoints, otherwise
aperture zeropoints are used. The number of threads has to be smaller or
equal to twice the number of cores per task; skip-calibs assumes all the
calibration files for this brick have already been generated; skip will
stop if there is already a tractor catalogue for this brick. The
checkpoint and pickle flags are optional and the files generated don't
get erased once the brick has completed and the pickle files can be
quite large. An easy way to clean up is via the following script:

    import qdo
    from utils import removeckpt

    rundir = "/global/cscratch1/sd/desiproc/DR5_code/"

    q = qdo.connect('dr5') 
    a = q.tasks(state=qdo.Task.SUCCEEDED) 
    n = len(a) 
    for i in range(n):
        brick = a[i].task 
        removeckpt(brick, rundir)

Note that the checkpoint and pickle file should be in a common directory
regardless of what machine the job is being run on: that way, a job that
is requeued and re-launched on another machine can use existing
checkpoint files.

Monitoring progress
-------------------

Aside from the commands listed in the qdo section, I have written a
number of small codes (including the small checkpoint clearing code in
the previous section) to check the status jobs, requeue jobs with
specific failure modes, etc. These are located in the following
directories and their names are (mostly) self-explanatory:

    /global/cscratch1/sd/desiproc/DRcode/runmanaging/
    /scratch1/scratchdirs/desiproc/DRcode/runmanaging/

Transfering output
------------------

The `coadd`, `tractor` and `metrics` directories are rsynced to
`/projecta`. The log files, will be tarred by directory and gzipped
before being transferred (also, log files for bricks that were processed
partly on both machines need to be concatenated).

In the same directory, the contents of the `$LEGACY_SURVEY_DIR` (except
the images directory which contains only symbolic links) are also
copied.

Post-tractor processing
=======================

Top-level depth files
---------------------

The first step to compute the top-level depth information.

    python legacypipe/py/legacyanalysis/depth-histogram.py

Prior to running, the hard-coded location of the release directory (the
place where the coadd directory can be found) must be modified. NOTE: I
may be missing some things here --ML.

Sweeps and external matching
----------------------------

The next step is to generate the sweep files, which contain a subset of
the columns of the tractor catalogues and matching to external
catalogues. For DR1-4, these steps were run by John Moustakas and for
DR5 by John and Adam Myers.

We must first set the location of the input Tractor catalogs and the
location of the bricks file (which speeds up the process of scanning
through files) and the name of the environment variable which will hold
the list of Tractor catalogs, as well as the desired output directories:
e.g.:

    export TRACTOR_INDIR=/global/project/projectdirs/cosmo/work/legacysurvey/dr5/DR5_out/tractor
    export BRICKSFILE=/global/cscratch1/sd/desiproc/dr5/survey-bricks.fits.gz
    export TRACTOR_FILELIST=$CSCRATCH/tractor_filelist
    export dr=dr5
    export SWEEP_OUTDIR=$CSCRATCH/$dr/sweep
    /usr/bin/mkdir -p $SWEEP_OUTDIR}

Then, we build the list of tractor files for this data release and
submit the job using a slurm script to generate the sweeps files:

    find $TRACTOR_INDIR -name 'tractor-*.fits' > $TRACTOR_FILELIST
    sbatch $LEGACYPIPE_DIR/bin/generate-sweep-files.slurm

The files will be written to `$SWEEP_OUTDIR`.

To generate the \"external files\" matched to other surveys (e.g. see
​http://legacysurvey.org/dr4/files/), we proceed in a similar fashion:

    export EXTERNAL_OUTDIR=$CSCRATCH/$dr/external
    /usr/bin/mkdir -p $EXTERNAL_OUTDIR
    sbatch $LEGACYPIPE_DIR/bin/match-external-catalog.slurm

The files will be written to `$EXTERNAL_OUTDIR`.

Other "after-burner" stages
---------------------------

-   Load images into the Legacy Survey viewer (Dustin Lang). Note that
    this is usually done as things progress.

-   Generate a picture gallery (John Moustakas & Ben Weaver).

-   Final vetting and moving into place (Ben Weaver). NOTE: Dustin is
    putting together some tools to do the vetting, so that it can be
    done right after completion.

-   Documentation (Adam Myers with input from the team).

-   Transfer to NOAO archive (NOAO staff).

Worked example
==============

In the directory `/global/cscratch1/sd/desiproc/dr6-example`, the file
`commands.txt` contains instructions on how to run a subset (15 bricks)
of DR6; the content of this file is reproduced below. This example
covers the tractor processing phase. It assumes that the required
contents is in place in `$LEGACY_SURVEY_DIR`. The setup shell script
contains the necessary export commands from sections "Alternate builds"
and "Legacysurvey codes".

    # Logon to Cori
    ssh me@cori.nersc.gov
    ssh corigrid
    sup desiproc
    # Go to example directory
    cd $CSCRATCH/dr6-example
    # Get environment for DR6
    source setup-dr6.sh
    # Calibs, zeropoints...  For now, assume they are done.
    # Create brick list for small region (1 sq deg).
    python -u $CSCRATCH/DRcode/legacypipe/py/legacypipe/queue-calibs.py \
          --touching --minra 180 --maxra 181 --mindec 32 --maxdec 33 > bricks.dat
    # Edit the file "bricks.dat" to remove all but the brick names.
    vi bricks.dat
    # Load bricks into new qdo queue.
    qdo load dr6-example bricks.dat
    # Check that the queue dr6-example has been created and that 15 tasks are pending.
    qdo list
    # Launch processing. This assumes processing on Cori, two bricks per node,
    # no hyperthreading (to limit the risk of the job going down due to memory usage).
    # This will use one node; alternatively, launch several shorter jobs.
    # This uses a job script identical to that used for DR6, except for
    # two points: outdir points to the example directory and the
    # checkpoints are written to outdir instead of rundir.
    qdo launch dr6-example 2 --cores_per_worker 16 --walltime=48:00:00 \
         --script $CSCRATCH/dr6-example/pbcp-example.sh --batchqueue regular \
         --keep_env --batchopts "-L CSCRATCH,project" --batchopts "-C haswell"
    # Check queue progress
    qdo list
    # Check your jobs are running, or how long it will be before they run
    sqs | grep dr6-example
    # Each brick will have files output in logs, coadd, metrics, tractor
    # and tractor-i, all in a subdirectory with the first 3 numbers of the
    # brick name; the slurm log file will appear in the directory where
    # the job was launched, so in principle this one.
    # If jobs get killed while a task is running, it will be necessary to requeue.
    qdo recover dr6-example
    # Edit the script to reduce the number of threads by half
    vi pbcp-example.sh
    # Re-launch
    qdo launch dr6-example 2 --cores_per_worker 16 --walltime=48:00:00 \
         --script $CSCRATCH/dr6-example/pbcp-example.sh --batchqueue regular \
         --keep_env --batchopts "-L CSCRATCH,project" --batchopts "-C haswell"
    # Once everything has completed, delete the queue
    qdo delete dr6-example --force

Acknowledgements
================

This document builds upon a wiki page written by Kaylan Burleigh; the
zeropoints section follows some of the documentation for the legacyzpts
code; and the sweeps and externals subsection follows the wiki page
written by Adam Myers. It has benefited from input from many people,
most notably Kaylan Burleigh, Dustin Lang, Adam Myers and Stephen
Bailey.
