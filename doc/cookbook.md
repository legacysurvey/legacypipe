------------------------------------------------------------------------

How to run a Legacy Survey data release

Martin Landriau

July 2017 -- July 2018

------------------------------------------------------------------------

Introduction
============

This report documents the procedure to run a data release from the
community pipeline (CP) processed images to the final data products at
NERSC. Instructions assume code is being run as desiproc.

Environment
===========

Using Shifter
---------

Starting from DR8, we shifted to using shifter images for the environment. Shifter images are basically docker images. To use it, you can either directly type

```bash
shifter --image docker:legacysurvey/legacypipe:nersc-dr8.3.1 bash
```

Or you can add `shifter --image docker:legacysurvey/legacypipe:nersc-dr8.3.1 bash` to your srun command. Example:

```bash
srun -n 80 --ntasks-per-node=8 --cpus-per-task=8 --exclusive --cpu_bind=cores --threads-per-core=1 --image=docker:legacysurvey/legacypipe:nersc-dr8.3.0 shifter qdo_do.sh
```

From this point, feel free to skip to the qdo section. Or you can read the now deprecated non-shifter section below.

Not Using Shifter
---------

Up to and including DR6, the environment variables were set by loading
modules in the in the `~/.bashrc.ext` file. From DR7 onwards, the
environment, except for QDO variables, can be set by sourcing\
`legacypipe/bin/legacypipe-env` or a local copy. Note that some relevant
commands may be commented out in this template, which assumes that the
code is being run from within its base directory. These variables
provide the location of the codes, dust maps, WISE coadds, Tycho and
Gaia catalogues, and the data release working directory where the
calibration files, zero-points and links to the images directories can
be found.

desiconda
---------

The above shell script also loads the desiconda module. Desiconda
provides a consistent build for all the dependencies required for the
Legacy Survey processing. To ensure we use the most up-to-date version,
we use explicitely the version number instead of the default.

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

Some of the above export commands (or a template for them) are normally
commented out from the `legacypipe-env` script and this needs to be
modified appropriately.

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

These commands are set in the `~/.bashrc.ext` file; however, if the
`legacypipe-env` script is sourced, we need to rerun the export amending
the `PYTHONPATH`.

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

Zeropoints
----------

The same code is used to generate the aperture and PSF-normalized
zeropoints, depending on the optional flags used. Here, we describe the
PSF zeropoints which have used since DR6. Since these require the
calibration files to be computed, given the appropriate flag, the code
can generate them if it cannot find them.

The first step in generating the zero-points is to make a list of images
for each camera. This can be done

    find /project/projectdirs/cosmo/staging/mosaicz/MZLS_CP/CP*/k4m*ooi*.fits.fz
         > mosaic_images.txt
    find /project/projectdirs/cosmo/staging/bok/BOK_CP/CP*/ksb*ooi*.fits.fz
         > 90prime_images.txt
    find /project/projectdirs/cosmo/staging/decam/*/c4d*ooi*.fits.fz
         > decam_images.txt

The image list needs to be trimmed for dates and possible duplicates
before they can now be converted into a qdo task list. Note that the
image list should have the full path to the image starting with, e.g.
`decam/` with the subdirectories for each camera in the `images`
directory.

    qdo load zpts_dr7 decam_images.txt

To launch, use a local copy of the script
`zpts_code/legacyzpts/bin/qdo_zpts.sh` and edit the `outdir` and
`caldir` environment variable (this should be the directory from which
jobs are launched) and then:

    cd rundirectory
    qdo launch dr7-zpts 32 --cores_per_worker 1 --batchqueue regular \
          --keep_env --walltime 04:00:00
          --script /global/cscratch1/sd/desiproc/dr7_zpts/qdo_zpts_dr7.sh

Note that the files `decam-tiles_obstatus.fits` and
`obstatus/bad_expid.txt` must be in the running directory as these have
hard coded paths. The script should appropriately amend the `PYTHONPATH`
environment variable, as well as the `PATH` if alternate builds to those
in desiconda are to be used. The main command the the script runs is

    python -u $CSCRATCH/DRcode/legacyzpts/py/legacyzpts/legacy_zeropoints.py \
            --camera ${camera} --image ${image_fn} --outdir ${out_dir} \
            --not_on_proj \
            --calibdir ${cal_dir} --splinesky --psf \
            --run-calibs \
            > $log 2>&1

where the `–run-calibs` flag tells the command to run the calibrations
commands if the appropriate files are not found. The code will look in
the `cal_dir` directory for the `psfex`, `psfex-merged`, `splinesky` and
`splinesky-merged` directory and will compute the single-CCD versions
for those that don't exist in either format.

Once all the zeropoints have been computed, the tables must be merged:

    find rundirectory/decam -name "*-legacypipe.fits" > done_zpt.txt
    python legacyzpts/py/legacyzpts/legacy_zeropoints_merge.py --file_list done_zpt.txt \
          --nproc 0 --cut --cut-expnum --outname merged_zpt.fits

The cut flag imposes several hard-coded cuts and the cut-expnum flag
ignores CCDs with no sources, that can cause problems downstream. The
same procedure can be used to merge the other outputs from the code (but
not the calibrations, which are merged differently: see below); e.g. to
create the `ccds-annotated.fits` file, to put in the survey and releases
directories, replace the `“*-legacypipe.fits”` by `“*-annotated.fits”`.

The final merging of the thousands of tables could take a while, so do
is better to do as a batch job, e.g. with an appropriately edited
version of legacyzpts `/bin/slurm_job_merge.sh`. It could also be done
on the interactive queue: budget two hours per merging.

For DR5, we imposed a depth cut in order to process some bricks with a
very high number of exposure. We have found that, for DR6, a depth cut
was not necessary. For DR7, we will use a differerent scheme in order to
keep as many images as possible. Other cuts are made, e.g. only CCDs
whose zeropoints are within a maximum distance from the nominal
zero-points are kept.

Calibration files
-----------------

Calibration files can be computed on the fly, but it is more efficient
to compute them in advance. Furthermore, if we want to use PSF
zeropoints, then we must compute the calibrations in advance because
they are needed to compute the PSF zeropoints. The easiest way is to run
them at the same time as the zeropoints as outlined in the previous
section. In order to run only the calibrations, there are two options:
run the zeropoints code but with the flag

    --run-calibs-only

The other is to call directly the code `legacypipe/run-calib.py`.

The calibration files are output by CCD. To merge them and create files
per exposure, we need to run

    legacypipe/merge-calibs.py --outdir calib --continue

with `LEGACY_SURVEY_DIR` set. The continue flag will make the code
continue to go over every exposure even if some of them fail. If a
survey-ccds files containing the zeropoints already exists, we can limit
the process to the exposure in that CCDs file bu using the flag

     --ccds <survey-ccds-file>.

kd-tree survey-ccds file
------------------------

Legacypipe will look for a unified kd-tree file and will only look for
others if it can't find this one. It must the include the info from all
`survey-ccds-{camera}-{band}.fits.gz`.

To create the kd-tree version of the zeropoint files,

    cp survey-ccds-90prime-g.fits /tmp/dr6.fits
    tabmerge survey-ccds-90prime-r.fits+1 /tmp/dr6.fits+1
    tabmerge survey-ccds-mosaic-z.fits+1 /tmp/dr6.fits+1
    startree -i /tmp/dr6.fits -o /tmp/dr6.kd -P -k -n ccds
    fitsgetext -i /tmp/dr6.kd -o /tmp/survey-ccds-dr6.kd.fits -e 0 -e 6 -e \
           1 -e 2 -e 3 -e 4 -e 5

Obviously, if there is only one `survey-ccds` file, the tabmerge
commands are not necessary.

For DR8, a new script runs the last two commands:

    legacypipe/create_kdtrees.py --infn /tmp/survey-ccds-dr8.fits --outfn survey-ccds-dr8.kd.fits

Generating the brick list
-------------------------

We need to create a database with a list of tasks. For our purpose, a
task is simply the brick name. The code that generates the brick list
from the CCDs files outputs some diagnostic information which must be
edited out before being fed to qdo:

    python legacypipe/queue-calibs.py --touching --region mzls >
    bricks.txt >> log
    vi bricks.txt

and edit out everything that is not a brick name, *e.g.* 1234p567.

Depth cut
---------

To perform the depth cut, the kd-tree version of the merged survey-ccds
file must reside in the (temporary) `LEGACY_SURVEY_DIR` directory.

The first step is to create a task list. For our purpose, a task is
simply the brick name. This list can serve for both for the next stage
of the depth cut process and the later tractor processing. After
generating the brick list as above, feed it to qdo:

    qdo load dr8 bricks.txt

The first step is to take the bricks and compute a corresponding
ccds-BRICK.fits files. Using the QDO queue just created:

    qdo launch dr8 100 --cores_per_worker 1 --walltime=01:00:00 --script legacyanalysis/depthcut.sh --batchqueue regular --keep_env

The command should be run in a temprary directory without any other FITS
files. The above command also assumes that legacypipe/py is in the
PYTHONPATH; the hard coded paths in the script should be adjusted. Once
the queue has been processed, we run in the same directory

    legacyanalysis/depth-cut-dr8.py

in order to take all the ccds-BRICK.fits files to a
survey-ccds-depthcut.fits file.

Afterwards, generate the final kd-tree survey-ccds file using the
`create_kdtrees.py` as described above which must be copied to the final
`LEGACY_SURVEY_DIR` directory.

Running tractor
===============

Staging of input data
---------------------

For DR1-4, input images were copied from project to Cori or Edison
scratch; for DR5 onwards, this was not done and things worked just fine.

Launching tractor
-----------------

    qdo launch dr6 48 --cores_per_worker 8 --walltime=36:00:00 \
                      --script pb.sh \
                      --batchqueue regular --keep_env --batchopts "-C haswell"

The script in the example is a local copy of
`legacypipe/bin/pipebrick-checkpoint.sh`. Some lines of the script need
to be modified for everything to run:

    export PYTHONPATH=$CSCRATCH/DRcode/legacypipe/py:$PYTHONPATH
    outdir=$CSCRATCH/dr7out
    rundir=$CSCRATCH/DRcode

The arguments to runbrick can be modified in this script. Here is a
generic call:

    python ${rundir}/legacypipe/py/legacypipe/runbrick.py \ 
          --skip \ 
          --threads 16 \ 
          --skip-calibs \ 
          --checkpoint ${rundir}/checkpoints/${bri}/checkpoint-${brick}.pickle \ 
          --pickle "${rundir}/pickles/${bri}/runbrick-\%(brick)s-\%\%(stage)s.pickle" \ 
          --write-stage srcs \
          --brick $brick --outdir $outdir \  
          >> $log 2>&1 

The number of threads has to be smaller or equal to twice the number of
cores per task; skip-calibs assumes all the calibration files for this
brick have already been generated; skip will stop if there is already a
tractor catalogue for this brick. The checkpoint and pickle flags are
optional and the files generated don't get erased once the brick has
completed and the pickle files can be quite large. An easy way to clean
up is via the following script:

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
`/projecta` or `/project`. The log files, will be tarred by directory
and gzipped before being transferred (also, log files for bricks that
were processed partly on both machines need to be concatenated).

In the same directory, the contents of the `$LEGACY_SURVEY_DIR` (except
the images directory which contains only symbolic links) are also
copied.

End game
--------

There will inevitably come a point where bricks take a long time to
complete or fail to complete at all due to large blobs. These can be due
to very bright stars, large galaxies or large artefacts. For DR7, a new
option to `runbrick.py` was introduced: `–bail-out`. Re-starting the
remaining bricks from the checkpoints (making a backup might be prudent)
to finish the bricks can be done mostly on the debug queue.

Post-tractor processing
=======================

Bright neighbours
-----------------

For DR7, an extra step in the processing was introduced to go over the
tractor catalogues and check whether the neighbouring bricks have large
blobs near their border. This process modifies the\
`coadd/legacysurvey-<brickname>-maskbits.fits.gz` and tractor
catalogues. The script\
`legacypipe/bin/copymaskbits.py` makes a backup of these files. Since we
archive the tractor-i files, backing up the tractor catalogues is not
necessary. This step is most efficiently carried out using qdo using the
script\
`legacypipe/bin/qdo-post-process.sh`, where the tasks are, like for the
main processing, the brickname.

Top-level depth files
---------------------

The first step to compute the top-level depth information.

    python legacypipe/py/legacyanalysis/depth-histogram.py

Prior to running, the hard-coded location of the release directory (the
place where the coadd directory can be found) must be modified.

brick summary
-------------

The file `survey-bricks-drN.fits.gz ` is generated from the `nexp` files
found in the coadd subdirectories. It takes a while to run, so it's
betterto split the work manually and merge the output afterwards. For
DR6, this procedure took about 1.5hrs on one node using the interactive
queue on Cori.

    if [ "$NERSC_HOST" == "cori" ]; then
        builddir=$CSCRATCH/DRcode/build
    elif [ "$NERSC_HOST" == "edison" ]; then
        builddir=$SCRATCH/DRcode/build
    fi
    export PATH=$builddir/bin:$PATH
    export PYTHONPATH=$builddir/lib/python3.5/site-packages:$PYTHONPATH
    export PYTHONPATH=$builddir/lib/python:$PYTHONPATH
    export PYTHONPATH=/global/cscratch1/sd/desiproc/DRcode/legacypipe/py:$PYTHONPATH
    
    drdir=/global/projecta/projectdirs/cosmo/work/legacysurvey/dr6/
    for ((b=0; b<36; b++))
    do
        B=$(printf %02i $b)
        python -u $CSCRATCH/DRcode/legacypipe/py/legacyanalysis/brick-summary.py \
               --dr5 -o dr6-bricks-summary-$B.fits \
               $drdir/coadd/$B*/*/*-nexp-*.fits.fz > bs-$B.log 2>&1 &
    done

The `–dr5` flag is to indicate post DR4 tractor catalogue format. Once
the jobs are all done, merge and create summary plots:

    python $CSCRATCH/DRcode/legacypipe/py/legacyanalysis/brick-summary.py \
        --merge -o survey-bricks-dr6.fits dr6-bricks-summary-*.fits
    module load latex
    python $CSCRATCH/DRcode/legacypipe/py/legacyanalysis/brick-summary.py \
        --plot survey-bricks-dr6.fits
    gzip survey-bricks-dr6.fits

Note that for the plotting to work, the files
`mosaic-tiles_obstatus.fits`, `bass-tiles_obstatus.fits` and
`decam-tiles_obstatus.fits` have to be in the directory. They can be
obtained from the DESI SVN repository.

For DR7, the number of bricks per 10 degree of RA exceeded the Python
limit on the number of arguments, so that the procedure outlined above
doesn't work for DECaLS. Splitting the brick list in bins of 1 degree in
RA is easy to implement and can be done via qdo on the debug queue in a
relatively short amount of time with the script
`legacypipe/bin/qdo-brick-summary.sh` for which the tasks are the
subdirectory names `000` - `359`. Note that the code will crash if there
are no bricks in the directory, but this failure is inconsequential.

Sweeps and external matching
----------------------------

The next step is to generate the sweep files, which contain a subset of
the columns of the tractor catalogues and matching to external
catalogues. The `legacypipe/bin` directory contains slurm scripts to
generate these files. The commands in them clash with the desiproc
environment. For DR6, the commands were run from the interactive queue
on Cori; the sweeps and externals took about 30 minutes on one node.

We must first set the location of the input Tractor catalogs and the
location of the bricks file (which speeds up the process of scanning
through files) and the name of the environment variable which will hold
the list of Tractor catalogs, as well as the desired output directories:
e.g.:

    set -x
    export ATP_ENABLED=0
    
    outdir=/global/cscratch1/sd/desiproc/dr6-out
    drdir=/global/projecta/projectdirs/cosmo/work/legacysurvey/dr6
    export LEGACYPIPE_DIR=/global/cscratch1/sd/desiproc/DRcode/legacypipe
    
    export TRACTOR_INDIR=$drdir/tractor
    export BRICKSFILE=$drdir/survey-bricks.fits.gz
    export TRACTOR_FILELIST=$outdir/tractor_filelist
    export SWEEP_OUTDIR=$outdir/sweep
    export PYTHONPATH=$LEGACYPIPE_DIR/py:${PYTHONPATH}

Then, we build the list of tractor files for this data release and
submit the job. This can be done on an interactive queue. For DR7, the
process took about 90 minutes on one Cori Haswell node:

    find $TRACTOR_INDIR -name 'tractor-*.fits' > $TRACTOR_FILELIST
    time srun -u --cpu_bind=no -n 1 python $LEGACYPIPE_DIR/bin/generate-sweep-files.py \
              -v --numproc 16 -I -f fits -F $TRACTOR_FILELIST --schema blocks \
              -d $BRICKSFILE $TRACTOR_INDIR $SWEEP_OUTDIR

The files will be written to `$SWEEP_OUTDIR`. The number of CPUs used
was reduced by half to 16 for DR7 because of memory failure.

To generate the \"external files\" matched to other surveys (e.g. see
​http://legacysurvey.org/dr4/files/), we proceed in a similar fashion.
With the same definitions as for the sweeps, we add

    export EXTERNAL_OUTDIR=$CSCRATCH/$dr/external
    /usr/bin/mkdir -p $EXTERNAL_OUTDIR
    export SDSSDIR=/global/projecta/projectdirs/sdss/data/sdss

The files will be written to `$EXTERNAL_OUTDIR`. For DR6, we matched to
five external catalogues:

    time srun -u --cpu_bind=no -N 1 python $LEGACYPIPE_DIR/bin/match-external-catalog.py \
         -v --numproc 32 -f fits -F $TRACTOR_FILELIST \
         $SDSSDIR/dr12/boss/qso/DR12Q/DR12Q.fits \
         $TRACTOR_INDIR \
         $EXTERNAL_OUTDIR/survey-$dr-dr12Q.fits --copycols MJD PLATE FIBERID RERUN_NUMBER
    
    time srun -u --cpu_bind=no -N 1 python $LEGACYPIPE_DIR/bin/match-external-catalog.py \
         -v --numproc 32 -f fits -F $TRACTOR_FILELIST \
         $SDSSDIR/dr7/dr7qso.fit.gz \
         $TRACTOR_INDIR \
         $EXTERNAL_OUTDIR/survey-$dr-dr7Q.fits --copycols SMJD PLATE FIBER RERUN
    
    time srun -u --cpu_bind=no -N 1 python $LEGACYPIPE_DIR/bin/match-external-catalog.py \
         -v --numproc 32 -f fits -F $TRACTOR_FILELIST \
         $SDSSDIR/dr12/boss/qso/DR12Q/Superset_DR12Q.fits \
         $TRACTOR_INDIR \
         $EXTERNAL_OUTDIR/survey-$dr-superset-dr12Q.fits --copycols MJD PLATE FIBERID
    
    time srun -u --cpu_bind=no -N 1 python $LEGACYPIPE_DIR/bin/match-external-catalog.py \
         -v --numproc 32 -f fits -F $TRACTOR_FILELIST \
         $SDSSDIR/dr14/sdss/spectro/redux/specObj-dr14.fits \
         $TRACTOR_INDIR \
         $EXTERNAL_OUTDIR/survey-$dr-specObj-dr14.fits --copycols MJD PLATE FIBERID RUN2D
    
    time srun -u --cpu_bind=no -N 1 python $LEGACYPIPE_DIR/bin/match-external-catalog.py \
         -v --numproc 32 -f fits -F $TRACTOR_FILELIST \
         $SDSSDIR/dr14/eboss/qso/DR14Q/DR14Q_v4_4.fits \
         $TRACTOR_INDIR \
         $EXTERNAL_OUTDIR/survey-$dr-dr14Q_v4_4.fits --copycols MJD PLATE FIBERID

Note that the script `legacypipe/bin/sweep-external-env.sh` provides a
template to set all the relevant variables described above.

Release directory structure and checksum files
----------------------------------------------

Most of the tractor input and output is already in release form, but
there are two exceptions:

-   The sweep output is in `sweep/7.0/`, where the number indicates the
    release, instead of simply `sweep`.

-   The tractor log files are tarred by output subdirectory and gzipped.

The tractor code produces a file for each brick containing the checksums
for all the output files for that brick. This doesn't follow the
convention for the naming and location of checksum files for a data
release, so a first step is to verify that the output files match the
checksums and re-package them into the appropriate format. The next step
is to generate the checksums for the rest of the output. This can be
handled by the following scripts (with directory location appropriately
amended):

    legacypipe/bin/tar-logfiles.py
    legacypipe/bin/post-process-qa.py
    legacypipe/bin/repackchecksum.py
    legacypipe/bin/checksumming-non-tractor.sh
    legacypipe/bin/checksumming-calibs.py

Other "after-burner" stages
---------------------------

-   Load images into the Legacy Survey viewer (Dustin Lang).

-   Generate a picture gallery (John Moustakas & Ben Weaver).

-   Final vetting and moving into place (Ben Weaver).

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
