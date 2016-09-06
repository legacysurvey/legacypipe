Running the legacypipe code
===========================

Preparing a LEGACY_SURVEY_DIR directory
---------------------------------------

The set of input images, calibration files, and output products, along
with metadata, are all placed in a directory.  The path to that directory
should be placed in an environment variable called LEGACY_SURVEY_DIR.  An
example of a directory is in the github repository:
    `http://github.com/legacysurvey/legacypipe-dir`
so to start out, you will probably want to clone that directory and then
modify it to your purposes:

    git clone http://github.com/legacysurvey/legacypipe-dir survey-dir
    export LEGACY_SURVEY_DIR=$(pwd)/survey-dir

You need not (indeed, probably don't want to) be in that directory
while running the code; typically it is easier to *cd* to the
*legacypipe/py* directory and run the python codes from there.

    git clone http://github.com/legacysurvey/legacypipe legacypipe
    cd legacypipe/py


Preparing the list of images to process
---------------------------------------

To run the legacypipe code on your own data, the first step is to
produce a *survey-ccds* table.  This is a FITS binary table that lists
metadata about the images that are to be processed.  In DECaLS, MzLS
and BASS, this file is created by the `decstat` or `mosstat` code,
then massaged by the `legacypipe/merge-zeropoints.py` code.

The CCDs table must contain the following fields:

 * camera -- string, see below
 * image_filename -- string, path relative to `$LEGACY_SURVEY_DIR/images`
 * image_hdu -- integer, FITS extension containing pixels
 * expnum -- integer exposure number counter.  This plus `ccdname` is used to identify a CCD.
 * ccdname -- string
 * filter -- string of length 1
 * exptime -- float, exposure time in seconds
 * seeing -- float, seeing FWHM in arcseconds
 * zpt -- float, zeropoint (average for the exposure)
 * crpix1 -- float, WCS reference pixel X
 * crpix2 -- float, WCS reference pixel Y
 * crval1 -- float, WCS reference RA
 * crval2 -- float, WCS reference Dec
 * cd1_1 -- float, WCS transformation matrix
 * cd1_2 -- float, WCS transformation matrix
 * cd2_1 -- float, WCS transformation matrix
 * cd2_2 -- float, WCS transformation matrix
 * width -- integer, image width in pixels
 * height -- integer, image height in pixels
 * ra -- float, image center RA
 * dec -- float, image center Dec

Each row of the table should correspond to one contiguous chunk of
pixels contained in one HDU of a FITS file, that is described by a
single astrometric World Coordinate System and photometric solution.

This so-called "CCDs table" file must be placed in your
$LEGACY_SURVEY_DIR directory.  The pipeline will look for all files
named "survey-ccds-*.fits.gz" and read each one to determine which
input images exist.  Any field that does not exist in a table will be
filled in with zeroes.

The `camera` field has a special meaning and purpose in the code: it
is a string such as "decam", "mosaic", "90prime" that tells the code
which python class to use to interpret the image.  There is a
dictionary (in the LegacySurveyData class) that holds this mapping.
If you need to add a new camera to this mapping, you might want to
either add it directly to the LegacySurveyData class, or create a
small wrapper script that adds your new class and then calls the main
`runbrick` code; for an example of this, please see
`legacypipe/runcosmos.py`.

Calibrating the images
----------------------

Some of the camera-specific image-handling classes perform some
additional calibration of each image -- for example, to produce a PSF
model or a sky model.  These calibration results are saved in files
within the `calib` directory of $LEGACY_SURVEY_DIR.  Each
camera-specific class defines which calibration processes it wants to
run.  The legacypipe can either run these calibration processes as
required when an image is about to be read, or they can be run in
pre-processing.

(more on this; queue-calibs)

Bricks
------

A `brick` is a region of the sky defined by an RA,Dec box.  The list
of bricks to be processed is contained in a FITS table in the
$LEGACY_SURVEY_DIR directory, `survey-bricks.fits.gz`.  You could
certainly define your own bricks if desired.  Bricks are named like
RRRR[pm]DDD, where RRRR is a (zero-padded) 4-digit string of the RA
times 10 (`'%04i' % (ra*10)` in python).  Similarly, DDD is a 3-digit
string of the Dec times 10.

Running the pipeline
--------------------

The main script is `legacypipe/runbrick.py`.  It takes many
command-line arguments, but at the very least you will need:

 * `--brick <brickname>`: this determines which part of sky to run.

By default, our bricks are 3600 x 3600 pixels, with a nominal pixel
scale of 0.262 arcseconds per pixel.  These can be adjusted with:

 * `--width <W> --height <H> --pixscale <p>` where `W,H` are in pixels, and `p` in arcsec/pix.

If you are using our default bricks, you should ensure that the brick
size is at least 0.25 degrees plus some padding.

It is also possible to run a "custom" brick at a given RA,Dec center:

 * `--radec <ra> <dec>` where `<ra>` and `<dec>` are in degrees.

By default, output products are written to the current directory; to change that:
 * `--outdir <d>`

You can also set the directory used instead of the $LEGACY_SURVEY_DIR environment variable;

 * `--survey-dir <d>`

The code uses the `stages` framework, which allows saving the state of
a computation between stages of processing.  State is saved in python
"pickle" files.  There are dependencies between stages, so if a
computation is resumed later, a pickle files can be read and the
computation resumed.  The stages in the `runbrick` code, and their
prerequisites, are listed in the `prereqs` dictionary in the
`legacypipe/runbrick.py` code.  There are some flags to control the stage
behavior:

 * `--stage <s>`, string `<s>`.  Which stage(s) (plus their
   prerequisites) to run.  Stages include:
      * `tims`: reads input images
      * `mask_junk`: eliminates satellite trails
      * `image_coadds`: early coadds
      * `srcs`: detects sources
      * `fitblobs`: fits sources
      * `coadds`: produces coadds, including models and residuals
      * `wise_forced`: WISE forced photometry
      * `writecat`: writes output tractor table

 * `--force-all`: ignore all pickle files and run all required stages
 * `--force <s>`: force a single stage
 * `--no-write`: do not write out pickle files
 * `--pickle <s>`: set the pickle filename pattern.  This has a
   somewhat silly format, because it goes through two rounds of string
   substitution.  The default is
   `pickles/runbrick-%(brick)s-%%(stage)s.pickle` (which you must put
   within single-quotes on the command-line to avoid strange shell
   behavior).  This is a python string-formatting string.  Note that
   first the `brick` is substituted, then the `stage` is substituted
   later, so the `%` of the stage formatting string is escaped with `%%`.




