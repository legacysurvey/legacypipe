# Outline of DECam Legacy Survey reductions for Data Release 2

## Bricks

Processing is done on bricks, which are defined by RA,Dec boxes.  For
processing, we define a 3600x3600-pixel TAN projection centered on the
brick center, which results in slight overlap between neighboring
bricks.  Some steps of the processing occur in this brick pixel space.

The first step is to find CCD images that overlap the current brick.
We then apply a photometricity cut based on the photometric
calibration parameters for each CCD.  The cuts include:

- exptime >= 30s
- number of matched Pan-STARRS stars >= 20
- |per-CCD zeropoint - per-exposure zeropoint| <= 0.1 mag
- (per-exposure zeropoint - nominal) within the range (-0.5, 0.25).  Nominal zeropoints: g=25.08, r=25.29, z=24.92.

The overlapping subimage of each photometric CCD is then read.

## Source detection

Next, we detect sources.

In DR1 we initialized from SDSS measurements.  In DR2, we are going to
start from scratch.

(Tractor-on-bricks...)

Deep source detection proceeds by convolving each image by a Gaussian
approximation to its PSF, given by the CP-measured FWHM.  Each image
is resampled to the brick pixel space, and the images within each band
are summed with weights that maximize the point-source detection
signal-to-noise.  We then compute and subtract a spatial median
(binning 4x4 and then filtering on a 50-pixel scale).  This is
intended to eliminate spurious detections in the halos of bright
stars.  [May not be required for DR2 since we have better sky
subtraction.]  Next, we define a number of spectral energy
distributions (SEDs) of sources we wish to detect.  These include
single-band g, r, and z as well as "flat" (zero color) and "red" (g-r
= r-z = 1) SEDs.  These SEDs are used to weight the detection maps
from the individual bands into a single detection map.

For each SED-matched detection map, we perform the following.  We find
pixels above our signal-to-noise detection threshold (6), dilate by 8
pixels, and fill holes.  We then find pixels that are larger than
their 8 neighbors; these are potential new peaks.  From brightest
(highest peak) to faintest, we decide whether to keep the peak, based
on whether it is separated by a low enough saddle from previously
found sources.  The saddle depth is defined as the maximum of 20% of
the peak height, or 2 sigma.  Peaks that fail this test are dropped.

Any region of pixels containing an active source is added to a running
mask of "hot" pixels.  The union of these pixels defines the "blobs"
of pixels we will use in the fitting.

The result of our source detection step is simply a list of x,y
positions in brick pixel space, plus the "blob" map of connected
pixels.

## Fitting

Fitting then proceeds on a blob-by-blob basis.

Subimages of each CCD are cut out.  The blob (which is defined in
brick pixel space) is resampled to each image, and the
inverse-variance maps of pixels outside the blob are zeroed out;
pixels outside the blob do not influence the fitting.  If the blob is
smaller than 400 by 400 pixels, we instantiate a constant PSF in the
blob center.

For sources we detected, we do not yet have any measurements other
than their nearest-pixel peak.  We instantiate PointSource models for
each source.

Next, we perform a quick linear fit of the flux for each source, with
its model held fixed.

Next, we fit sources in order of brightness.  We begin by subtracting
off the initial models for all sources.  For each source, we add back
in its initial model, then optimize the model (without changing the
model type).  We then subtract off the optimized model before
proceeding to the next source.

If there are 10 of fewer sources in the blob, we perform a
simultaneous optimization.

### Model selection

We then proceed to perform model selection on each source, in
decreasing order of brightness.  As before, we subtract the initial
models for all sources, then when considering a source, we add its
initial model back in.  As a baseline, we compute the log-likelihood
of the images without the source.

For all sources, we start by trying the PointSource and SimpleGalaxy
models.  For each model to try, we optimize the model, then compute
the final log-likelihood.  For this, rather than the straight
log-likelihood, we penalize models with negative fluxes.  We compute
the optimized chi-squared sums per band.  For bands with positive
flux, the chi-squared improvement versus the "nothing" model count *for*
the model, while bands with negative flux count *against* the model.
This is quite a strong penalty for negative fluxes!

For sources that were initially point sources, if the SimpleGalaxy
model is superior to the PointSource model, or if the PointSource
model is better than the "nothing" model by more than a chi-squared
difference of 400, we compute the DevGalaxy and ExpGalaxy models.

If one of the DevGalaxy or ExpGalaxy is better than the PointSource or
SimpleGalaxy model plus a margin of 12 in chi-squared, then we compute
the CompositeGalaxy model.

The ExpGalaxy and DevGalaxy models are initialized to have the same
position and flux as the PointSource or SimpleGalaxy model, and a
shape with effective radius of about 0.4" and zero ellipticity
(round).  The CompositeGalaxy model is initialized with the ExpGalaxy
and DevGalaxy shapes and the flux split equally between the two
components.

All the galaxy models are fit using a "softened" ellipse
parameterization with a prior on the ellipticities.  The ellipse
parameters include log-radius and softened e1 and e2 ellipticity
components.  The "softening" avoids the hard edge in the parameter
space at |e| = e1^2 + e2^2 = 1, by putting |e| through a sigmoid
function, 1 - exp(-|e|).  This means that the parameter space seen by
the optimizer is unbounded, which is convenient.  We put a Gaussian
prior on the ellipticity components, centered on zero and with
standard deviation of 0.25.  We also apply a top-hat prior that the
log-radius be less than 5 (an effective radius of about 150
arcseconds).

Once all the relevant models have been computed, we perform model
selection.  This is based on the (negative flux penalized) chi-squared
differences versus the "no-source" model.  The recipe is as follows.
In order to be kept at all, at least one of the models must have a
parameter-penalized chi-squared difference greater than 25.  The
parameter penalization is 2 for the PointSource and SimpleGalaxy
models, 5 for DevGalaxy and ExpGalaxy, and 9 for CompositeGalaxy.  If
a source passes this criterion and is to be kept, we first choose
between the PointSource and SimpleGalaxy models, based on straight
chi-squared.  If the DevGalaxy and ExpGalaxy models were computed, we
switch to the better of these two models if it surpasses the
PointSource or SimpleGalaxy model by the maximum of a chi-squared
difference of 12, and 2% of the PointSource-versus-nothing chi-squared
difference.  This means that brighter sources require a larger
improvement from the simple models to the galaxy models.  Similarly,
if the DevGalaxy or ExpGalaxy is selected, we switch to the
CompositeGalaxy model if it exceeds the DevGalaxy or ExpGalaxy by that
same margin.

Note that the DCHISQ values in the catalogs are the
(negative-flux-penalized) chi-squared differences versus the
"no-source" model used for model selection.

Finally, after selecting the model, we subtract it and continue model
selection for the remaining sources in the blob.

After model selection has been run for each source, we compute the
uncertainties (inverse-variances) on each source's model parameters,
as well as the various other metrics (RCHI, FRACFLUX, FRACIN,
FRACMASKED).

## Coadd & catalog production

Once all the blobs have been run, we produce the coadded data and
model images, the residual maps, depth maps, and depth histograms.  We
perform aperture photometry on the coadded images at each source's
position (with no deblending).  We then perform forced photometry on
the unWISE images, and finally, write out the catalog.

