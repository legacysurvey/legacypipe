#! /bin/bash
# Validate the number of CCDs in each test region against DR6 & DR7.

# DR8 CCDs
#for region in dr8-test-hsc-sgc dr8-test-hsc-ngc dr8-test-edr dr8-test-hsc-north dr8-test-deep2-egs dr8-test-s82 dr8-test-overlap; do
for region in dr8-test-overlap; do
  for camera in 90prime-mosaic decam; do
      python $LEGACYPIPE_DIR/py/legacypipe/queue-calibs.py --ccds ../survey-ccds-dr8b-$camera-nocuts.kd.fits --touching --region $region --save_to_fits --ignore_cuts
      /usr/bin/mv -f bricks-$region-touching.fits dr8b/bricks-$region-$camera.fits 
      /usr/bin/mv -f ccds-$region-cut.fits dr8b/ccds-$region-$camera.fits 
      /usr/bin/rm -f bricks-$region-touching.fits bricks-$region-cut.fits ccds-$region-cut.fits 
  done
done

# DR7 CCDs
#for region in dr8-test-hsc-sgc dr8-test-hsc-ngc dr8-test-edr dr8-test-hsc-north dr8-test-deep2-egs dr8-test-s82 dr8-test-overlap; do
for region in dr8-test-overlap; do
  python $LEGACYPIPE_DIR/py/legacypipe/queue-calibs.py --ccds /global/project/projectdirs/cosmo/work/legacysurvey/dr7/survey-ccds-dr7.kd.fits --touching --region $region --save_to_fits --ignore_cuts
  /usr/bin/mv -f bricks-$region-touching.fits dr7/bricks-$region.fits 
  /usr/bin/mv -f ccds-$region-cut.fits dr7/ccds-$region.fits 
  /usr/bin/rm -f bricks-$region-touching.fits bricks-$region-cut.fits ccds-$region-cut.fits 
done

# DR6 CCDs
#for region in dr8-test-hsc-sgc dr8-test-hsc-ngc dr8-test-edr dr8-test-hsc-north dr8-test-deep2-egs dr8-test-s82 dr8-test-overlap; do
for region in dr8-test-overlap; do
  python $LEGACYPIPE_DIR/py/legacypipe/queue-calibs.py --ccds /global/project/projectdirs/cosmo/work/legacysurvey/dr6/survey-ccds-dr6plus.kd.fits --touching --region $region --save_to_fits --ignore_cuts
  /usr/bin/mv -f bricks-$region-touching.fits dr6/bricks-$region.fits 
  /usr/bin/mv -f ccds-$region-cut.fits dr6/ccds-$region.fits 
  /usr/bin/rm -f bricks-$region-touching.fits bricks-$region-cut.fits ccds-$region-cut.fits 
done
