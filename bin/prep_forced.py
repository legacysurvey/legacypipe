from astropy.io import fits
import sys
with open('/project/projectdirs/uLens/ZTF/Tractor/data/ZTF18abcfdzu/tractor/survey-ccds-ztf.fits','rb') as f: 
    hdul=fits.open(f)
    print(hdul[1].data)  
    expnums = hdul[1].data['EXPNUM']
    ccdsnames = hdul[1].data['CCDNAME']
print(expnums,ccdsnames)

for expnum in expnums:
    with open('runbrick_ZTF18abcfdzu.sh','a') as g:
        g.write('python $PROJECTPATH/legacypipe/py/legacypipe/forced_photom.py --no-ceres --no-move-gaia --catalog-dir=$LEGACY_SURVEY_DIR --catalog $LEGACY_SURVEY_DIR/tractor-i/cus/tractor-custom-230217p54215.fits ' +str(expnum)+' CCD0  $LEGACY_SURVEY_DIR/tractor/cus/forced_'+str(expnum)+'.fits\n\n')
