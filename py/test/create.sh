#! /bin/bash

python legacypipe/runbrick.py -b 2447p120 --zoom 1020 1070 2775 2815 --no-wise --force-all --outdir out3 -s image_coadds
python legacyanalysis/create_testcase.py out3/coadd/244/2447p120/legacysurvey-2447p120-ccds.fits testcase3 2447p120

     
#python legacypipe/runbrick.py -b 1102p240 --zoom 500 600 650 750 --outdir tc6 --force-all --no-write -s image_coadds
#python legacyanalysis/create_testcase.py tc6/coadd/110/1102p240/legacysurvey-1102p240-ccds.fits testcase6 1102p240

 
