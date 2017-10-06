import matplotlib
matplotlib.use('Agg')
from astrometry.util.fits import *
from legacypipe.survey import *
import numpy as np
import pylab as plt

survey = LegacySurveyData()
ccds = survey.get_ccds()
print(len(ccds), 'CCDs')

# Check consistency -- yes.
# psfex = []
# I = np.flatnonzero(np.isfinite(ccds.fwhm))
# I = np.random.permutation(I)[:100]
# for i in I:
#     im = survey.get_image_object(ccds[i])
#     psf = im.read_psf_model(0, 0, pixPsf=True)
#     psfex.append(psf.fwhm)
# 
# plt.clf()
# plt.plot(ccds.fwhm[I], psfex, 'b.')
# plt.xlabel('CCDs FWHM')
# plt.ylabel('PsfEx FWHM')
# plt.savefig('fwhm.png')

I = np.flatnonzero(np.logical_not(np.isfinite(ccds.fwhm)))
print(len(I), 'with NaN FWHM')
for i in I:
    im = survey.get_image_object(ccds[i])
    #try:
    psf = im.read_psf_model(0, 0, pixPsf=True)
    # except:
    #     import traceback
    #     traceback.print_exc()
    #     continue
    ccds.fwhm[i] = psf.fwhm
ccds.writeto('ccds-fwhm.fits')
