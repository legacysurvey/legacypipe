import sys
import os
import numpy as np
import argparse
import glob
from astropy.io import fits

from astrometry.util.fits import fits_table
from astrometry.util.starutil_numpy import degrees_between, hmsstring2ra, dmsstring2dec
from astrometry.util.util import Tan, Sip, anwcs_t

def exposure_metadata(filenames, hdus=None, trim=None):
    '''
    Creates a CCD table row object by reading metadata from a FITS
    file header.

    Parameters
    ----------
    filenames : list of strings
        Filenames to read
    hdus : list of integers; None to read all HDUs
        List of FITS extensions (HDUs) to read
    trim : string
        String to trim off the start of the *filenames* for the
        *image_filename* table entry

    Returns
    -------
    A table that looks like the CCDs table.
    '''
    nan = np.nan
    primkeys = [('FILTER',''),
                ('RA', nan),
                ('DEC', nan),
                ('AIRMASS', nan),
                ('DATE-OBS', ''),
                ('EXPTIME', nan),
                ('EXPNUM', 0),
                ('JULIAN', 0),
                ('PROPID', ''),
                ('INSTRUME', ''),
                ('SEEING', nan),
    ]
    hdrkeys = [('SKYVAL', nan),
               ('GAIN', nan),
               ('FWHM', nan),
               ('CRPIX1',nan),
               ('CRPIX2',nan),
               ('CRVAL1',nan),
               ('CRVAL2',nan),
               ('CD1_1',nan),
               ('CD1_2',nan),
               ('CD2_1',nan),
               ('CD2_2',nan),
               ('CCDNAME',''),
               ('CCDNUM',''),
               ]

    otherkeys = [('IMAGE_FILENAME',''), ('IMAGE_HDU',0),
                 ('HEIGHT',0),('WIDTH',0),
                 ]

    allkeys = primkeys + hdrkeys + otherkeys

    vals = dict([(k,[]) for k,d in allkeys])

    for i,fn in enumerate(filenames):
        print('Reading', (i+1), 'of', len(filenames), ':', fn)
        F = fits.open(fn)
        primhdr = F[0].header
        expstr = '%08i' % primhdr['EXPNUM']

        # # Parse date with format: 2014-08-09T04:20:50.812543
        # date = datetime.datetime.strptime(primhdr.get('DATE-OBS'),
        #                                   '%Y-%m-%dT%H:%M:%S.%f')
        # # Subract 12 hours to get the date used by the CP to label the night;
        # # CP20140818 includes observations with date 2014-08-18 evening and
        # # 2014-08-19 early AM.
        # cpdate = date - datetime.timedelta(0.5)
        # #cpdatestr = '%04i%02i%02i' % (cpdate.year, cpdate.month, cpdate.day)
        # #print 'Date', date, '-> CP', cpdatestr
        # cpdateval = cpdate.year * 10000 + cpdate.month * 100 + cpdate.day
        # print 'Date', date, '-> CP', cpdateval

        cpfn = fn
        if trim is not None:
            cpfn = cpfn.replace(trim, '')
        print('CP fn', cpfn)

        if hdus is not None:
            hdulist = hdus
        else:
            hdulist = range(1, len(F))

        for hdu in hdulist:
            hdr = F[hdu].header

            #'extname': 'S1', 'dims': [4146L, 2160L]
            W,H = hdr['NAXIS1'],hdr['NAXIS2'] 

            for k,d in primkeys:
                vals[k].append(primhdr.get(k, d))
            for k,d in hdrkeys:
                vals[k].append(hdr.get(k, d))

            vals['IMAGE_FILENAME'].append(cpfn)
            vals['IMAGE_HDU'].append(hdu)
            vals['WIDTH'].append(int(W))
            vals['HEIGHT'].append(int(H))
            #FWHM and SEEING not in image fits header, get it from psfex file
            F_psfex = fits.open(os.path.join(os.path.dirname(fn),'./psfex',os.path.basename(fn)))
            vals['FWHM'].append( float(F_psfex[hdu].header['PSF_FWHM']) )
            vals['SEEING'].append( vals['FWHM'][-1]/2.355 )
            print('how convert FWHM to SEEING?')
            raise ValueError

    T = fits_table()
    for k,d in allkeys:
        T.set(k.lower().replace('-','_'), np.array(vals[k]))
    #T.about()

    # DECam: INSTRUME = 'DECam'
    T.rename('INSTRUME'.lower(), 'camera') 
    T.camera = np.array([t.lower() for t in T.camera])
    
    T.rename('gain', 'ARAWGAIN'.lower()) #ARAWGAIN is name for gain from decam parser
    T.rename('SKYVAL'.lower(), 'AVSKY'.lower()) #AVSKY is name for sky from decam parser
    T.rename('JULIAN'.lower(), 'MJD-OBS'.lower()) #MJD-OBS is name for julian from decam parser
    
    T.ccdname = np.array([t.strip() for t in T.ccdname])
    T.ccdnum = np.array([t.strip()[-1] for t in T.ccdname])
    
    T.filter = np.array([s.strip()[0] for s in T.filter])
    T.ra_bore  = np.array([hmsstring2ra (s) for s in T.ra ])
    T.dec_bore = np.array([dmsstring2dec(s) for s in T.dec])

    T.ra  = np.zeros(len(T))
    T.dec = np.zeros(len(T))
    for i in range(len(T)):
        W,H = T.width[i], T.height[i]

        wcs = Tan(T.crval1[i], T.crval2[i], T.crpix1[i], T.crpix2[i],
                  T.cd1_1[i], T.cd1_2[i], T.cd2_1[i], T.cd2_2[i], float(W), float(H))
        
        xc,yc = W/2.+0.5, H/2.+0.5
        rc,dc = wcs.pixelxy2radec(xc,yc)
        T.ra [i] = rc
        T.dec[i] = dc

    return T

parser = argparse.ArgumentParser(description="test")
parser.add_argument("-search_str",action="store",help='path/to/images/ + str to search with')
args = parser.parse_args()

fns= glob.glob(args.search_str)
if len(fns) == 0: raise ValueError
#make ccd table
T=exposure_metadata(fns)
for i in T.get_columns(): print('%s=' % i,T.get(i))
T.writeto('bok-ccds.fits')
print 'wrote ccds table'
