"""
Script to correct airmass values in survey-ccds and ccds-annotated files
for Mosaic3 exposures.
Code is extracted from legacypipe/py/legacyzpts/legacy_zeropoints.py
"""

from astropy.io import fits
import sys
import numpy as np

hduS = fits.open("v20200124/survey-ccds-mosaic-dr9.fits.gz")
hduA = fits.open("v20200124/ccds-annotated-mosaic-dr9.fits.gz")

nccds = len(hduS[1].data)
#print(nccds)

def get_site():
    from astropy.coordinates import EarthLocation
    from astropy.units import m
    from astropy.utils import iers
    iers.conf.auto_download = False  
    return EarthLocation(-1994503. * m, -5037539. * m, 3358105. * m)

def get_airmass(mjd_obs, exptime, ra_bore, dec_bore, site):
    from astropy.time import Time
    from astropy.coordinates import SkyCoord, AltAz
    time = Time(mjd_obs+0.5*exptime/24./3600., format='mjd')
    coords = SkyCoord(ra_bore, dec_bore, unit='deg')
    altaz = coords.transform_to(AltAz(obstime=time, location=site))
    airmass = altaz.secz
    return airmass

def get_transparency(dzpt, kext, airmass):
    transp = 10.**(-0.4 * (-dzpt - kext * (airmass - 1.0)))
    return transp

site = get_site()
keff = 0.06 # Good for Mosaic3 z-band only
zp0 = 26.552 # Same
for i in range(nccds):
    fn = hduS[1].data['image_filename'][i]
    ccdname = hduS[1].data['ccdname'][i]
    if hduA[1].data['image_filename'][i] != fn or hduA[1].data['ccdname'][i] != ccdname:
        print("survey-ccds and annotated-ccds are not in the same order!")
        sys.exit()
    mjd_obs = hduA[1].data['mjd_obs'][i]
    exptime = hduA[1].data['exptime'][i]
    ra_bore = hduA[1].data['ra_bore'][i]
    dec_bore = hduA[1].data['dec_bore'][i]
    airmass = get_airmass(mjd_obs, exptime, ra_bore, dec_bore, site)
    zpt = hduA[1].data['zpt'][i]
    dzpt = zpt = zp0
    transp = get_transparency(dzpt, keff, airmass)
    hduS[1].data['airmass'][i] = airmass
    hduA[1].data['airmass'][i] = airmass
    #hduS[1].data['transp'][i] = transp
    #hduA[1].data['transp'][i] = transp
    if np.mod(i,1000) == 0:
        print(str(i)+" CCDs out of "+str(nccds)+" updated.")

hduA.writeto("./ccds-annotated-mosaic-dr9.fits")
hduS.writeto("./survey-ccds-mosaic-dr9.fits")

