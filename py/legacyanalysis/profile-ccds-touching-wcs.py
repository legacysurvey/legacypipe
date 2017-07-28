import numpy as np
from astrometry.util.util import Tan
from astrometry.util.miscutils import polygons_intersect
from astrometry.util.starutil_numpy import degrees_between
from legacypipe.survey import LegacySurveyData

@profile
def ccds_touching_wcs(targetwcs, ccds, ccdrad=0.17, polygons=True):
    '''
    targetwcs: wcs object describing region of interest
    ccds: fits_table object of CCDs

    ccdrad: radius of CCDs, in degrees.  Default 0.17 is for DECam.
    #If None, computed from T.

    Returns: index array I of CCDs within range.
    '''
    trad = targetwcs.radius()
    if ccdrad is None:
        ccdrad = max(np.sqrt(np.abs(ccds.cd1_1 * ccds.cd2_2 -
                                    ccds.cd1_2 * ccds.cd2_1)) *
                     np.hypot(ccds.width, ccds.height) / 2.)

    rad = trad + ccdrad
    r,d = targetwcs.radec_center()
    I, = np.nonzero(np.abs(ccds.dec - d) < rad)
    I = I[np.atleast_1d(degrees_between(ccds.ra[I], ccds.dec[I], r, d) < rad)]

    if not polygons:
        return I
    # now check actual polygon intersection
    tw,th = targetwcs.imagew, targetwcs.imageh
    targetpoly = [(0.5,0.5),(tw+0.5,0.5),(tw+0.5,th+0.5),(0.5,th+0.5)]
    cd = targetwcs.get_cd()
    tdet = cd[0]*cd[3] - cd[1]*cd[2]
    if tdet > 0:
        targetpoly = list(reversed(targetpoly))
    targetpoly = np.array(targetpoly)

    keep = []
    for i in I:
        W,H = ccds.width[i],ccds.height[i]
        wcs = Tan(*[float(x) for x in
                    [ccds.crval1[i], ccds.crval2[i], ccds.crpix1[i], ccds.crpix2[i],
                     ccds.cd1_1[i], ccds.cd1_2[i], ccds.cd2_1[i], ccds.cd2_2[i], W, H]])
        cd = wcs.get_cd()
        wdet = cd[0]*cd[3] - cd[1]*cd[2]
        poly = []
        for x,y in [(0.5,0.5),(W+0.5,0.5),(W+0.5,H+0.5),(0.5,H+0.5)]:
            rr,dd = wcs.pixelxy2radec(x,y)
            ok,xx,yy = targetwcs.radec2pixelxy(rr,dd)
            poly.append((xx,yy))
        if wdet > 0:
            poly = list(reversed(poly))
        poly = np.array(poly)
        if polygons_intersect(targetpoly, poly):
            keep.append(i)
    I = np.array(keep)
    return I

def setup():
    survey = LegacySurveyData()
    ccds = survey.get_annotated_ccds()
    return ccds

def example():
    pixscale = 0.25       # [arcsec/pix]
    radius = 45.0         # [arcsec]
    ra, dec = 120.0, 15.0 # [degrees]
    diam = np.ceil(radius/pixscale).astype('int16') # [pixels]

    wcs = Tan(ra, dec, diam/2+0.5, diam/2+0.5,
              -pixscale/3600.0, 0.0, 0.0, pixscale/3600.0,
              float(diam), float(diam))
    
    return wcs

def doit(ccds, wcs):
    ccds_touching_wcs(wcs, ccds)

if __name__ == '__main__':

    ccds = setup()
    wcs = example()

    doit(ccds, wcs)
