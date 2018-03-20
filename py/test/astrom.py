import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np
from legacypipe.survey import LegacySurveyData
from legacyanalysis.gaiacat import GaiaCatalog
from legacypipe.survey import GaiaSource, GaiaPosition, LegacySurveyWcs
from astrometry.util.util import Tan
from astrometry.util.starutil_numpy import mjdtodate
from tractor import TAITime

#ra,dec = 357.3060, 2.3957
#ccd1 = ccds[(ccds.expnum == 563212) * (ccds.ccdname == 'N17')]
ra,dec = 124.0317, 1.3028
expnum, ccdname = 393203, 'N11'

survey = LegacySurveyData()

W,H = 200,200
pixscale = 0.262
cd = pixscale / 3600.
targetwcs = Tan(ra, dec, W/2., H/2., -cd, 0., 0., cd, float(W), float(H))

rr,dd = targetwcs.pixelxy2radec([1,W,W,1,1], [1,1,H,H,1])
targetrd = np.vstack((rr,dd)).T

ccds = survey.ccds_touching_wcs(targetwcs)
print(len(ccds), 'CCDs touching WCS')

print('MJDs', ccds.mjd_obs)
ccds.writeto('test-ccds.fits')

ccd1 = ccds[(ccds.expnum == expnum) * (ccds.ccdname == ccdname)]
print('CCD1:', ccd1)
im1 = survey.get_image_object(ccd1[0])
print('Im:', im1)

#wcs = im1.get_wcs()
#x0,x1,y0,y1,slc = im1.get_image_extent(wcs=wcs, radecpoly=targetrd)

tim1 = im1.get_tractor_image(radecpoly=targetrd, pixPsf=True, hybridPsf=True,
                             normalizePsf=True, splinesky=True)
print('Tim', tim1)

##
tims = [tim1]
ccd1.ccd_x0 = np.array([tim.x0 for tim in tims]).astype(np.int16)
ccd1.ccd_y0 = np.array([tim.y0 for tim in tims]).astype(np.int16)
ccd1.ccd_x1 = np.array([tim.x0 + tim.shape[1]
                        for tim in tims]).astype(np.int16)
ccd1.ccd_y1 = np.array([tim.y0 + tim.shape[0]
                        for tim in tims]).astype(np.int16)
ccd1.writeto('ccd1.fits')



bands = ['g','r','z']


gaia = GaiaCatalog().get_catalog_in_wcs(targetwcs)
print('Got Gaia stars:', gaia)
gaia.about()

gaia.G = gaia.phot_g_mean_mag
gaia.pointsource = np.logical_or(
    (gaia.G <= 18.) * (gaia.astrometric_excess_noise < 10.**0.5),
    (gaia.G >= 18.) * (gaia.astrometric_excess_noise < 10.**(0.5 + 1.25/4.*(gaia.G-18.))))

gaia.writeto('test-gaia.fits')

gaiacat = []
for g in gaia:
    gaiacat.append(GaiaSource.from_catalog(g, bands))
    print('Created source:', gaiacat[-1])


print(); print()
# Select the largest-parallax source.

gaia.parallax[np.logical_not(np.isfinite(gaia.parallax))] = 0.
i = np.argmax(gaia.parallax)
gsrc = gaia[i]
src = GaiaSource.from_catalog(gsrc, bands)

print('Source:', src)

ra,dec = src.pos.ra, src.pos.dec

mjdmin = ccds.mjd_obs.min()
mjdmax = ccds.mjd_obs.max()

wcs = tim1.wcs

print('WCS position:')
pos0 = wcs.positionToPixel(src.pos)
print(pos0)

rr,dd = [],[]
for mjd in np.arange(mjdmin, mjdmax, 10):
    rd = src.pos.getPositionAtTime(TAITime(None, mjd=mjd))
    #print('Position at MJD', mjd, ':', rd)
    rr.append(rd.ra)
    dd.append(rd.dec)

tt = []
for ccd in ccds:
    rd = src.pos.getPositionAtTime(TAITime(None, mjd=ccd.mjd_obs))
    rr.append(rd.ra)
    dd.append(rd.dec)
    date = mjdtodate(ccd.mjd_obs)
    tt.append('%4i-%02i-%02i %i %s %s' % (date.year, date.month, date.day,
                                          ccd.expnum, ccd.ccdname, ccd.filter))
rr = np.array(rr)
dd = np.array(dd)

dr = rr - ra
dd = dd - dec
cosdec = np.cos(np.deg2rad(dec))

dr = 3600. * dr * cosdec
dd = 3600 * dd

tdr = dr[-len(tt):]
tdd = dd[-len(tt):]
dr = dr[:-len(tt)]
dd = dd[:-len(tt)]

plt.clf()
plt.plot(dr, dd, 'k-')
plt.plot(tdr, tdd, 'ro')
for r,d,t in zip(tdr, tdd, tt):
    #plt.text(r, d, t, fontsize=8, color='k', rotation=60., ha='left', va='center')
    plt.text(r, d, t, fontsize=8, color='k', rotation=90., ha='center', va='bottom')
plt.axis('equal')
plt.xlabel('dRA (arcsec)')
plt.ylabel('dDec (arcsec)')
plt.title('Gaia source_id %s, RA,Dec (%.4f, %.4f)\nPM (%.1f, %.1f) mas/yr, Parallax %.1f mas' %
          (gsrc.source_id, gsrc.ra, gsrc.dec, gsrc.pmra, gsrc.pmdec, gsrc.parallax))
plt.savefig('rd.png')



ref_epoch = 2015.0
ref_mjd = (float(ref_epoch) - 2000.) * TAITime.daysperyear + TAITime.mjd2k
print('Ref MJD:', ref_mjd)
ref_tai = TAITime(None, mjd=ref_mjd)

for testcase, ra,dec in [(1, 0.,0.),
                     (2, 0.,45.),
                     (3, 45.,45.),
                     (4, 90.,45.),]:
    pos1 = GaiaPosition(ra, dec, ref_tai.getValue(), 0., 0., 1000.)
    pos2 = GaiaPosition(ra, dec, ref_tai.getValue(), 0., 0.,  500.)
    #fakewcs1 = Tan(0., 0., W/2., H/2., -cd, 0., 0., cd, float(W), float(H))
    
    MJD = np.linspace(ref_mjd, ref_mjd + 365.25, 200)
    rr1,dd1 = np.zeros(len(MJD)), np.zeros(len(MJD))
    rr2,dd2 = np.zeros(len(MJD)), np.zeros(len(MJD))
    for i,mjd in enumerate(MJD):
        #wcs = LegacySurveyWcs(fakewcs1, TAITime(None, mjd=mjd))
        #rd = wcs.positionToPixel(pos1)
        rd = pos1.getPositionAtTime(TAITime(None, mjd=mjd))
        rr1[i] = rd.ra
        dd1[i] = rd.dec
        rd = pos2.getPositionAtTime(TAITime(None, mjd=mjd))
        rr2[i] = rd.ra
        dd2[i] = rd.dec
    
    # wrap
    rr1 += -360.*(rr1 > 180)
    rr2 += -360.*(rr2 > 180)
    
    cosdec = np.cos(np.deg2rad(dec))
    dr1 = 3600. * (rr1 - ra) * cosdec
    dd1 = 3600. * (dd1 - dec)
    dr2 = 3600. * (rr2 - ra) * cosdec
    dd2 = 3600. * (dd2 - dec)
    
    plt.clf()
    plt.plot(dr1, dd1, 'k-')
    plt.plot(dr2, dd2, 'r-')
    plt.plot(dr1[0], dd1[0], 'ko')
    # plt.plot(tdr, tdd, 'ro')
    # for r,d,t in zip(tdr, tdd, tt):
    #     #plt.text(r, d, t, fontsize=8, color='k', rotation=60., ha='left', va='center')
    #     plt.text(r, d, t, fontsize=8, color='k', rotation=90., ha='center', va='bottom')
    plt.axis('equal')
    plt.xlabel('dRA (arcsec)')
    plt.ylabel('dDec (arcsec)')
    plt.title('Test sources: RA,Dec %g,%g, pm 0,0, parallax 1 / 0.5 arcsec' % (ra,dec))
    plt.savefig('rd%i.png' % testcase)


for testcase, ra,dec,pmra,pmdec in [(10, 0.,0., 1000., 0.),
                                    (11, 0.,0., 0., 1000.),
                                    (12, 45.,45., 1000., 0.),]:
    pos1 = GaiaPosition(ra, dec, ref_tai.getValue(), pmra, pmdec, 0.)
    pos2 = GaiaPosition(ra, dec, ref_tai.getValue(), pmra, pmdec, 1000.)
    
    MJD = np.linspace(ref_mjd, ref_mjd + 365.25*5, 200)
    rr1,dd1 = np.zeros(len(MJD)), np.zeros(len(MJD))
    rr2,dd2 = np.zeros(len(MJD)), np.zeros(len(MJD))
    for i,mjd in enumerate(MJD):
        rd = pos1.getPositionAtTime(TAITime(None, mjd=mjd))
        rr1[i] = rd.ra
        dd1[i] = rd.dec
        rd = pos2.getPositionAtTime(TAITime(None, mjd=mjd))
        rr2[i] = rd.ra
        dd2[i] = rd.dec
    
    # wrap
    rr1 += -360.*(rr1 > 180)
    rr2 += -360.*(rr2 > 180)
    
    cosdec = np.cos(np.deg2rad(dec))
    dr1 = 3600. * (rr1 - ra) * cosdec
    dd1 = 3600. * (dd1 - dec)
    dr2 = 3600. * (rr2 - ra) * cosdec
    dd2 = 3600. * (dd2 - dec)
    
    plt.clf()
    plt.plot(dr1, dd1, 'k-')
    plt.plot(dr2, dd2, 'r-')
    plt.plot(dr1[0], dd1[0], 'ko')
    # plt.plot(tdr, tdd, 'ro')
    # for r,d,t in zip(tdr, tdd, tt):
    #     #plt.text(r, d, t, fontsize=8, color='k', rotation=60., ha='left', va='center')
    #     plt.text(r, d, t, fontsize=8, color='k', rotation=90., ha='center', va='bottom')
    plt.axis('equal')
    plt.xlabel('dRA (arcsec)')
    plt.ylabel('dDec (arcsec)')
    plt.title('Test sources: RA,Dec %g,%g, pm (%g, %g)  parallax 0 / 1 arcsec' % (ra,dec, pmra,pmdec))
    plt.savefig('rd%i.png' % testcase)





