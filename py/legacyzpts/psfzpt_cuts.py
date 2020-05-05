from __future__ import print_function
import numpy as np
import fitsio
from astrometry.util.fits import fits_table, merge_tables
from collections import Counter

def psf_cuts_to_string(ccd_cuts, join=', '):
    s = []
    for k,v in CCD_CUT_BITS.items():
        if ccd_cuts & v:
            s.append(k)
    return join.join(s)

# Bit codes for why a CCD got cut, used in cut_ccds().
CCD_CUT_BITS= dict(
    err_legacyzpts = 0x1,
    not_grz = 0x2,
    not_third_pix = 0x4, # Mosaic3 one-third-pixel interpolation problem
    exptime = 0x8,
    ccdnmatch = 0x10,
    zpt_diff_avg = 0x20,
    zpt_small = 0x40,
    zpt_large = 0x80,
    sky_is_bright = 0x100,
    badexp_file = 0x200,
    phrms = 0x400,
    radecrms = 0x800,
    seeing_bad = 0x1000,
    early_decam = 0x2000,
    depth_cut = 0x4000,
    too_many_bad_ccds = 0x8000,
    flagged_in_des = 0x10000,
    phrms_s7 = 0x20000,
)

MJD_EARLY_DECAM = 56730.

# DECam CCD name to number mapping.
ccdnamenumdict = {'S1': 25, 'S2': 26, 'S3': 27, 'S4':28,
                  'S5': 29, 'S6': 30, 'S7': 31,
                  'S8': 19, 'S9': 20, 'S10': 21, 'S11': 22, 'S12': 23,
                  'S13': 24,
                  'S14': 13, 'S15': 14, 'S16': 15, 'S17': 16, 'S18': 17,
                  'S19': 18,
                  'S20': 8, 'S21': 9, 'S22': 10, 'S23': 11, 'S24': 12,
                  'S25': 4, 'S26': 5, 'S27': 6, 'S28': 7,
                  'S29': 1, 'S30': 2, 'S31': 3,
                  'N1': 32, 'N2': 33, 'N3': 34, 'N4': 35,
                  'N5': 36, 'N6': 37, 'N7': 38,
                  'N8': 39, 'N9': 40, 'N10': 41, 'N11': 42, 'N12': 43,
                  'N13': 44,
                  'N14': 45, 'N15': 46, 'N16': 47, 'N17': 48, 'N18': 49,
                  'N19': 50,
                  'N20': 51, 'N21': 52, 'N22': 53, 'N23': 54, 'N24': 55,
                  'N25': 56, 'N26': 57, 'N27': 58, 'N28': 59,
                  'N29': 60, 'N30': 61, 'N31': 62,
                  }

def detrend_zeropoints(P, airmass_terms, mjd_terms):
    '''
    Correct zeropoints for trends with airmass and MJD
    before making too-big/too-small cuts.

    *airmass_terms*: list of (band, airmass) coeffs.
    *mjd_terms*: list of (band, zpt0, [ (terms...)]) coeffs.
    '''
    zpt_corr = P.ccdzpt.copy()
    ntot = 0
    for band,k in airmass_terms:
        I = np.flatnonzero((P.filter == band) * (P.airmass >= 1.0))
        if len(I) == 0:
            continue
        ntot += len(I)
        zpt_corr[I] += k * (P.airmass[I] - 1.0)

    if ntot < len(P):
        print('In detrend_zeropoints: did not detrend for airmass variation for', len(P)-ntot, 'CCDs due to unknown band or bad airmass')

    ntot = 0
    mjd0 = 56658.5
    for band,zpt0,terms in mjd_terms:
        I = np.flatnonzero((P.filter == band) * (P.mjd_obs > 0))
        if len(I) == 0:
            continue
        day = P.mjd_obs[I] - mjd0
        # Piecewise linear function
        for day_i, day_f, zpt_i, zpt_f, c0, c1 in terms:
            c1 = (zpt_f - zpt_i) / (day_f - day_i)
            Jday = (day >= day_i) * (day < day_f)
            J = I[Jday]
            if len(J) == 0:
                continue
            ntot += len(J)
            zpt_corr[J] += zpt0 - (c0 + c1*day[Jday])
    if ntot < len(P):
        print('In detrend_zeropoints: did not detrend for temporal variation for', len(P)-ntot, 'CCDs due to unknown band or MJD_OBS')

    # Zeros stay zero!
    zpt_corr[P.ccdzpt == 0] = 0

    return zpt_corr

def detrend_decam_zeropoints(P):
    '''
    Per Arjun's email 2019-02-27 "Zeropoint variations with MJD for
    DECam data".
    '''
    airmass_terms = [
        ('g', 0.173),
        ('r', 0.090),
        ('i', 0.054),
        ('z', 0.060),
        ('Y', 0.058)]

    mjd_terms = [
        ('g', 25.08, [
            (   0.0,  160.0, 25.170, 25.130, 25.170,  -2.5001e-04),
            ( 160.0,  480.0, 25.180, 25.080, 25.230,  -3.1250e-04),
            ( 480.0,  810.0, 25.080, 25.080, 25.080,   0.0000e+00),
            ( 810.0,  950.0, 25.130, 25.130, 25.130,   0.0000e+00),
            ( 950.0, 1250.0, 25.130, 25.040, 25.415,  -2.9999e-04),
            (1250.0, 1650.0, 25.080, 25.000, 25.330,  -2.0000e-04),
            (1650.0, 1900.0, 25.270, 25.210, 25.666,  -2.4001e-04),]),
        ('r', 25.29, [
            (   0.0,  160.0, 25.340, 25.340, 25.340,   0.0000e+00),
            ( 160.0,  480.0, 25.370, 25.300, 25.405,  -2.1876e-04),
            ( 480.0,  810.0, 25.300, 25.280, 25.329,  -6.0602e-05),
            ( 810.0,  950.0, 25.350, 25.350, 25.350,   0.0000e+00),
            ( 950.0, 1250.0, 25.350, 25.260, 25.635,  -3.0000e-04),
            (1250.0, 1650.0, 25.320, 25.240, 25.570,  -2.0000e-04),
            (1650.0, 1900.0, 25.440, 25.380, 25.836,  -2.4001e-04),]),
        ('i', 25.26, []),
        ('z', 24.92, [
            (   0.0,  160.0, 24.970, 24.970, 24.970,   0.0000e+00),
            ( 160.0,  480.0, 25.030, 24.950, 25.070,  -2.5000e-04),
            ( 480.0,  760.0, 24.970, 24.900, 25.090,  -2.5000e-04),
            ( 760.0,  950.0, 24.900, 25.030, 24.380,   6.8422e-04),
            ( 950.0, 1150.0, 25.030, 24.880, 25.743,  -7.5001e-04),
            (1150.0, 1270.0, 24.880, 25.030, 23.442,   1.2500e-03),
            (1270.0, 1650.0, 25.030, 24.890, 25.498,  -3.6842e-04),
            (1650.0, 1900.0, 25.070, 24.940, 25.928,  -5.2000e-04),]),
        ('Y', 23.87, []),
    ]

    return detrend_zeropoints(P, airmass_terms, mjd_terms)

def detrend_mzlsbass_zeropoints(P):

    airmass_terms = [
        ('g', 0.291),
        ('r', 0.176),
        ('z', 0.165),
        ]

    mjd_terms = [
        ('g', 25.74, [
            (   0.0,  720.0, 25.900, 25.900,  25.900,  0.0000e+00),
            ( 720.0,  810.0, 25.900, 25.750,  27.100, -1.6667e-03),
            ( 810.0,  900.0, 25.880, 25.780,  26.780, -1.1111e-03),
            ( 900.0,  950.0, 25.780, 25.920,  23.260,  2.8000e-03),
            ( 950.0, 1100.0, 25.950, 25.950,  25.950,  0.0000e+00),
            (1100.0, 1255.0, 25.950, 25.750,  27.369, -1.2903e-03),
            (1255.0, 1280.0, 25.850, 25.400,  48.440, -1.8000e-02),
            (1280.0, 1500.0, 25.880, 25.800,  26.345, -3.6364e-04),
            (1500.0, 1520.0, 25.800, 25.880,  19.800,  4.0000e-03),
            (1520.0, 1550.0, 25.750, 25.900,  18.150,  5.0000e-03),
            (1550.0, 1580.0, 25.700, 25.850,  17.950,  5.0000e-03),
            (1580.0, 1600.0, 25.850, 25.800,  29.800, -2.5001e-03),
            (1600.0, 1615.0, 25.800, 25.800,  25.800,  0.0000e+00),
            (1615.0, 1621.0, 25.800, 25.700,  52.716, -1.6666e-02),
            (1621.0, 1626.0, 25.700, 25.850, -22.930,  3.0000e-02),
            (1626.0, 1645.0, 25.830, 25.790,  29.253, -2.1052e-03),
            (1645.0, 1658.0, 25.800, 25.600,  51.108, -1.5385e-02),
            (1658.0, 1668.0, 25.600, 25.850, -15.850,  2.5000e-02),
            ]),
        ('r', 25.52, [
            (   0.0,  720.0, 25.600, 25.600, 25.600,  0.0000e+00),
            ( 720.0,  815.0, 25.600, 25.500, 26.358, -1.0526e-03),
            ( 815.0,  882.0, 25.600, 25.600, 25.600,  0.0000e+00),
            ( 882.0,  930.0, 25.450, 25.600, 22.694,  3.1250e-03),
            ( 930.0, 1100.0, 25.680, 25.680, 25.680,  0.0000e+00),
            (1100.0, 1220.0, 25.600, 25.540, 26.150, -5.0000e-04),
            (1220.0, 1280.0, 25.550, 25.380, 29.007, -2.8333e-03),
            (1280.0, 1420.0, 25.500, 25.500, 25.500,  0.0000e+00),
            (1420.0, 1450.0, 25.650, 25.700, 23.283,  1.6667e-03),
            (1450.0, 1550.0, 25.550, 25.550, 25.550,  0.0000e+00),
            (1550.0, 1610.0, 25.500, 25.600, 22.917,  1.6667e-03),
            (1610.0, 1635.0, 25.450, 25.600, 15.790,  6.0000e-03),
            (1635.0, 1670.0, 25.550, 25.400, 32.557, -4.2857e-03),
            ]),
        ('z', 26.20, [
            (   0.0,  720.0, 26.200, 26.200, 26.200,  0.0000e+00),
            ( 720.0,  920.0, 26.550, 26.050, 28.350, -2.5000e-03),
            ( 920.0, 1030.0, 26.150, 26.150, 26.150,  0.0000e+00),
            (1030.0, 1070.0, 26.500, 26.500, 26.500,  0.0000e+00),
            (1070.0, 1115.0, 26.500, 26.350, 30.067, -3.3333e-03),
            (1115.0, 1300.0, 26.470, 26.330, 27.314, -7.5675e-04),
            (1300.0, 1355.0, 26.250, 26.200, 27.432, -9.0908e-04),
            (1355.0, 1500.0, 26.350, 26.350, 26.350,  0.0000e+00),
            ]),
    ]

    return detrend_zeropoints(P, airmass_terms, mjd_terms)

def psf_zeropoint_cuts(P, pixscale,
                       zpt_cut_lo, zpt_cut_hi, bad_expid, camera,
                       radec_rms, skybright, zpt_diff_avg, image2coadd=''):
    '''
    zpt_cut_lo, zpt_cut_hi: dict from band to zeropoint.
    '''

    ## PSF zeropoints cuts

    P.ccd_cuts = np.zeros(len(P), np.int32)

    seeing = np.isfinite(P.fwhm) * P.fwhm * pixscale
    P.zpt[np.logical_not(np.isfinite(P.zpt))] = 0.
    P.ccdzpt[np.logical_not(np.isfinite(P.ccdzpt))] = 0.
    P.ccdphrms[np.logical_not(np.isfinite(P.ccdphrms))] = 1.
    P.ccdrarms[np.logical_not(np.isfinite(P.ccdrarms))] = 1.
    P.ccddecrms[np.logical_not(np.isfinite(P.ccddecrms))] = 1.

    keys = zpt_cut_lo.keys()

    if camera == 'decam':
        ccdzpt = detrend_decam_zeropoints(P)
    else:
        ccdzpt = detrend_mzlsbass_zeropoints(P)
    ccdname = np.array([n.strip() for n in P.ccdname])

    cuts = [
        ('not_grz',   np.array([f.strip() not in 'grz' for f in P.filter])),
        ('ccdnmatch', P.ccdnphotom < 20),
        ('zpt_small', np.array([zpt < zpt_cut_lo.get(f.strip(),0) for f,zpt in zip(P.filter, ccdzpt)])),
        ('zpt_large', np.array([zpt > zpt_cut_hi.get(f.strip(),0) for f,zpt in zip(P.filter, ccdzpt)])),
        ('phrms',     P.phrms > 0.1),
        ('exptime', P.exptime < 30),
        ('seeing_bad', np.logical_not(np.logical_and(seeing > 0, seeing < 3.0))),
        ('badexp_file', np.array([expnum in bad_expid for expnum in P.expnum])),
        ('radecrms',  np.hypot(P.ccdrarms, P.ccddecrms) > radec_rms),
        ('sky_is_bright', np.array([sky > skybright.get(f.strip(), 1e6) for f,sky in zip(P.filter, P.ccdskycounts)])),
        ('zpt_diff_avg', np.abs(P.ccdzpt - P.zpt) > zpt_diff_avg),
        ('phrms_s7', (P.ccdphrms > 0.1) & (ccdname == 'S7')),
    ]

    if camera == 'mosaic':
        cuts.append(('not_third_pix', (np.logical_not(P.yshift) * (P.mjd_obs < 57674.))))

    if camera == 'decam':
        if image2coadd != '':
            cuts.append(('flagged_in_des', not_in_image2coadd(P, image2coadd)))
        else:
            print('Removing all early DECam data')
            cuts.append(('early_decam', P.mjd_obs < MJD_EARLY_DECAM))

    for name,cut in cuts:
        P.ccd_cuts += CCD_CUT_BITS[name] * cut
        print(np.count_nonzero(cut), 'CCDs cut by', name)

def not_in_image2coadd(P, image2coadd):
    image2coadd = fits_table(image2coadd)
    ccdid = (P.expnum * 100 +
             np.array([ccdnamenumdict[c.strip()] for c in P.ccdname]))
    ccdidi2c = image2coadd.expnum * 100 + image2coadd.ccdnum
    s = np.argsort(ccdidi2c)
    ind = np.searchsorted(ccdidi2c[s], ccdid)
    match = (ind >= 0) & (ind < len(ccdidi2c))
    match[match] = ccdidi2c[s[ind[match]]] == ccdid[match]
    desy1mjd = 57432
    # max MJD in image2coadd.fits is ~57431.3
    print('Flagging images not in DES image2coadd.fits before MJD %d'
          % desy1mjd)
    mindesy1 = (P.propid == '2012B-0001') & (P.mjd_obs < desy1mjd)
    return mindesy1 & ~match


def add_psfzpt_cuts(T, camera, bad_expid, image2coadd=''):
    from legacyzpts.legacy_zeropoints import get_pixscale
    pixscale = get_pixscale(camera)

    if camera == 'mosaic':
        # Arjun: 2019-03-15
        z0 = 26.20
        dz = (-1.0, 0.8)
        radec_rms = 0.1
        skybright = dict(z=200.)
        zpt_diff_avg = 0.1
        zpt_lo = dict(z=z0+dz[0])
        zpt_hi = dict(z=z0+dz[1])
        psf_zeropoint_cuts(T, pixscale, zpt_lo, zpt_hi, bad_expid, camera, radec_rms,
                           skybright, zpt_diff_avg)

    elif camera == '90prime':
        g0 = 25.74
        r0 = 25.52
        dg = (-0.5, 0.18)
        dr = (-0.5, 0.18)
        radec_rms = 0.2
        skybright = {}
        zpt_diff_avg = 0.1
        zpt_lo = dict(g=g0+dg[0], r=r0+dr[0])
        zpt_hi = dict(g=g0+dg[1], r=r0+dr[1])
        psf_zeropoint_cuts(T, pixscale, zpt_lo, zpt_hi, bad_expid, camera, radec_rms,
                           skybright, zpt_diff_avg)

    elif camera == 'decam':
        # These are from DR5; eg
        # https://github.com/legacysurvey/legacypipe/blob/dr5.0/py/legacypipe/decam.py#L50
        g0 = 25.08
        r0 = 25.29
        i0 = 25.26
        z0 = 24.92
        Y0 = 23.87
        dg = (-0.5, 0.25)
        di = (-0.5, 0.25)
        dr = (-0.5, 0.25)
        dz = (-0.5, 0.25)
        dY = (-0.5, 0.25)
        radec_rms = 0.4
        skybright = dict(g=90., r=150., z=180.)
        zpt_diff_avg = 0.25
        zpt_lo = dict(g=g0+dg[0], r=r0+dr[0], z=z0+dz[0], i=i0+di[0],
                      Y=Y0+dY[0])
        zpt_hi = dict(g=g0+dg[1], r=r0+dr[1], z=z0+dz[1], i=i0+di[1],
                      Y=Y0+dY[1])
        psf_zeropoint_cuts(T, pixscale, zpt_lo, zpt_hi, bad_expid, camera, radec_rms,
                           skybright, zpt_diff_avg, image2coadd=image2coadd)
    else:
        assert(False)

def read_bad_expid(fn='bad_expid.txt'):
    bad_expid = {}
    f = open(fn)
    for line in f.readlines():
        #print(line)
        if len(line) == 0:
            continue
        if line[0] == '#':
            continue
        words = line.split()
        if len(words) < 2:
            continue
        try:
            expnum = int(words[0], 10)
        except:
            print('Skipping line', line)
            continue
        reason = ' '.join(words[1:])
        bad_expid[expnum] = reason
    return bad_expid

if __name__ == '__main__':
    import sys
    from pkg_resources import resource_filename
    import pylab as plt

    # MzLS, BASS DR8b updates
    T = fits_table('/global/project/projectdirs/cosmo/work/legacysurvey/dr8b/runbrick-90prime-mosaic/survey-ccds-dr8b-90prime-mosaic-nocuts.kd.fits')

    from collections import Counter
    print('Cameras:', Counter(T.camera))

    camera = 'mosaic'
    fn = resource_filename('legacyzpts', 'data/{}-bad_expid.txt'.format(camera))
    print('Reading', fn)
    bad_expid = read_bad_expid(fn)

    I, = np.nonzero([cam.strip() == camera for cam in T.camera])
    Tm = T[I]
    add_psfzpt_cuts(Tm, camera, bad_expid)

    camera = '90prime'
    ## NO BAD_EXPID!
    I, = np.nonzero([cam.strip() == camera for cam in T.camera])
    Tb = T[I]
    add_psfzpt_cuts(Tb, camera, [])

    T = merge_tables([Tm, Tb])
    T.writeto('/tmp/survey-ccds-updated.fits')

    g0 = 25.74
    r0 = 25.52
    z0 = 26.20
    dg = (-0.5, 0.18)
    dr = (-0.5, 0.18)
    dz = (-0.8, 0.8)
    zpt_lo = dict(g=g0+dg[0], r=r0+dr[0], z=z0+dz[0])
    zpt_hi = dict(g=g0+dg[1], r=r0+dr[1], z=z0+dz[1])
    for band in ['g','r','z']:
        I, = np.nonzero([f[0] == band for f in T.filter])
        detrend = detrend_mzlsbass_zeropoints(T[I])
        from astrometry.util.plotutils import *
        plt.clf()
        plt.subplot(2,1,1)
        ylo,yhi = zpt_lo[band], zpt_hi[band]
        ha = dict(doclf=False, docolorbar=False, nbins=200,
                  range=((T.mjd_obs.min(), T.mjd_obs.max()),
                         (ylo-0.01, yhi+0.01)))
        loghist(T.mjd_obs[I], np.clip(T.ccdzpt[I], ylo, yhi), **ha)
        plt.title('Original zpt %s' % band)
        plt.subplot(2,1,2)
        loghist(T.mjd_obs[I], np.clip(detrend, ylo, yhi), **ha)
        plt.title('Detrended')
        plt.savefig('detrend-h2-%s.png' % band)
    sys.exit(0)

    # DECam updated for DR8, post detrend_decam_zeropoints.
    camera = 'decam'
    fn = resource_filename('legacyzpts', 'data/{}-bad_expid.txt'.format(camera))
    if os.path.isfile(fn):
        print('Reading {}'.format(fn))
        bad_expid = read_bad_expid(fn)
    else:
        print('No bad exposure file for camera {}'.format(camera))
        raise IOError

    # T = fits_table()
    # T.mjd_obs = np.arange(56658, 58500)
    # T.airmass = np.ones(len(T))
    # T.ccdzpt = np.zeros(len(T)) + 25.0
    #
    # T.filter = np.array(['g'] * len(T))
    # corr = detrend_decam_zeropoints(T)
    #
    # plt.clf()
    # plt.plot(T.mjd_obs, corr, 'g.')
    #
    # T.filter = np.array(['r'] * len(T))
    # corr = detrend_decam_zeropoints(T)
    #
    # plt.plot(T.mjd_obs, corr, 'r.')
    #
    # T.filter = np.array(['z'] * len(T))
    # corr = detrend_decam_zeropoints(T)
    #
    # plt.plot(T.mjd_obs, corr, 'm.')
    # plt.savefig('corr.png')
    # sys.exit(0)


    g0 = 25.08
    r0 = 25.29
    i0 = 25.26
    z0 = 24.92
    dg = (-0.5, 0.25)
    di = (-0.5, 0.25)
    dr = (-0.5, 0.25)
    dz = (-0.5, 0.25)
    zpt_lo = dict(g=g0+dg[0], r=r0+dr[0], i=i0+dr[0], z=z0+dz[0])
    zpt_hi = dict(g=g0+dg[1], r=r0+dr[1], i=i0+dr[1], z=z0+dz[1])

    TT = []
    for band in ['g','r','z']:
        infn = '/global/project/projectdirs/cosmo/work/legacysurvey/dr8/DECaLS/survey-ccds-decam-%s.fits.gz' % band

        T = fits_table(infn)
        print('Read', len(T), 'CCDs for', band)
        print('Initial:', np.sum(T.ccd_cuts == 0), 'CCDs pass cuts')

        ylo,yhi = zpt_lo[band], zpt_hi[band]

        plt.figure(figsize=(8,8))

        plt.clf()
        plt.subplot(2,1,1)
        plt.plot(T.airmass, T.ccdzpt, 'b.', alpha=0.01)
        plt.ylim(ylo,yhi)
        plt.subplot(2,1,2)
        mjd = T.mjd_obs
        T.mjd_obs = np.zeros(len(T))
        detrend = detrend_decam_zeropoints(T)
        plt.plot(T.airmass, detrend, 'b.', alpha=0.01)
        plt.ylim(ylo,yhi)
        plt.savefig('airmass-%s.png' % band)

        T.mjd_obs = mjd

        plt.clf()
        detrend = detrend_decam_zeropoints(T)
        plt.subplot(2,1,1)
        plt.plot(T.mjd_obs, np.clip(T.ccdzpt, ylo, yhi), 'b.', alpha=0.02)
        plt.ylim(ylo-0.01, yhi+0.01)
        plt.title('Original zpt')
        plt.subplot(2,1,2)
        plt.plot(T.mjd_obs, np.clip(detrend, ylo, yhi), 'b.', alpha=0.02)
        plt.ylim(ylo-0.01, yhi+0.01)
        plt.title('Detrended')
        plt.savefig('detrend-%s.png' % band)

        plt.clf()
        plt.subplot(2,1,1)
        ha = dict(bins=100, range=(ylo-0.01, yhi+0.01))
        plt.hist(np.clip(T.ccdzpt, ylo, yhi), **ha)
        plt.xlim(ylo-0.01, yhi+0.01)
        plt.ylim(0, 100000)
        plt.title('Original zpt')
        plt.subplot(2,1,2)
        plt.hist(np.clip(detrend, ylo, yhi), **ha)
        plt.xlim(ylo-0.01, yhi+0.01)
        plt.ylim(0, 100000)
        plt.title('Detrended')
        plt.savefig('detrend-h-%s.png' % band)

        from astrometry.util.plotutils import *
        plt.clf()
        plt.subplot(2,1,1)
        ha = dict(doclf=False, docolorbar=False, nbins=200,
                  range=((T.mjd_obs.min(), T.mjd_obs.max()),
                         (ylo-0.01, yhi+0.01)))
        loghist(T.mjd_obs, np.clip(T.ccdzpt, ylo, yhi), **ha)
        plt.title('Original zpt %s' % band)
        plt.subplot(2,1,2)
        loghist(T.mjd_obs, np.clip(detrend, ylo, yhi), **ha)
        plt.title('Detrended')
        plt.savefig('detrend-h2-%s.png' % band)

        psf_zeropoint_cuts(T, 0.262, zpt_lo, zpt_hi, bad_expid, camera)
        print('Final:', np.sum(T.ccd_cuts == 0), 'CCDs pass cuts')
        TT.append(T)

    T = merge_tables(TT)
    fn = 'survey-ccds.fits'
    T.writeto(fn)
    from legacypipe.create_kdtrees import create_kdtree
    kdfn = 'survey-ccds.kd.fits'
    create_kdtree(fn, kdfn, True)
    print('Wrote', kdfn)

    sys.exit(0)

    ################################

    g0 = 25.74
    r0 = 25.52
    z0 = 26.20

    dg = (-0.5, 0.18)
    dr = (-0.5, 0.18)
    dz = (-0.6, 0.6)

    P = fits_table('psfzpts-pre-cuts-mosaic-dr6plus5.fits')
    psf_zeropoint_cuts(P, 0.262,
                       dict(z=z0+dz[0]), dict(z=z0+dz[1]),
                       bad_expid, 'mosaic')
    P.writeto('survey-ccds-mosaic-dr6plus5.fits')
    sys.exit(0)

    P = fits_table('psfzpts-pre-cuts-mosaic-dr6plus4.fits')
    psf_zeropoint_cuts(P, ['z'], 0.262,
                       z0+dz[0], z0+dz[1], bad_expid, 'mosaic')
    P.writeto('survey-ccds-mosaic-dr6plus4.fits')
    sys.exit(0)

    P = fits_table('psfzpts-pre-cuts-mosaic-dr6plus3.fits')
    psf_zeropoint_cuts(P, ['z'], 0.262,
                       z0+dz[0], z0+dz[1], bad_expid, 'mosaic')
    P.writeto('survey-ccds-mosaic-dr6plus3.fits')
    sys.exit(0)

    P = fits_table('psfzpts-pre-cuts-mosaic-dr6plus2.fits')
    psf_zeropoint_cuts(P, ['z'], 0.262,
                       z0+dz[0], z0+dz[1], bad_expid, 'mosaic')
    P.writeto('survey-ccds-mosaic-dr6plus2.fits')
    sys.exit(0)

    P = fits_table('dr6plus.fits')
    psf_zeropoint_cuts(P, ['z'], 0.262,
                       z0+dz[0], z0+dz[1], bad_expid, 'mosaic')
    P.writeto('survey-ccds-dr6plus.fits')
    sys.exit(0)

    for X in [
            (#'apzpts/survey-ccds-90prime-legacypipe.fits.gz',
                'apzpts/survey-ccds-90prime.fits.gz',
                'survey-ccds-90prime-psfzpts.fits',
                #'90prime-psfzpts.fits',
                'g', 'BASS g', 'g', 20, 25, 26.25, 0.45,
                #25.2, 26.0,
                g0+dg[0], g0+dg[1], {}, '90prime'),
            (#'apzpts/survey-ccds-90prime-legacypipe.fits.gz',
                'apzpts/survey-ccds-90prime.fits.gz',
                'survey-ccds-90prime-psfzpts.fits',
                #'90prime-psfzpts.fits',
                'r', 'BASS r', 'r', 19.5, 24.75, 25.75, 0.45,
                #24.9, 25.7,
                r0+dr[0], r0+dr[1], {}, '90prime'),
            (#'apzpts/survey-ccds-mosaic-legacypipe.fits.gz',
                'apzpts/survey-ccds-mosaic.fits.gz',
                'survey-ccds-mosaic-psfzpts.fits',
                #'mosaic-psfzpts.fits',
                'z', 'MzLS z', 'z', 19.5, 25, 27, 0.262,
                #25.2, 26.8,
                z0+dz[0], z0+dz[1], bad_expid, 'mosaic'),
    ]:
        run(*X)
