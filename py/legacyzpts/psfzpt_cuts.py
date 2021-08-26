from __future__ import print_function
import numpy as np
from astrometry.util.fits import fits_table

from legacypipe.bits import CCD_CUTS as CCD_CUT_BITS

def psf_cuts_to_string(ccd_cuts, join=', '):
    s = []
    for k,v in CCD_CUT_BITS.items():
        if ccd_cuts & v:
            s.append(k)
    return join.join(s)

MJD_EARLY_DECAM = 56730.


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
    Per Arjun's email 2021-08-06 "Zeropoint variations with MJD for
    DECam data - DR10 edition".
    '''
    airmass_terms = [
        ('g', 0.1660),
        ('r', 0.0862),
        ('i', 0.0495),
        ('z', 0.0715),
        ('Y', 0.058)]
    '''
    the columns are:
    * 1 and 2: the lower and upper bound of the relative MJD, where zero
      corresponds to 1/1/2014 or MJD_OBS = 56658.500
    * 3 and 4: the values of the zero points defining the ends of the
      line segment (i.e., at the lower and upper bound of MJD_OBS
      specified in cols 1 and 2)
    * 5 and 6: the zero point and slope of the linear fit defining the
      segment (calculable from columns 1 through 4)
    '''
    mjd_terms = [
        ('g', 25.08, [
            ( -150.0,     0.0, 25.220, 25.200, 25.200,  -1.3332e-04 ),
            (    0.0,   227.0, 25.130, 25.150, 25.130,   8.8108e-05),
            (  227.0,   400.0, 25.150, 25.130, 25.176,  -1.1561e-04),
            (  400.0,   500.0, 25.050, 25.120, 24.770,   7.0002e-04),
            (  500.0,   795.0, 25.090, 25.070, 25.124,  -6.7798e-05),
            (  795.0,   930.0, 25.120, 25.130, 25.061,   7.4062e-05),
            (  930.0,   970.0, 25.130, 25.130, 25.130,   0.0000e+00),
            (  970.0,  1110.0, 25.120, 25.070, 25.466,  -3.5715e-04),
            ( 1110.0,  1240.0, 24.990, 25.080, 24.222,   6.9231e-04),
            ( 1240.0,  1462.0, 25.080, 25.030, 25.359,  -2.2522e-04),
            ( 1462.0,  1525.0, 25.030, 24.890, 28.279,  -2.2222e-03),
            ( 1525.0,  1650.0, 24.920, 25.030, 23.578,   8.8000e-04),
            ( 1650.0,  1897.0, 25.250, 25.180, 25.718,  -2.8340e-04),
            ( 1897.0,  2040.0, 25.210, 25.260, 24.547,   3.4966e-04),
            ( 2040.0,  2400.0, 25.250, 25.150, 25.817,  -2.7778e-04),
            ( 2400.0,  2700.0, 25.250, 25.120, 26.290,  -4.3333e-04),
            ]),
        ('r', 25.29, [
            ( -150.0,     0.0, 25.380, 25.330, 25.330,  -3.3333e-04),
            (    0.0,   227.0, 25.310, 25.330, 25.310,   8.8108e-05),
            (  227.0,   400.0, 25.330, 25.310, 25.356,  -1.1561e-04),
            (  400.0,   500.0, 25.230, 25.310, 24.910,   8.0000e-04),
            (  500.0,   795.0, 25.270, 25.250, 25.304,  -6.7798e-05),
            (  795.0,   930.0, 25.320, 25.320, 25.320,   0.0000e+00),
            (  930.0,   970.0, 25.320, 25.320, 25.320,   0.0000e+00),
            (  970.0,  1110.0, 25.320, 25.270, 25.666,  -3.5714e-04),
            ( 1110.0,  1240.0, 25.200, 25.280, 24.517,   6.1538e-04),
            ( 1240.0,  1462.0, 25.290, 25.270, 25.402,  -9.0092e-05),
            ( 1462.0,  1525.0, 25.240, 25.170, 26.864,  -1.1111e-03),
            ( 1525.0,  1650.0, 25.170, 25.240, 24.316,   5.6000e-04),
            ( 1650.0,  1897.0, 25.420, 25.310, 26.155,  -4.4535e-04),
            ( 1897.0,  2040.0, 25.350, 25.400, 24.687,   3.4965e-04),
            ( 2040.0,  2400.0, 25.400, 25.300, 25.967,  -2.7778e-04),
            ( 2400.0,  2700.0, 25.370, 25.290, 26.010,  -2.6667e-04),
            ]),
        ('i', 25.26, [
            (2040.0,  2400.0, 25.350, 25.210, 26.143,  -3.8889e-04),
            (2400.0,  2700.0, 25.320, 25.240, 25.960,  -2.6667e-04),
            ]),
        ('z', 24.92, [
            (-150.0,     0.0, 25.120, 25.020, 25.020,  -6.6667e-04),
            (   0.0,   227.0, 24.950, 25.040, 24.950,   3.9648e-04),
            ( 227.0,   400.0, 25.040, 24.970, 25.132,  -4.0463e-04),
            ( 400.0,   500.0, 24.940, 25.030, 24.580,   9.0000e-04),
            ( 500.0,   795.0, 25.000, 24.980, 25.034,  -6.7798e-05),
            ( 795.0,   930.0, 24.970, 25.020, 24.676,   3.7038e-04),
            ( 930.0,   970.0, 25.020, 25.060, 24.090,   9.9998e-04),
            ( 970.0,  1110.0, 25.060, 24.980, 25.614,  -5.7143e-04),
            (1110.0,  1240.0, 24.930, 25.000, 24.332,   5.3846e-04),
            (1240.0,  1462.0, 25.060, 24.970, 25.563,  -4.0541e-04),
            (1462.0,  1525.0, 24.970, 24.850, 27.755,  -1.9047e-03),
            (1525.0,  1650.0, 24.910, 24.960, 24.300,   3.9999e-04),
            (1650.0,  1897.0, 25.130, 24.960, 26.266,  -6.8826e-04),
            (1897.0,  2040.0, 24.980, 25.120, 23.123,   9.7903e-04),
            (2040.0,  2400.0, 25.120, 24.950, 26.083,  -4.7222e-04),
            (2400.0,  2700.0, 25.050, 24.950, 25.850,  -3.3333e-04),
            ]),
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

    if camera == 'decam':
        ccdzpt = detrend_decam_zeropoints(P)
    else:
        ccdzpt = detrend_mzlsbass_zeropoints(P)
    ccdname = np.array([n.strip() for n in P.ccdname])

    skybr = np.zeros(len(P), bool)
    for i,(f, sky, exptime) in enumerate(zip(P.filter, P.ccdskycounts, P.exptime)):
        if not np.isfinite(sky):
            skybr[i] = True
            continue
        skybr[i] = (sky > skybright.get(f.strip(), 1e6)) or (sky*exptime > 35000)

    cuts = [
        ('not_grz',   np.array([f.strip() not in 'grz' for f in P.filter])),
        ('ccdnmatch', P.ccdnphotom < 20),
        ('zpt_small', np.array([zpt < zpt_cut_lo.get(f.strip(),0) for f,zpt in zip(P.filter, ccdzpt)])),
        ('zpt_large', np.array([zpt > zpt_cut_hi.get(f.strip(),100) for f,zpt in zip(P.filter, ccdzpt)])),
        ('phrms',     P.phrms > 0.1),
        ('exptime',   P.exptime < 30),
        ('seeing_bad', np.logical_not(np.logical_and(seeing > 0, seeing < 3.0))),
        ('badexp_file', np.array([((expnum, None) in bad_expid or
                                   (expnum, ccdname0) in bad_expid)
                                  for expnum, ccdname0 in zip(P.expnum, ccdname)])),
        ('radecrms',  np.hypot(P.ccdrarms, P.ccddecrms) > radec_rms),
        ('sky_is_bright', skybr),
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

    if camera == 'hsc':
        # Excise chips in the outer ring.
        # FIXME - these are the bad DET-IDs, but we only have DETNAME = ccdname!
        # 4, 9, 22, 29, 70, 77, 90, 101, 102, 103, 104
        # 95 -> 0_42
        cuts.append(('flagged_in_des', np.array([n.strip() in set(['0_42']) for n in P.ccdname])))

    for name,cut in cuts:
        P.ccd_cuts += CCD_CUT_BITS[name] * cut
        print(np.count_nonzero(cut), 'CCDs cut by', name)

def not_in_image2coadd(P, image2coadd):
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
    from legacypipe.survey import LegacySurveyData
    survey = LegacySurveyData()
    imageType = survey.image_class_for_camera(camera)
    pixscale = imageType.get_nominal_pixscale()

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
        u0 = 23.20
        g0 = 25.08
        r0 = 25.29
        i0 = 25.26
        z0 = 24.92
        Y0 = 23.87
        du = (-0.5, 0.25)
        dg = (-0.5, 0.25)
        di = (-0.5, 0.25)
        dr = (-0.5, 0.25)
        dz = (-0.5, 0.25)
        dY = (-0.5, 0.25)
        radec_rms = 0.4
        skybright = dict(g=90., r=150., i=165., z=180.)
        zpt_diff_avg = 0.25
        zpt_lo = dict(g=g0+dg[0], r=r0+dr[0], z=z0+dz[0], i=i0+di[0],
                      Y=Y0+dY[0], u=u0+du[0])
        zpt_hi = dict(g=g0+dg[1], r=r0+dr[1], z=z0+dz[1], i=i0+di[1],
                      Y=Y0+dY[1], u=u0+du[1])
        psf_zeropoint_cuts(T, pixscale, zpt_lo, zpt_hi, bad_expid, camera, radec_rms,
                           skybright, zpt_diff_avg, image2coadd=image2coadd)
    elif camera == 'hsc':
        g0 = 25.0
        r0 = 25.0
        z0 = 25.0
        dg = (-5, +5)
        dr = (-5, +5)
        dz = (-5, +5)
        radec_rms = 0.2
        skybright = {}
        zpt_diff_avg = 0.1
        zpt_lo = dict(g=g0+dg[0], r=r0+dr[0], z=z0+dz[0])
        zpt_hi = dict(g=g0+dg[1], r=r0+dr[1], z=z0+dz[1])
        psf_zeropoint_cuts(T, pixscale, zpt_lo, zpt_hi, bad_expid, camera, radec_rms,
                           skybright, zpt_diff_avg)

    elif camera == 'megaprime':
        g0 = 25.0
        r0 = 26.1
        z0 = 25.0
        dg = (-5, +5)
        dr = (-0.5, +0.25)
        dz = (-5, +5)
        radec_rms = 0.2
        skybright = {}
        zpt_diff_avg = 0.1
        zpt_lo = dict(g=g0+dg[0], r=r0+dr[0], z=z0+dz[0])
        zpt_hi = dict(g=g0+dg[1], r=r0+dr[1], z=z0+dz[1])
        psf_zeropoint_cuts(T, pixscale, zpt_lo, zpt_hi, bad_expid, camera, radec_rms,
                           skybright, zpt_diff_avg)
        
    elif camera == 'panstarrs':
        g0 = 25.0
        r0 = 25.0
        z0 = 25.0
        dg = (-0.5, +0.5)
        dr = (-0.5, +0.5)
        dz = (-0.5, +0.5)
        radec_rms = 0.2
        skybright = {}
        zpt_diff_avg = 0.1
        zpt_lo = dict(g=g0+dg[0], r=r0+dr[0], z=z0+dz[0])
        zpt_hi = dict(g=g0+dg[1], r=r0+dr[1], z=z0+dz[1])
        psf_zeropoint_cuts(T, pixscale, zpt_lo, zpt_hi, bad_expid, camera, radec_rms,
                           skybright, zpt_diff_avg)
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
        if len(words) < 1:
            continue
        if '-' in words[0]:
            idparts = words[0].split('-')
            if len(idparts) != 2:
                print('Skipping line', line)
                continue
            expidstr = idparts[0]
            ccd = idparts[1].strip()
        else:
            expidstr = words[0]
            ccd = None
        try:
            expnum = int(expidstr, 10)
        except ValueError:
            print('Skipping line', line)
            continue
        reason = words[1:] if len(words) > 1 else 'unknown'
        reason = ' '.join(reason)
        bad_expid[(expnum, ccd)] = reason
    return bad_expid
