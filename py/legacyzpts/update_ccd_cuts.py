import os
import numpy
import pdb
from astrometry.util.fits import fits_table
from legacyzpts import psfzpt_cuts

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

pidlist_byobject = [
    '2013A-0741', '2013B-0440',
    '2014A-0035', '2014A-0412', '2014A-0624', '2016A-0618',
    '2015A-0397', '2015B-0314',
    '2016A-0366', '2016B-0301', '2016B-0905', '2016B-0909',
    '2017A-0388', '2017A-0916', '2017B-0110', '2017B-0906',
    '2018A-0242', '2018A-0273', '2018A-0913', '2018A-0914',
    '2018A-0386', '2019A-0272', '2019A-0305', '2019A-0910',
    '2019B-0323', '2020A-0399', '2020A-0909', '2020B-0241',
    '2019B-0371', '2019B-1014', '2020A-0908',
    '2021A-0149', '2021A-0922', '2022A-597406']

class subslices:
    "Iterator for looping over subsets of an array"
    def __init__(self, data, uind=None, **kw):
        if uind is None:
            _, self.uind = numpy.unique(data, return_index=True, **kw)
        else:
            self.uind = uind
        self.uind = numpy.sort(self.uind)
        # 1st elements.  Must convert to last elements.
        if len(self.uind) > 0:
            self.uind = numpy.sort(self.uind)
            self.uind = numpy.concatenate(
                [self.uind[1:]-1, [len(data)-1]])
        else:
            self.uind = uind.copy()
        self.ind = 0
    def __iter__(self):
        return self
    def __len__(self):
        return len(self.uind)
    def __next__(self):
        if self.ind == len(self.uind):
            raise StopIteration
        if self.ind == 0:
            first = 0
        else:
            first = self.uind[self.ind-1]+1
        last = self.uind[self.ind]+1
        self.ind += 1
        return first, last
    def next(self):
        return self.__next__()


def depthcut(survey, ccds, annotated, tilefile=None, imlist=None):
    # 1 = keep; opposite sign of ccd_cuts.
    if survey == 'decam':
        return depthcut_decam(ccds, annotated, tilefile)
    elif survey == '90prime':
        imlist = fits_table(imlist)
        ccds = repair_object_names(ccds, imlist, prefix='')
        return depthcut_90prime_alternative(ccds, annotated)
    elif survey == 'mosaic':
        return depthcut_mosaic(ccds, annotated, tilefile)
    elif survey == 'hsc':
        return numpy.ones(len(ccds), dtype='bool')
    raise ValueError('No such survey?')

def depthcut_90prime_alternative(ccds, annotated, n=6):
    s = numpy.lexsort([ccds.image_hdu, ccds.image_filename])
    ccds = ccds[s]
    sa = numpy.lexsort([annotated.image_hdu, annotated.image_filename])
    annotated = annotated[sa]
    if not numpy.all((ccds.image_hdu == annotated.image_hdu) &
                     (ccds.image_filename == annotated.image_filename)):
        raise ValueError('Inconsistent ccds & annotated ccds file?')
    keep = numpy.zeros(len(ccds), dtype='bool')

    def int_noexception(x, sentinel=-1):
        try:
            y = int(x)
        except:
            return sentinel
        return y

    tileid = numpy.array([int_noexception(o, -1) for o in ccds.object],
                         dtype='i4')

    m = (ccds.ccd_cuts == 0) & (tileid >= 0)
    keep[s[m]] = keep_deepest_ccds(
        ccds.image_filename[m], tileid[m], ccds.filter[m],
        annotated.psfdepth[m], n=n)
    return keep


def repair_object_names(ccds, imlist, prefix='90prime/'):
    ccds = ccds.copy()
    imfilename = numpy.array([prefix+f for f in imlist.filename])
    s = numpy.argsort(imfilename)
    imlist = imlist[s]
    imfilename = imfilename[s]
    cobjectname = numpy.array([o.strip() for o in ccds.object])
    iobjectname = numpy.array([o.replace("'",'').strip()
                               for o in imlist.object])
    m = (cobjectname == '')
    cfilename = numpy.array([f.strip() for f in ccds.image_filename])
    ind = numpy.searchsorted(imfilename, cfilename)
    if not numpy.all(imfilename[ind] == cfilename):
        raise AssertionError('some image file names not found in image list?')
    ccds.object[m] = iobjectname[ind[m]]
    return ccds


def depthcut_90prime(ccds, annotated, tilefile, n=6):
    return depthcut_mosaic(ccds, annotated, tilefile, n=n)


def depthcut_mosaic(ccds, annotated, tilefile, n=2):
    if len(tilefile) == 0:
        raise ValueError('Mosaic depth cut requires a tile file.')
    # original proposal was something like take only 'MzLS'.
    # 2677 OBJECT Observation(s) in tile file.
    # 361 MOSAIC_XYZ_z observations in file list.  None in tile file.
    # 2015B-2001
    s = numpy.lexsort([ccds.image_hdu, ccds.image_filename])
    ccds = ccds[s]
    sa = numpy.lexsort([annotated.image_hdu, annotated.image_filename])
    annotated = annotated[sa]
    if not numpy.all((ccds.image_hdu == annotated.image_hdu) &
                     (ccds.image_filename == annotated.image_filename)):
        raise ValueError('Inconsistent ccds & annotated ccds file?')
    keep = numpy.zeros(len(ccds), dtype='bool')
    m = ccds.ccd_cuts == 0
    keep[s[m]] = keep_deepest_tiles(ccds[m], annotated[m], tilefile, n=n)
    return keep


def keep_deepest_images(tileid, filt, depth, n=2):
    s = numpy.lexsort([depth, tileid, filt])
    _, u = numpy.unique([tileid[s], filt[s]], axis=1, return_index=True)
    u = numpy.sort(u)
    res = numpy.ones(len(tileid), dtype='bool')
    for f, l in subslices(tileid, uind=u):
        ind = s[f:l]
        if len(ind) <= n:
            continue
        res[ind[:-n]] = 0  # don't keep shallow tiles
        if ((not numpy.all(tileid[ind] == tileid[ind[0]])) or
            (numpy.max(depth[ind[:-n]]) > numpy.min(depth[ind[-n:]]))):
            pdb.set_trace()
    return res


def keep_deepest_ccds(filename, tileid, filt, depth, n=2):
    if len(filename) == 0:
        return numpy.ones(0, dtype='bool')
    s = numpy.argsort(filename)
    meddepth = {}
    for f, l in subslices(filename[s]):
        ind = s[f:l]
        meddepth[filename[ind[0]]] = numpy.median(depth[ind])
    imdepth = numpy.array([meddepth[n] for n in filename])
    _, u = numpy.unique(filename, return_index=True)
    tokeep = keep_deepest_images(tileid[u], filt[u], imdepth[u], n=n)
    tokeepdict = {filename[ind]: tokeep[i] for (i, ind) in enumerate(u)}
    return numpy.array([tokeepdict[f] for f in filename], dtype='bool')


def keep_deepest_tiles(ccds, annotated, tilefile, n=2):
    tiles = fits_table(tilefile)
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    res = numpy.zeros(len(ccds), dtype='bool')
    m = numpy.flatnonzero(ccds.ccd_cuts == 0)
    ct = SkyCoord(ra=tiles.ra*u.degree, dec=tiles.dec*u.degree)
    cc = SkyCoord(ra=ccds.ra_bore[m]*u.degree,
                  dec=ccds.dec_bore[m]*u.degree)
    mt, dct, _ = cc.match_to_catalog_sky(ct)
    tileid = tiles.tileid[mt]
    mmatch = (dct < 0.05*u.degree)
    m = m[mmatch]
    keep = keep_deepest_ccds(ccds.image_filename[m], tileid[mmatch],
                             ccds.filter[m], annotated.psfdepth[m], n=n)
    m = m[keep]
    res = numpy.zeros(len(ccds), dtype='bool')
    res[m] = 1
    return res


def keep_deepest_des(ccds, annotated):
    tileid = ccds.object
    return keep_deepest_ccds(ccds.image_filename, tileid, ccds.filter,
                             annotated.psfdepth, n=3)

def keep_deepest_delve(ccds, annotated):
    tileid = ccds.object.copy()
    for i, tname in enumerate(tileid):
        try:
            contents = tname.split()
            field, pnum, filt = contents[-1].split('-')
            pnum = min([pnum, '05'])
            newend = '-'.join([field, pnum, filt])
            newtname = ' '.join([*contents[:-1], newend])
            tileid[i] = newtname
        except:
            pass
    return keep_deepest_ccds(
        ccds.image_filename, tileid, ccds.filter, annotated.psfdepth, n=2)


def depthcut_decam(ccds, annotated, tilefile):
    if len(tilefile) == 0:
        raise ValueError('DECam depth cut requires a tile file.')
    # need ccds & annotated ccds: former has ccd_cuts, latter has psfdepth
    if not numpy.all((ccds.image_hdu == annotated.image_hdu) &
                     (ccds.image_filename == annotated.image_filename)):
        raise ValueError('Inconsistent ccds & annotated ccds file?')
    # we will only keep exposures that:
    # - have no ccd_cuts
    # - are from a set of chosen propids, with good object names within those
    # propids
    # - match a tile
    # - are among the top two deepest exposures satisfying the above.
    keep = depthcut_propid_decam(ccds) & (ccds.ccd_cuts == 0)
    mdelve = numpy.array([(('DELVE' in s) or ('PALS' in s))
                          for s in ccds.object])
    mbyobject = numpy.ones(len(ccds), dtype='bool')

    for pp in pidlist_byobject:
        mbyobject |= (ccds.propid == pp) & ~mdelve
    m = ~(mdelve | mbyobject)
    # for LS-like surveys
    keep[m & keep] = keep_deepest_tiles(
        ccds[m & keep], annotated[m & keep], tilefile, n=2)
    # for surveys where we just do object name selection
    keep[mbyobject & keep] = keep_deepest_des(
        ccds[mbyobject & keep], annotated[mbyobject & keep])
    # for delve-like surveys with too many passes
    keep[mdelve & keep] = keep_deepest_delve(ccds[mdelve & keep],
                                             annotated[mdelve & keep])
    return keep

def depthcut_propid_decam(ccds):
    # this just takes everything in certain propids, that have certain
    # survey-like object names, and trusts the surveys not to get too many
    # exposures on a single tile center.  David prefers an approach that
    # explicitly tries to limit the number of epochs on each tile center.
    mpropid = numpy.zeros(len(ccds), dtype='bool')
    # DECaLS, DECaLS+, DES, Bonaca, BLISS, DeROSITA
    pidlist = ['2014B-0404', '2016A-0190', '2012B-0001', '2015A-0616',
               '2015A-0620',
               '2017A-0260', '2017A-0388', ]
    # new DR10 propids: DeROSITA, DELVE, MAGLITES, some others
    pidlist += pidlist_byobject
    for pid in pidlist:
        mpropid = mpropid | (ccds.propid  == pid)
    # within these surveys, limit to 'normal' program, as identified
    # by object name.  This excludes, e.g., special DECaLS observations on the
    # COSMOS field, DES SN fields, ...
    lobject = numpy.array([o.lower() for o in ccds.object])
    keepstrings = [
        'DECaLS_', 'DES survey', 'DES wide', 'BLISS field',
        'DeCaLS', 'Tile ', 'DELVE field', '.0', 'BF_', 'bf',
        'CenA_', 'Scl_', 'n1f', 'n2f', 'n3f', 'n4f',
        'MAGLITES field', 'MAGLITES-II field', 'Field',
        'dark siren', 'event', 'field', 'ch', 'kp',
        'Carina', 'c1A', 'c1B', 'c2A', 'c2B', 'c3A', 'c3B',
        'c4A', 'c4B', 'Sculptor']
    mobject = numpy.zeros(len(ccds), dtype='bool')
    for x in keepstrings:
        mobject |= numpy.array([x.lower() in o
                                for o in lobject])
    def isint(x):
        try:
            _ = int(x)
        except:
            return False
        return True
    misint = numpy.array([isint(x) for x in ccds.object])
    m = mpropid & (mobject | misint)
    return m


def make_plots(ccds, camera, nside=512,
               xrange=[360, 0], yrange=[-90, 90],
               filename=None, vmin=0, vmax=12):
    import util_efs
    from matplotlib import pyplot as p
    ccdname = numpy.array([name.strip() for name in ccds.ccdname])
    if camera == 'decam':
        rad = 1.0
        ccds = ccds[ccdname == 'N4']
    elif camera == 'mosaic':
        rad = 0.33
        ccds = ccds[ccdname == 'CCD1']
    elif camera == '90prime':
        rad = 0.5
        ccds = ccds[ccdname == 'CCD1']
    else:
        raise ValueError('unrecognized camera!')
    depthbit = psfzpt_cuts.CCD_CUT_BITS['depth_cut']
    ccdcut = ccds.ccd_cuts & ~depthbit
    depthcut = (ccds.ccd_cuts & depthbit) == 0  # passes depth cut
    filts = numpy.unique(ccds.filter[ccdcut == 0])
    _, u = numpy.unique(ccds.expnum, return_index=True)
    if filename is not None:
        from matplotlib.backends.backend_pdf import PdfPages
        pdf = PdfPages(filename)
    for f in filts:
        mf = (ccds.filter[u] == f)
        maps = {'original': (mf & (ccdcut[u] == 0)),
                'depthcut': (mf & (ccdcut[u] == 0) & depthcut[u]),
                'excluded': (mf & (ccdcut[u] == 0) & ~depthcut[u])}
        for (label, m) in maps.items():
            _, ncov = util_efs.paint_map(
                ccds.ra_bore[u[m]], ccds.dec_bore[u[m]],
                numpy.ones(numpy.sum(m)), rad, nside=nside)
            p.figure(label + ' ' + f)
            p.clf()
            util_efs.imshow(ncov, xrange=xrange, yrange=yrange,
                            vmin=vmin, vmax=vmax)
            p.colorbar().set_label('n_exp')
            p.title(label + ' ' + f)
            if filename is not None:
                pdf.savefig()
    if filename is not None:
        pdf.close()


def good_ccd_fraction(survey, ccds):
    # skip this cut for some cameras (with a single chip per file)...
    if survey in ['hsc', 'wiro']:
        return 1.0

    nccdmap = {'decam': 62,
               '90prime': 4,
               'mosaic': 4,
               'megaprime': 40,
               'panstarrs': 1,
               'suprimecam': 10,
    }

    nccds = nccdmap[survey]
    ngooddict = {}
    for expnum, cut in zip(ccds.expnum, ccds.ccd_cuts):
        if cut == 0:
            ngooddict[expnum] = ngooddict.get(expnum, 0) + 1
    ngoodccds = numpy.array([ngooddict.get(e, 0) for e in ccds.expnum])
    if numpy.any(ngoodccds > nccds):
        I = numpy.flatnonzero(ngoodccds > nccds)
        raise ValueError('Some exposures have more good CCDs than should be possible: %s vs %i!' %
                         (', '.join(['%i'%i for i in ngoodccds[I]]), nccds))
    return ngoodccds/float(nccds)


def match(a, b):
    sa = numpy.argsort(a)
    ua = numpy.unique(a[sa])
    if len(ua) != len(a):
        raise ValueError('All keys in a must be unique.')
    ind = numpy.searchsorted(a[sa], b)
    m = (ind >= 0) & (ind < len(a))
    matches = a[sa[ind[m]]] == b[m]
    m[m] &= matches
    return sa[ind[m]], numpy.flatnonzero(m)


def patch_zeropoints(zps, ccds, ccdsa, decboundary=-29.25):
    if not numpy.all((ccds.image_hdu == ccdsa.image_hdu) &
                     (ccdsa.image_filename == ccdsa.image_filename)):
        raise ValueError('ccds and ccdsa must be row matched!')

    # UGH, dr10-v2 CCDs table contains some ccdname='' entries!
    ok = numpy.array([len(name.strip())>0 for name in ccds.ccdname])
    if not numpy.all(ok):
        print('Cutting to', numpy.sum(ok), 'of', len(ccds), 'CCDs with valid CCDNAME')
        ccds.cut(ok)
        ccdsa.cut(ok)

    mreplace = ccds.dec < decboundary
    oldccdzpt = ccds.ccdzpt.copy()
    ccds.zpt[mreplace] = 0
    ccds.ccdzpt[mreplace] = 0
    ccds.ccdphrms[mreplace] = 0
    olderr = numpy.seterr(invalid='ignore')
    mokim = ((zps.scatter > 0) & (zps.scatter < 0.02) &
             (numpy.abs(zps.resid) < 0.2))
    numpy.seterr(**olderr)
    zps = zps[mokim]
    mz, mc = match(zps.mjd_obs, ccds.mjd_obs)
    ccdnum = numpy.array([ccdnamenumdict[name.strip()]
                          for name in ccds.ccdname])
    newzpt = zps.zp[mz] - zps.resid[mz]
    newccdzpt = zps.zp[mz] - zps.resid[mz] - zps.mnchip[mz, ccdnum[mc]]
    newccdphrms = zps.sdchip[mz, ccdnum[mc]]
    newccdnphotom = zps.nstarchip[mz, ccdnum[mc]]
    m = (ccds.dec[mc] < decboundary)
    # S7 == ccdnum 31; removes worst-performing 5% of S7 CCDs.
    ms7 = (ccdnum[mc] != 31) | ((newccdnphotom > 3) & (newccdphrms < 0.02))
    m = m & ms7
    mz = mz[m]
    mc = mc[m]
    ccds.zpt[mc] = newzpt[m]
    ccds.ccdzpt[mc] = newccdzpt[m]
    ccds.ccdphrms[mc] = newccdphrms[m]
    ccds.phrms[mc] = zps.scatter[mz]
    ccdsa.zpt[mc] = newzpt[m]
    ccdsa.ccdzpt[mc] = newccdzpt[m]
    ccdsa.ccdphrms[mc] = newccdphrms[m]
    ccdsa.phrms[mc] = zps.scatter[mz]
    oldzp = numpy.where(oldccdzpt != 0, oldccdzpt, 22.5)
    newzp = numpy.where(ccds.ccdzpt != 0, ccds.ccdzpt, 22.5)
    oldzpscale = 10.**((oldzp-22.5)/2.5)
    newzpscale = 10.**((newzp-22.5)/2.5)
    rescale1 = oldzpscale/newzpscale
    ccds.sig1 = ccds.sig1*rescale1
    ccdsa.sig1 = ccdsa.sig1*rescale1
    rescale2 = 10.**((oldccdzpt-ccds.ccdzpt)/2.5)
    ccdsa.meansky = ccdsa.meansky*rescale2
    ccdsa.stdsky = ccdsa.stdsky*rescale2
    ccdsa.minsky = ccdsa.minsky*rescale2
    ccdsa.maxsky = ccdsa.maxsky*rescale2
    dzp = newzp-oldzp
    ccdsa.psfdepth += dzp
    ccdsa.gausspsfdepth += dzp
    ccdsa.galdepth += dzp
    ccdsa.gaussgaldepth += dzp

def main(args=None):
    import argparse
    parser = argparse.ArgumentParser(
        description='Update ccd_cuts, add depth & CCD-fraction cuts.',
        epilog='EXAMPLE: %(prog)s survey survey-ccds ccds-annotated survey-ccds-out')
    parser.add_argument('camera', type=str, help='decam, 90prime, or mosaic')
    parser.add_argument('survey-ccds', type=str,
                        help='survey-ccds file name')
    parser.add_argument('ccds-annotated', type=str,
                        help='ccds-annotated file name')
    parser.add_argument('survey-ccds-out', type=str, default='',
                        help='output survey-ccds file')
    parser.add_argument('ccds-annotated-out', type=str, default='',
                        help='output annotated-ccds file')
    parser.add_argument('--tilefile', type=str, default='',
                        help='appropriate tile file for survey')
    parser.add_argument('--imlist', type=str, default='',
                        help='image list for survey, needed for 90prime')
    parser.add_argument('--image2coadd', type=str, default='',
                        help='list of DES good exposures')
    parser.add_argument('--zeropoints', type=str, default='',
                        help='ucal zero points for declination < -29.25')
    parser.add_argument('--newccdphrms', type=str, default=None,
                        help='filename for replacement ccdphrms file')
    parser.add_argument('--not-griz', default=False, action='store_true',
                        help='Omit the not-griz cut')
    parser.add_argument('--early-decam', default=False, action='store_true',
                        help='Omit the cut on early DECam data')
    parser.add_argument('--depth-cut', default=True, action='store_false',
                        help='Omit the depth cut')
    parser.add_argument('--nmatch', default=False, action='store_true',
                        help='Omit the "ccdnmatch" cut')
    parser.add_argument('--good-ccd-fraction', default=0.7, type=float,
                        help='Fraction of CCDs in an exposure that must be good to keep any chips')
    parser.add_argument('--max-seeing', default=None, type=float,
                        help='Seeing cut (default 3 arcsec)')
    parser.add_argument('--phrms-cut', default=None, type=float,
                        help='Cut on phrms (photometric scatter), default 0.1 mag')
    parser.add_argument('--exptime-cut', default=None, type=float,
                        help='Cut on exptime, default 30 sec')
    numpy.seterr(invalid='raise')
    args = parser.parse_args(args=args)
    ccds = fits_table(getattr(args, 'survey-ccds'))
    annotated = fits_table(getattr(args, 'ccds-annotated'))
    if args.newccdphrms is not None:
        ccdphrms = fits_table(args.newccdphrms)
        okrms = numpy.isfinite(ccdphrms.ccdphrms)
        ccds.ccdphrms = numpy.where(okrms, ccdphrms.ccdphrms, 0.)
        annotated.ccdphrms = ccds.ccdphrms
        ccds.ccdzpt = numpy.where(okrms, ccdphrms.ccdzpt, 0.)
        annotated.ccdzpt = ccds.ccdzpt
        s = numpy.argsort(ccds.image_filename)
        ccds.phrms = numpy.zeros_like(ccds.ccdphrms)
        for f, l in subslices(ccds.image_filename[s]):
            ind = s[f:l]
            m = ccds.ccdzpt[ind] != 0
            if numpy.sum(m) == 0:
                ccds.zpt[ind] = 0
                ccds.phrms[ind] = 0
            else:
                ccds.zpt[ind[m]] = numpy.nanmedian(ccds.ccdzpt[ind[m]])
                ccds.phrms[ind[m]] = numpy.nanmedian(ccds.ccdphrms[ind[m]])
        annotated.zpt = ccds.zpt
        annotated.phrms = ccds.phrms
    if numpy.any((ccds.image_filename != annotated.image_filename) |
                 (ccds.image_hdu != annotated.image_hdu)):
        raise ValueError('survey and annotated CCDs files must be row-matched!')
    if not numpy.all(ccds.ccd_cuts == 0):
        print('Warning: zeroing existing ccd_cuts')
        ccds.ccd_cuts = 0.
    if len(args.zeropoints) > 0:
        zp = fits_table(args.zeropoints)
        patch_zeropoints(zp, ccds, annotated)

    from pkg_resources import resource_filename
    fn = resource_filename('legacyzpts',
                           'data/{}-bad_expid.txt'.format(args.camera))
    if os.path.exists(fn):
        bad_expid = psfzpt_cuts.read_bad_expid(fn)
    else:
        bad_expid = {}

    kwargs = {}
    if args.max_seeing:
        kwargs.update(max_seeing=args.max_seeing)
    if args.phrms_cut:
        kwargs.update(phrms_cut=args.phrms_cut)
    if args.exptime_cut:
        kwargs.update(exptime_cut=args.exptime_cut)

    psfzpt_cuts.add_psfzpt_cuts(ccds, args.camera, bad_expid,
                                image2coadd=args.image2coadd,
                                **kwargs)

    if args.not_griz:
        ccds.ccd_cuts &= ~psfzpt_cuts.CCD_CUT_BITS['not_griz']
    if args.early_decam:
        ccds.ccd_cuts &= ~psfzpt_cuts.CCD_CUT_BITS['early_decam']
    if args.nmatch:
        ccds.ccd_cuts &= ~psfzpt_cuts.CCD_CUT_BITS['ccdnmatch']
    depthbit = psfzpt_cuts.CCD_CUT_BITS['depth_cut']
    manybadbit = psfzpt_cuts.CCD_CUT_BITS['too_many_bad_ccds']
    if not numpy.all((ccds.ccd_cuts & depthbit) == 0):
        print('Warning: some depth_cut bits already set; zeroing...')
    ccds.ccd_cuts = ccds.ccd_cuts & ~depthbit
    mbad = good_ccd_fraction(args.camera, ccds) < args.good_ccd_fraction
    ccds.ccd_cuts = ccds.ccd_cuts | (manybadbit * mbad)
    if args.depth_cut:
        dcut = depthcut(args.camera, ccds, annotated, tilefile=args.tilefile,
                        imlist=args.imlist)
        ccds.ccd_cuts = ccds.ccd_cuts | (depthbit * ~dcut)
    annotated.ccd_cuts = ccds.ccd_cuts
    ccds.write_to(getattr(args, 'survey-ccds-out'))
    annotated.writeto(getattr(args, 'ccds-annotated-out'))

if __name__ == '__main__':
    main()
