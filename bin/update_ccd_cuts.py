import os
import numpy
from astropy.io import fits
import pdb
from astrometry.util.fits import fits_table, merge_tables


class subslices:
    "Iterator for looping over subsets of an array"
    def __init__(self, data, uind=None, **kw):
        if uind is None:
            _, self.uind = numpy.unique(data, return_index=True, **kw)
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
        ccds = repair_object_names(ccds, imlist)
        return depthcut_90prime_alternative(ccds, annotated)
    elif survey == 'mosaic':
        return depthcut_mosaic(ccds, annotated, tilefile)
    raise ValueError('No such survey?')


def keep_deepest_des(ccds, annotated):
    tileid = ccds.object
    return keep_deepest_ccds(ccds.image_filename, tileid, ccds.filter,
                             annotated.psfdepth, n=1)

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
    return res


def keep_deepest_ccds(filename, tileid, filt, depth, n=2):
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
                             annotated.psfdepth, n=1)


def depthcut_decam(ccds, annotated, tilefile):
    if len(tilefile) == 0:
        raise ValueError('DECam depth cut requires a tile file.')
    # need ccds & annotated ccds: former has ccd_cuts, latter has psfdepth
    # these are not row matched!
    s = numpy.lexsort([ccds.image_hdu, ccds.image_filename])
    ccds = ccds[s]
    sa = numpy.lexsort([annotated.image_hdu, annotated.image_filename])
    annotated = annotated[sa]
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
    m = ccds.propid != '2012B-0001'
    keep[m & keep] = keep_deepest_tiles(
        ccds[m & keep], annotated[m & keep], tilefile)
    keep[~m & keep] = keep_deepest_des(ccds[~m & keep], annotated[~m & keep])
    keep_unsorted = numpy.zeros(len(keep), dtype='bool')
    keep_unsorted[s] = keep
    return keep_unsorted
    


def depthcut_propid_decam(ccds):
    # this just takes everything in certain propids, that have certain
    # survey-like object names, and trusts the surveys not to get too many
    # exposures on a single tile center.  David prefers an approach that
    # explicitly tries to limit the number of epochs on each tile center.
    mpropid = numpy.zeros(len(ccds), dtype='bool')
    # DECaLS, DECaLS+, DES, Bonaca, BLISS, DeROSITA
    for pid in ['2014B-0404', '2016A-0190', '2012B-0001', '2015A-0620', 
                '2017A-0260', '2017A-0388']: 
        mpropid = mpropid | (ccds.propid  == pid)
    # within these surveys, limit to 'normal' program, as identified
    # by object name.  This excludes, e.g., special DECaLS observations on the 
    # COSMOS field, DES SN fields, ...
    mobject = numpy.array([('DECaLS_' in o) or ('DES survey' in o) or 
                           ('BLISS field' in o) or ('DeCaLS' == o.strip()) or 
                           ('Tile ' in o) for o in ccds.object])
    def isint(x):
        try:
            _ = int(x)
        except:
            return False
        return True
    misint = numpy.array([isint(x) for x in ccds.object])
    m = mpropid & (mobject | misint)
    return m


def make_plots(ccds, depthcut, rad=1., nside=512, 
               xrange=[360, 0], yrange=[-90, 90], vmin=0, vmax=10, 
               filename=None):
    import util_efs
    from matplotlib import pyplot as p
    ccdcut = ccds.ccd_cuts
    filts = numpy.unique(ccds.filter[ccdcut])
    _, u = numpy.unique(ccds.expnum, return_index=True)
    for f in filts:
        mf = (ccds.filter[u] == f)
        maps = {'original': (mf & (ccdcut[u] == 0)),
                'depthcut': (mf & (ccdcut[u] == 0) & depthcut[u]),
                'excluded': (mf & (ccdcut[u] == 0) & ~depthcut[u])}
        for (label, m) in maps.items():
            _, ncov = util_efs.paint_map(
                ccds.ra_bore[u[m]], ccds.dec_bore[u[m]], 
                numpy.ones(numpy.sum(m)), rad, nside=nside)
            p.figure(label)
            p.clf()
            util_efs.imshow(ncov, xrange=xrange, yrange=yrange, 
                            vmin=vmin, vmax=vmax)
            p.colorbar().set_label('n_exp')
            p.title(label)
            if filename is not None:
                p.savefig(filename % (label, f))


def good_ccd_fraction(survey, ccds):
    if survey not in ['decam', '90prime', 'mosaic']:
        raise ValueError('survey not recognized!')
    nccds = 62 if survey == 'decam' else 4
    ngooddict = {}
    for expnum, cut in zip(ccds.expnum, ccds.ccd_cuts):
        if cut == 0:
            ngooddict[expnum] = ngooddict.get(expnum, 0) + 1
    ngoodccds = numpy.array([ngooddict.get(e, 0) for e in ccds.expnum])
    if numpy.any(ngoodccds > nccds):
        raise ValueError('Some exposures have more good CCDs than should be possible!')
    return ngoodccds/float(nccds)


if __name__ == "__main__":
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
    parser.add_argument('--tilefile', type=str, default='',
                        help='appropriate tile file for survey')
    parser.add_argument('--imlist', type=str, default='',
                        help='image list for survey, needed for 90prime')
    args = parser.parse_args()
    ccds = fits_table(getattr(args, 'survey-ccds'))
    if not numpy.all(ccds.ccd_cuts == 0):
        print('Warning: zeroing existing ccd_cuts')
        ccds.ccd_cuts = 0.
    from legacyzpts import psfzpt_cuts
    from pkg_resources import resource_filename
    fn = resource_filename('legacyzpts', 
                           'data/{}-bad_expid.txt'.format(args.camera))
    bad_expid = psfzpt_cuts.read_bad_expid(fn)
    psfzpt_cuts.add_psfzpt_cuts(ccds, args.camera, bad_expid)

    annotated = fits_table(getattr(args, 'ccds-annotated'))
    depthbit = psfzpt_cuts.CCD_CUT_BITS['depth_cut']
    manybadbit = psfzpt_cuts.CCD_CUT_BITS['too_many_bad_ccds']
    if not numpy.all((ccds.ccd_cuts & depthbit) == 0):
        print('Warning: some depth_cut bits already set; zeroing...')
    ccds.ccd_cuts = ccds.ccd_cuts & ~depthbit
    mbad = good_ccd_fraction(args.camera, ccds) < 0.7
    ccds.ccd_cuts = ccds.ccd_cuts | (manybadbit * mbad)
    dcut = depthcut(args.camera, ccds, annotated, tilefile=args.tilefile,
                    imlist=args.imlist)
    ccds.ccd_cuts = ccds.ccd_cuts | (depthbit * ~dcut)
    ccds.write_to(getattr(args, 'survey-ccds-out'))
