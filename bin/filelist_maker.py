# example usage: python filelist_maker.py decam filelist dr8-ondisk-decam-v5.fits --allsummaryfn dr8-allfiles-decam-v5.fits
# 'filelist' is the result from something like: 
# find $IMDIR/BOK_CP/ $IMDIR/BOK_TEST/ > filelist 
# dr8-ondisk-decam-v5.fits is the final
# output file list the 'allsummary' file contains the scraped headers
# from all files (not just the matching image files, but weight and dq
# files as well).  In principle code (not yet written) could append new
# directories to the allfiles file and then run just the downselection
# to get the final file list, which would save retrawling the full directory
# structure again.

import os
import pdb
import numpy
from astropy.io import fits
from astropy.coordinates import Angle

def match(tfile, calib, flavor):
    res = [0, 0, 0]
    for i, field in enumerate(['expnum', 'plver', 'procdate']):
        if field not in calib.dtype.names:
#            print('Missing field %s for expnum %d, %s' % 
#                  (field, tfile['expnum'], flavor))
            res[i] = res[i] | 2**0
        elif not numpy.all(calib[field] == tfile[field]):
#            print('Mismatch in %s for expnum %d, %s' % 
#                  (field, tfile['expnum'], flavor))
            res[i] = res[i] | 2**1
    return res

def check(flist, calibdir, prefix, mask=True):
    if mask:
        flist = flist[flist['qkeep'] != 0]
    # at the moment, check merged skies and merged psf match David's file.
    res = numpy.zeros(len(flist), 
                      dtype=[('splinesky', '3i4'), ('psfex', '3i4')])
    for i, tfile in enumerate(flist):
        if (i % 1000) == 0:
            print('Progress %d/%d' % (i, len(flist)))
        expnum = tfile['expnum']
        procdate = tfile['procdate']
        plver = tfile['plver']
        filt = tfile['filter']
        # just read merged, for the moment.
        expnumstr = '%08d' % expnum
        try:
            psfex = fits.getdata(os.path.join(
                calibdir, 'psfex-merged', expnumstr[:5], 
                '%s-%s.fits' % (prefix, expnumstr)))
            res['psfex'][i, :] = match(tfile, psfex, 'psfex')
        except FileNotFoundError:
            # print('Missing psfex file', tfile['expnum'])
            res['psfex'][i, :] |= 2**2
        try:
            splinesky = fits.getdata(os.path.join(
                calibdir, 'splinesky-merged', expnumstr[:5], 
                '%s-%s.fits' % (prefix, expnumstr)))
            res['splinesky'][i, :] = match(tfile, splinesky, 'splinesky')
        except FileNotFoundError:
            # print('Missing splinesky file', tfile['expnum'])
            res['splinesky'][i, :] |= 2**2
    return res


def get_expnum(primhdr):
    if 'EXPNUM' in primhdr:
        if not isinstance(primhdr['EXPNUM'], fits.Undefined):
            return primhdr['EXPNUM']
    if primhdr.get('detname', 'empty').strip() == '90prime':
        return get_expnum_90prime(primhdr)
    # At the beginning of the survey, eg 2016-01-24, the EXPNUM
    # cards are blank.  Fake up an expnum like 160125082555
    # (yymmddhhmmss), same as the CP filename.
    # OBSID   = 'kp4m.20160125T082555' / Observation ID
    if 'OBSID' not in primhdr:
        return -1
    obsid = primhdr['OBSID']
    obsid = obsid.strip().split('.')
    if len(obsid) == 1:
        return -1
    obsid = obsid[1].replace('T', '')
    obsid = int(obsid[2:], 10)
    return obsid


def get_expnum_90prime(primhdr):
    import re
    # /descache/bass/20160710/d7580.0144.fits --> 75800144
    base = (os.path.basename(primhdr['DTACQNAM'])
            .replace('.fits','')
            .replace('.fz',''))
    return int( re.sub(r'([a-z]+|\.+)','',base) )


def flistsummary(flist):
    out = numpy.zeros(len(flist), dtype=[
        ('filename', 'U200'), ('object', 'U40'), ('propid', 'U20'),
        ('expnum', 'i8'), ('obstype', 'U10'),
        ('wcscal', 'U20'), ('scampflg', 'i4'),
        ('photcal', 'U20'), ('yshift', 'f4'),
        ('proctype', 'U20'), ('prodtype', 'U20'), ('exptime', 'f4'),
        ('ra', 'f8'), ('dec', 'f8'), ('date_obs', 'U26'),
        ('mjd_obs', 'f8'), ('airmass', 'f4'), ('filter', 'U32'),
        ('plver', 'U8'), ('procdate', 'U19'), ('plprocid', 'U20'),
        ('qkeep', 'bool')])
    for i, f in enumerate(flist):
        if (i % 10000) == 0:
            print('Progress: %d/%d' % (i, len(flist)))
        out['filename'][i] = flist[i]
        try:
            hdr = fits.getheader(f)
            expnum = get_expnum(hdr)
        except (OSError, fits.VerifyError):
            out['expnum'][i] = -1
            out['object'][i] = 'could not read in, corrupt?'
            print('corrupt?  %s' % f)
            continue
        for field in out.dtype.names:
            if field in ['filename', 'qkeep']:
                continue
            if field == 'expnum':
                out[field][i] = get_expnum(hdr)
            elif field == 'ra':
                if 'CENTRA' in hdr:
                    out[field][i] = hdr.get('CENTRA', numpy.nan)
                elif 'RA' in hdr:
                    fieldhdr = hdr['RA']
                    if isinstance(fieldhdr, float):
                        print(f, fieldhdr)
                        fieldhdr = str(fieldhdr)
                    out[field][i] = Angle(fieldhdr + ' hours').degree
                else:
                    out[field][i] = numpy.nan
            elif field == 'dec':
                if 'CENTDEC' in hdr:
                    out[field][i] = hdr.get('CENTDEC', numpy.nan)
                elif 'DEC' in hdr:
                    fieldhdr = hdr['DEC']
                    if isinstance(fieldhdr, float):
                        print(f, fieldhdr)
                        fieldhdr = str(fieldhdr)
                    out[field][i] = Angle(fieldhdr + ' degrees').degree
                else:
                    out[field][i] = numpy.nan
            elif field == 'procdate':
                out[field][i] = hdr.get('DATE', 'empty')
            elif field in ['object', 'prodtype', 'filter', 'plver', 
                           'wcscal',
                           'photcal', 'plprocid', 'propid']:
                out[field][i] = hdr.get(field, 'empty')
            elif field in ['sortver', 'airmass', 'exptime', 'mjd_obs',
                           'scampflg']:
                out[field][i] = hdr.get(field.replace('_', '-'), -1)
            elif field == 'yshift':
                out[field][i] = hdr.get(field, numpy.nan)
            else:
                out[field][i] = hdr[field.replace('_', '-')]
    return out


def check_ooidw(flist, fields=['plprocid', 'plver', 'expnum']):
    # for every image, we want two other images with appropriate names to
    # exist.  Their prodtypes should also match their names.
    idw = numpy.zeros(len(flist), dtype='bool')
    dirs = numpy.array([os.path.dirname(f) for f in flist['filename']])
    s = numpy.lexsort((dirs,)+ tuple(flist[f] for f in fields))
    inds = [tuple(x) for x in zip(
        dirs[s], *(flist[f][s] for f in fields))]
    delts = numpy.array([i != j for i, j in zip(inds[:-1], inds[1:])])
    linds = numpy.concatenate([numpy.flatnonzero(delts)+1, [len(inds)]])
    finds = numpy.concatenate([[0], numpy.flatnonzero(delts)+1])
    prodtypedict = {'image': '_ooi_',
                    'wtmap': '_oow_',
                    'dqmask': '_ood_'}
    imstrs = list(prodtypedict.values())
    filename = flist['filename']
    if isinstance(filename, numpy.chararray):
        newfilename = numpy.zeros(len(filename), dtype=filename.dtype)
        newfilename[:] = filename
        filename = newfilename
    for f, l in zip(finds, linds):
        fnames = [filename[i] for i in s[f:l]]
        for i in range(f, l):
            fname = filename[s[i]]
            repstr = prodtypedict.get(
                flist['prodtype'][s[i]], None)
            if repstr is None:
                continue
            if (repstr == '_ooi_') and (repstr not in fname):
                repstr = '_oki_'
            temp = [fname.replace(repstr, imstr) in fnames
                    for imstr in imstrs]
            idw[s[i]] = numpy.all(temp)
    return idw


def most_recent_versions(flist):
    if len(flist) == 0:
        return numpy.zeros(len(flist), dtype='bool')
    # only send images, not weights or dqmasks, and only good ones.
    dirs = numpy.array([os.path.dirname(f) for f in flist['filename']])
    from distutils import version
    versions = [version.LooseVersion(v)
                for v in flist['plver']]
    s = numpy.lexsort((dirs, flist['plprocid'], versions, flist['expnum']))
    delts = numpy.array([i != j for i, j in 
                         zip(flist['expnum'][s[:-1]], flist['expnum'][s[1:]])])
    linds = numpy.concatenate([numpy.flatnonzero(delts), [len(flist)-1]])
    m = numpy.zeros(len(flist), dtype='bool')
    m[s[linds]] = 1
    return m


def flag_all(flist):
    wcsok = (flist['wcscal'] == 'Successful') | (flist['scampflg'] == 0)
    flag = ((flist['expnum'] <= 0)*2**0 +
            (flist['exptime'] <= 29)*2**1 +
            (flist['proctype'] != 'InstCal')*2**2 +
            (~wcsok)*2**3 +
            (check_ooidw(flist) == 0)*2**4)
    return flag


def qkeep_all(flist, problem=False):
    flag = flag_all(flist)
    res = flag == 0
    if problem:
        res = (res, flag)
    return res


def flag_mosaic(flist):
    yshift = flist['yshift'].copy()
    yshift[~numpy.isfinite(yshift)] = 0.
    shifted = (yshift > 0.33) & (yshift < 0.34)
    return ((flist['filter'] != 'zd DECam k1038')*2**0 +
            ((flist['mjd_obs'] <= 57674) & (shifted == 0))*2**1)

def qkeep_mosaic(flist, problem=False):
    # m = flist['filter'] == 'zd DECam k1038'
    # yshift = flist['yshift'].copy()
    # yshift[~numpy.isfinite(yshift)] = 0.
    # shifted = (yshift > 0.33) & (yshift < 0.34)
    # m = m & (flist['mjd_obs'] > 57674) | (
    #     (flist['mjd_obs'] < 57674) & shifted)
    # bad_expid?
    flag = flag_mosaic(flist)
    res = flag == 0
    if problem:
        res = (res, flag)
    return res


def qkeep_decam(flist, problem=False):
    m = numpy.zeros(len(flist), dtype='bool')
    filters = [
        'g DECam SDSS c0001 4720.0 1520.0',
        'r DECam SDSS c0002 6415.0 1480.0',
        'i DECam SDSS c0003 7835.0 1470.0',
        'z DECam SDSS c0004 9260.0 1520.0',
        'Y DECam c0005 10095.0 1130.0',
#        'VR DECam c0007 6300.0 2600.0',
#        'solid plate 0.0 0.0',
#        'u DECam c0006 3500.0 1000.0',
    ]
    for f in filters:
        m = m | (flist['filter'] == f)
    # m = m & (flist['mjd_obs'] > 56730)
    # MJD cut can occur post-calibrations.
    # bad_expid?
    if problem:
        m = (m, ~m)
    return m


def qkeep_90prime(flist, problem=False):
    m = (flist['filter'] == 'g') | (flist['filter'] == 'bokr')
    if problem:
        m = (m, ~m)
    return m


def fits_to_flist(fits):
    # this tries to handle some annoying unicode / bytes conversion issues
    # in usings FITS to write/read in flist files.
    # it feels like this should not work, so it is liable to break when numpy /
    # astropy.io.fits changes its unicode handling.
    origdtype = fits.dtype
    newdtype = [(n, t.replace('S', 'U')) for (n, t) in origdtype.descr]
    out = numpy.zeros(len(fits), dtype=newdtype)
    for field, _ in newdtype:
        out[field] = fits[field]
    return out
    


def report_problems(flist):
    # want all the exposures that have qkeep = 0
    # for all versions, because of inconsistency
    s = numpy.argsort(flist['expnum'])
    mcorrupt = numpy.array(['could not read in' in o for o in flist['object']])
    ind = numpy.flatnonzero(mcorrupt)
    if numpy.any(ind):
        print('Corrupt files:')
        for i in ind:
            print(flist['filename'][i])
        print('')
    s = numpy.lexsort((flist['filename'], flist['expnum']))
    flists = flist[s]
    _, finds = numpy.unique(flists['expnum'], return_index=True)
    linds = numpy.concatenate([finds[1:]-1, [len(s)-1]])
    print('Inconsistent/absent ooi/oow/ood...')
    import pdb
    for f, l in zip(finds, linds):
        if numpy.all((flists['flag1'][f:l] & 2**4) != 0):
            print(flists['filename'][f])
    print('')


def filelist(flist, survey, stripndir=6, report=True):
    m, flag_a = qkeep_all(flist, problem=True)
    qkeepfun = { 'mosaic': qkeep_mosaic,
                 '90prime': qkeep_90prime,
                 'decam': qkeep_decam }
    if survey in qkeepfun:
        ms, flag_s = qkeepfun[survey](flist, problem=True)
    else:
        raise ValueError('unknown survey!')
    m = m & ms
    flist['qkeep'] = m
    from numpy.lib.recfunctions import rec_append_fields
    flist = rec_append_fields(flist, ['flag1', 'flag2'], [flag_a, flag_s])
    if report:
        report_problems(flist)
    flist = flist[(flist['prodtype'] == 'image') & 
                  (flist['proctype'] == 'InstCal')]
    m = flist['qkeep'] != 0
    m2 = most_recent_versions(flist[m])
    flist['qkeep'][m] = m2
    if survey == '90prime':
        # repair OBJECT keywords
        ind = numpy.flatnonzero(flist['object'] == '')
        for i, f in zip(ind, flist['filename'][ind]):
            #newpath = f.replace('BOK_CP', 'BOK_Raw')
            #newpath = newpath.replace('CP', '')
            dirs = f.split('/')
            dirs = dirs[:-3]+dirs[-2:]  # get rid of V2.0
            dirs[-3] = 'BOK_Raw'
            dirs[-2] = dirs[-2][2:]
            newpath = '/'.join(dirs)
            cut = newpath.find('_ooi_')
            newpath = newpath[:cut]+'_ori.fits.fz'
            flist['object'][i] = fits.getheader(newpath)['OBJECT']
            
    flist['filename'] = ['/'.join(f.split('/')[stripndir:])
                         for f in flist['filename']]
    return flist


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Make file list.',
        epilog='EXAMPLE: %(prog)s survey flist outflist.fits')
    parser.add_argument('survey', type=str, help='decam, 90prime, or mosaic')
    parser.add_argument('filelist', type=str, 
                        help='output of `find` listing files to consider')
    parser.add_argument('outfn', type=str,
                        help='output file name (.fits)')
    parser.add_argument('--allsummaryfn', type=str, default='',
                        help='allsumary file name (.fits)')
    parser.add_argument('--skip', type=int, default=0,
                        help='number of exposures to skip (testing only)')
    args = parser.parse_args()
    flist = open(args.filelist, 'r').readlines()
    flist = [f.strip() for f in flist if '.fits' in f]
    flist = flist[args.skip:]
    summary = flistsummary(flist)
    if len(args.allsummaryfn) > 0:
        fits.writeto(args.allsummaryfn, summary)
    out = filelist(summary, args.survey)
    fits.writeto(args.outfn, out)
