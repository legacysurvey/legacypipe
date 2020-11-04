#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np

from legacypipe.internal import sharedmem
from legacypipe.internal.io import iter_tractor, parse_filename, get_units, git_version

import argparse
import os, sys
from time import time
from scipy.spatial import cKDTree as KDTree

import fitsio
import platform

print('Running from %s' % platform.node())

def main():
    ns = parse_args()
        
    if ns.ignore_errors:
        print("Warning: *** Will ignore broken tractor catalog files ***")
        print("         *** Disable -I for final data product.         ***")

    bricks = list_bricks(ns)

    # ADM grab a {FIELD: unit} dict from the first Tractor file.
    unitdict = get_units(bricks[0][1])

    # convert to radian
    tol = ns.tolerance / (60. * 60.)  * (np.pi / 180)

    tree, nobj, morecols, maxdups = read_external(ns.external, tol, ns)

    # get the data type of the match
    brickname, path = bricks[0]
    peek = fitsio.read(path, 1, upper=True)
    matched_catalog = sharedmem.empty(nobj, dtype=peek.dtype)
    matched_catalog['OBJID'] = -1

    matched_distance = sharedmem.empty(nobj, dtype='f4')

    matched_distance[:] = tol
    nprocessed = np.zeros((), dtype='i8')
    nmatched = np.zeros((), dtype='i8')
    ntotal = np.zeros((), dtype='i8')
    t0 = time()

    with sharedmem.MapReduce(np=ns.numproc) as pool:
        def work(brickname, path):
            try:
                objects = fitsio.read(path, 1, upper=True)
            except:
                if ns.ignore_errors:
                    print ("IO Error on %s" %path)
                    return None, None, None
                else:
                    raise

            # ADM limit to just PRIMARY objects from imaging.
            bp = objects["BRICK_PRIMARY"]
            objects = objects[bp]
            pos = radec2pos(objects['RA'], objects['DEC'])

            # ADM query tree allowing duplicates.
            dd, ii = tree.query(pos, maxdups, distance_upper_bound=tol)

            # ADM collect relevant information (retaining duplicates).
            _s = ii[dd < tol]           # ADM the spec object indices.
            _p = np.where(dd < tol)[0]  # ADM the imaging object indices.
            _d = dd[dd < tol]           # ADM the matching distances.

            # ADM bail if there are no matches.
            if len(_s) == 0:
                return brickname, 0, len(objects)

            # ADM look-up dictionaries of the relevant distances and
            # ADM imaging object indices for each spec object index.
            ddict, pdict = {s: [] for s in _s}, {s: [] for s in _s}
            _ = [ddict[s].append(d) for s, d in zip(_s, _d)]
            _ = [pdict[s].append(p) for s, p in zip(_s, _p)]

            # ADM collapse the lookup dict based on minimum distances.
            sdp = [[s, d[np.argmin(d)], p[np.argmin(d)]] for s, d, p in
                   zip(ddict.keys(), ddict.values(), pdict.values())]

            # ADM we're left with the spectroscopic and photometric indexes
            # ADM distances and indexes contingent on the minimum distances.
            i = np.array(sdp, dtype='i4')[:,0]
            d = np.array(sdp, dtype='f4')[:,1]
            iphot = np.array(sdp, dtype='i4')[:,2]

            assert (objects['OBJID'] != -1).all()
            with pool.critical:
                mask = d < matched_distance[i]
                i = i[mask]
                iphot = iphot[mask]
                matched_catalog[i] = objects[iphot][list(matched_catalog.dtype.names)]
                matched_distance[i] = d[mask]
            matched = mask.sum()

            return brickname, matched, len(objects)

        def reduce(brickname, matched, total):
            if brickname is None:
                return
            nprocessed[...] += 1
            nmatched[...] += matched
            ntotal[...] += total
            if ns.verbose:
                if nprocessed % 1000 == 0:
                    print("Processed %d files, %g / second, matched %d / %d brick primary objects."
                        % (nprocessed, nprocessed / (time() - t0), nmatched, ntotal)
                        )

        pool.map(work, bricks, star=True, reduce=reduce)

        nrealmatched = (matched_catalog['OBJID'] != -1).sum()
        if ns.verbose:
            print("Processed %d files, %g / second, matched %d / %d objects into %d slots."
                % (nprocessed, nprocessed / (time() - t0), 
                    nmatched, ntotal, 
                    nrealmatched)
                )

        try:
            os.makedirs(os.path.dirname(ns.dest))
        except OSError:
            pass
        
        hdr = fitsio.FITSHDR()
        hdr.add_record(dict(name='NMATCHED', value=nrealmatched,
                            comment='Number of unique matches.'))
        hdr.add_record(dict(name='NCOLL', value=nmatched - nrealmatched,
                            comment='Total number of matches.'))
        hdr.add_record(dict(name='NCOLL', value=nrealmatched,
                            comment='Total number of matches.'))
        hdr.add_record(dict(name='RADIUS', value=ns.tolerance,
                            comment='Search radius (arcsec).'))
        value = ns.external
        if len(value) > 68:
            hdr.add_record(dict(name='EXTERNAL', value=value[:67]+'&'))
            while len(value):
                value = value[67:]
                if len(value) == 0:
                    break
                hdr.add_record(dict(name='CONTINUE', value="  '%s%s'" % (
                    value[:67], '&' if len(value) > 67 else '')))
            added_long = True
        else:
            added_long = False

        if added_long:
            hdr.add_record(dict(name='LONGSTRN', value='OGIP 1.0',
                                comment='CONTINUE cards are used'))

        # Optionally add the new columns
        if len(morecols) > 0:
            newdtype = matched_catalog.dtype.descr
    
            for coldata, col in zip( morecols, ns.copycols ):
                newdtype = newdtype + [(col, coldata.dtype)]
            newdtype = np.dtype(newdtype)
        
            _matched_catalog = np.empty(matched_catalog.shape, dtype=newdtype)
            for field in matched_catalog.dtype.fields:
                _matched_catalog[field] = matched_catalog[field]
            for coldata, col in zip( morecols, ns.copycols ):
                _matched_catalog[col] = coldata
                
            matched_catalog = _matched_catalog.copy()
            del _matched_catalog

        for format in ns.format:
            save_file(ns.dest, matched_catalog, hdr, format, unitdict=unitdict)

def save_file(filename, data, header, format, unitdict=None):
    basename = os.path.splitext(filename)[0]
    if format == 'fits':
        units = None
	# ADM derive the units from the data columns if possible.
        if unitdict is not None:
            # ADM some columns from external-match files might not have
            # ADM units, so pass an empty string for external columns.
            units = [unitdict[col] if col in unitdict.keys() else ""
                     for col in data.dtype.names]

        # ADM add the external match code version header dependency.
        dep = [int(key.split("DEPNAM")[-1]) for key in header.keys()
               if 'DEPNAM' in key]
        if len(dep) == 0:
            nextdep = 0
        else:
            nextdep = np.max(dep) + 1
        header["DEPNAM{:02d}".format(nextdep)] = 'match_external'
        header["DEPVER{:02d}".format(nextdep)] = git_version()

        filename = basename + '.fits'
        fitsio.write(filename, data, extname='MATCHED', header=header,
                     clobber=True, units=units)
    elif format == 'hdf5':
        filename = basename + '.hdf5'
        import h5py
        with h5py.File(filename, 'w') as ff:
            dset = ff.create_dataset('MATCHED', data=data)
            for key in header:
                dset.attrs[key] = header[key]
    else:
        raise ValueError("Unknown format")
    

def radec2pos(ra, dec):
    pos = np.empty(len(ra), ('f4', 3))
    pos[:, 2] = np.sin(dec / 180. * np.pi)
    pos[:, 1] = np.cos(dec / 180. * np.pi)
    pos[:, 0] = pos[:, 1]
    pos[:, 0] *= np.sin(ra / 180. * np.pi)
    pos[:, 1] *= np.cos(ra / 180. * np.pi)
    return pos

def read_external(filename, tol, ns=None):
    t0 = time()
    cat = fitsio.FITS(filename, upper=True)[1][:]

    # ADM some defaults to make it easier to import this function.
    _verbose = True
    _copycols = None
    if ns is not None:
        _verbose = ns.verbose
        _copycols = ns.copycols

    if _verbose:
        print("reading external catalog took %g seconds." % (time() - t0))
        print("%d objects." % len(cat))

    t0 = time()
    for raname, decname in [
            ('RA', 'DEC'), 
            ('PLUG_RA', 'PLUG_DEC')
            ]:
        if raname in cat.dtype.names \
        and decname in cat.dtype.names: 
            ra = cat[raname]
            dec = cat[decname]
            if _verbose:
                print('using %s/%s for positions.' % (raname, decname))
            break
    else:
        raise KeyError("No RA/DEC or PLUG_RA/PLUG_DEC in the external catalog")

    pos = radec2pos(ra, dec)
    # work around NERSC overcommit issue.
    pos = sharedmem.copy(pos)

    tree = KDTree(pos)

    # ADM determine the maximum possible number of duplicated objects.
    ndups = 0
    maxdups = 1000
    if _verbose:
        print("Determing max number of duplicates in external catalog")
    while ndups < maxdups:
        ndups += 100
        d, _ = tree.query(pos, ndups, distance_upper_bound=2*tol)
        maxdups = np.max(np.sum(d != np.inf, axis=1))
        if _verbose:
            print("maximum number of duplicates is {}".format(maxdups))

    if _verbose:
        print("Building KD-Tree took %g seconds." % (time() - t0))

    morecols = []
    if _copycols is not None:
        for col in np.atleast_1d(ns.copycols):
            if col not in cat.dtype.names:
                print('Column {} does not exist in external catalog!'.format(col))
                raise IOError
            morecols.append(cat[col])

    return tree, len(cat), morecols, maxdups

def list_bricks(ns):
    t0 = time()

    if ns.filelist is not None:
        d = dict([(parse_filename(fn.strip()), fn.strip()) 
            for fn in open(ns.filelist, 'r').readlines()])
    else:
        d = dict(iter_tractor(ns.src))

    if ns.verbose:
        print('enumerated %d bricks in %g seconds' % (
            len(d), time() - t0))

    #- Load list of bricknames to use
    if ns.bricklist is not None:
        bricklist = np.loadtxt(ns.bricklist, dtype='S8')
        # TODO: skip unknown bricks?
        d = dict([(brickname.decode(), d[brickname]) 
                             for brickname in bricklist])

    t0 = time()

    bricks = sorted(d.items())

    return bricks
    
def parse_args():
    ap = argparse.ArgumentParser(
    description="""Match to an external catalogs.
        """
        )

    ap.add_argument("external", help="External catalog. e.g. /global/project/projectdirs/cosmo/work/sdss/cats/specObj-dr12.fits")
    ap.add_argument("src", help="Path to the root directory of all tractor files")
    ap.add_argument("dest", help="Path to the output file")

    ap.add_argument('-f', "--format", choices=['fits', 'hdf5'], nargs='+', default=["fits"],
        help="Format of the output file")

    ap.add_argument('-t', "--tolerance", default=1.5, type=float,
        help="Tolerance of the angular distance for a match, in arcseconds")

    ap.add_argument('-F', "--filelist", default=None,
        help="list of tractor brickfiles to use; this will avoid expensive walking of the path.")

    ap.add_argument('-b', "--bricklist", 
        help="""Filename with list of bricknames to include. 
                If not set, all bricks in src are included, sorted by brickname.
            """)

    ap.add_argument('-v', "--verbose", action='store_true')

    ap.add_argument('-I', "--ignore-errors", action='store_true')

    ap.add_argument("--numproc", type=int, default=None,
        help="""Number of concurrent processes to use. 0 for sequential execution. 
            Default is to use OMP_NUM_THREADS, or the number of cores on the node.""")

    ap.add_argument("--copycols", nargs='*', help="List of columns to copy from external to matched output catalog (e.g., MJD, FIBER, PLATE)", default=None)

    return ap.parse_args()

if __name__ == '__main__':
    main()
