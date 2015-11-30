#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np

from legacyproduct.internal import sharedmem
from legacyproduct.internal.io import iter_tractor, parse_filename

import argparse
import os, sys
from time import time
from scipy.spatial import cKDTree as KDTree

import fitsio

def main():
    ns = parse_args()
    
    bricks = list_bricks(ns)

    tree, boss = read_boss(ns.boss, ns)

    # convert to radian
    tol = ns.tolerance / (60. * 60. * 180) * np.pi
    nprocessed = np.zeros((), dtype='i8')
    t0 = time()
    with sharedmem.MapReduce(np=ns.numproc) as pool:
        def work(brickname, path):

            data = process(brickname, path, tree, boss, tol)
            mask = data['FIBERID'] != 0xdead
            matched = mask.sum()

            destpath = os.path.join(ns.dest, os.path.relpath(path, ns.src))
            destpath = destpath.replace('tractor', 'decals-boss')
            mainpath = os.path.splitext(destpath)[0]
            
            try:
                os.makedirs(os.path.dirname(destpath))
            except OSError:
                pass
            header = {}

            header['NMATCHED'] = matched
            header['TOL_ARCSEC'] = ns.tolerance

            for format in ns.format:
                destpath = mainpath + '.' + format
                save_file(destpath, data, header, format)

            return brickname, matched, len(data)

        def reduce(brickname, matched, total):
            nprocessed[...] += 1
            if ns.verbose:
                if nprocessed % 50 == 0:
                    print("Processed %d files, %g / second."
                        % (nprocessed, nprocessed / (time() - t0))
                        )

        pool.map(work, bricks, star=True, reduce=reduce)

def process(brickname, path, tree, boss, tol):
    objects = fitsio.read(path, 1, upper=True)
    pos = radec2pos(objects['RA'], objects['DEC'])
    d, i = tree.query(pos, 1)

    mask = d < tol
    result = np.empty(len(objects), boss.dtype)
    result[mask] = boss[i[mask]]
    result[~mask]['FIBERID'] = 0xdead
    return result

def save_file(filename, data, header, format):
    if format == 'fits':
        fitsio.write(filename, data, extname='DECALS-BOSS', header=header, clobber=True)
    elif format == 'hdf5':
        import h5py
        with h5py.File(filename, 'w') as ff:
            dset = ff.create_dataset('DECALS-BOSS', data=data)
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

def read_boss(filename, ns):
    t0 = time()
    boss = fitsio.FITS(filename, upper=True)[1][:]
    boss = sharedmem.copy(boss)

    if ns.verbose:
        print("reading BOSS catlaogue took %g seconds." % (time() - t0))
        print("%d BOSS objects." % len(boss))

    t0 = time()
    for raname, decname in [
            ('RA', 'DEC'), 
            ('PLUG_RA', 'PLUG_DEC')
            ]:
        if raname in boss.dtype.names \
        and decname in boss.dtype.names: 
            ra = boss[raname]
            dec = boss[decname]
            if ns.verbose:
                print('using %s/%s for positions.' % (raname, decname))
            break
    else:
        raise KeyError("No RA/DEC or PLUG_RA/PLUG_DEC in the BOSS catalogue")

    pos = radec2pos(ra, dec)
    pos = sharedmem.copy(pos)

    tree = KDTree(pos)

    if ns.verbose:
        print("Building KD-Tree took %g seconds." % (time() - t0))

    return tree, boss

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
        d = dict([(brickname, d[brickname]) 
                             for brickname in bricklist])

    t0 = time()

    bricks = sorted(d.items())

    return bricks
    
def parse_args():
    ap = argparse.ArgumentParser(
    description="""Match Boss Catalogue for DECALS. 
        This will create a mirror of tractor catalogue directories, but each file would only contains
        The corresponding object in BOSS DR12.
        """
        )

    ap.add_argument("boss", help="BOSS DR12 catalogue. e.g. /global/project/projectdirs/cosmo/work/sdss/cats/specObj-dr12.fits")
    ap.add_argument("src", help="Path to the root directory of all tractor files")
    ap.add_argument("dest", help="Path to the root directory of output matched catalogue")

    ap.add_argument('-f', "--format", choices=['fits', 'hdf5'], nargs='+', default=["fits"],
        help="Format of the output sweep files")

    ap.add_argument('-t', "--tolerance", default=1., type=float,
        help="Tolerance of the angular distance for a match, in arc-seconds")

    ap.add_argument('-F', "--filelist", default=None,
        help="list of tractor brickfiles to use; this will avoid expensive walking of the path.")

    ap.add_argument('-b', "--bricklist", 
        help="""Filename with list of bricknames to include. 
                If not set, all bricks in src are included, sorted by brickname.
            """)

    ap.add_argument('-v', "--verbose", action='store_true')

    ap.add_argument("--numproc", type=int, default=None,
        help="""Number of concurrent processes to use. 0 for sequential execution. 
            Default is to use OMP_NUM_THREADS, or the number of cores on the node.""")

    return ap.parse_args()
   

if __name__ == '__main__':
    main() 
