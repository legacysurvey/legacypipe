#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np

from legacypipe.internal import sharedmem
from legacypipe.internal.io import iter_tractor, parse_filename, get_units, git_version

import argparse
import os, sys
from time import time

import fitsio

# ADM these are the directory names for the sweep directory,
# ADM the "light-curves only" directory,
# ADM and the "extra Tractor columns not in the sweeps directory".
# ADM (these apply BELOW the level of the /sweep directory.
outdirnames = ["X.X", "X.X-lightcurves", "X.X-extra"]

def main():
    ns = parse_args()
    if ns.ignore_errors:
        print("Warning: *** Will ignore broken tractor catalogue files ***")
        print("         *** Disable -I for final data product.         ***")
    if ns.mopup:
        print("Warning: *** The mopup flag was passed. Existing ***")
        print("         *** sweep files will NOT be overwritten ***")
    # avoid each subprocess importing h5py again and again.
    if 'hdf5' in ns.format:
        import h5py

    # this may take a while on a file system with slow meta-data
    # access
    # bricks = [(name, filepath, region), ...]
    bricks = list_bricks(ns)

    # ADM get a {FIELD: unit} dictionary from one of the Tractor files.
    fn = bricks[0][1]
    unitdict = get_units(fn)
    # ADM read in a small amount of information from one of the Tractor
    # ADM files to establish the full dtype.
    testdata = fitsio.read(fn, rows=[0], upper=True)
    ALL_DTYPE = testdata.dtype

    t0 = time()

    for odn in outdirnames:
        try:
            os.makedirs(os.path.join(ns.dest, odn))
        except OSError:
            pass

    # blocks or ra stripes?
    schemas = {
        'ra' : sweep_schema_ra(360),
        'blocks' : sweep_schema_blocks(36, 36),
        'blocksdr10' : sweep_schema_blocks(72, 36),
        'dec' : sweep_schema_dec(180),
        }

    sweeps = schemas[ns.schema]

    t0 = time()

    nbricks_tot = np.zeros((), 'i8')
    nobj_tot = np.zeros((), 'i8')

    def work(sweep):
        # ADM the general format for a sweeps file.
        template = "sweep-%(ramin)s%(decmin)s-%(ramax)s%(decmax)s.%(format)s"
        def formatdec(dec):
            return ("%+04g" % dec).replace('-', 'm').replace('+', 'p')
        def formatra(ra):
            return ("%03g" % ra)

        # ADM the various flavors of sweeps file.
        ender = [".fits", "-lc.fits", "-ex.fits"]

        # ADM if we're mopping up, move on if all requisite files exist.
        # ADM strictly, this only checks for existence of the fits files.
        if ns.mopup:
            filename = template %  \
                       dict(ramin=formatra(sweep[0]),
                            decmin=formatdec(sweep[1]),
                            ramax=formatra(sweep[2]),
                            decmax=formatdec(sweep[3]),
                            format="fits")
            allexist = True
            for odn, end in zip(outdirnames, ender):
                fn = filename.replace(".fits", end)
                dest = os.path.join(ns.dest, odn, fn)
                allexist &= os.path.exists(dest)
            # ADM if allexist remains True, all requisite files exist.
            if allexist:
                print("won't overwrite files related to {}".format(filename))
                return filename, 0, 0

        data, header, nbricks = make_sweep(sweep, bricks, ns, ALL_DTYPE=ALL_DTYPE)

        header.update({
            'RAMIN'  : sweep[0],
            'DECMIN' : sweep[1],
            'RAMAX'  : sweep[2],
            'DECMAX' : sweep[3],
            })

        for format in ns.format:
            filename = template %  \
                dict(ramin=formatra(sweep[0]),
                     decmin=formatdec(sweep[1]),
                     ramax=formatra(sweep[2]),
                     decmax=formatdec(sweep[3]),
                     format=format)

            if len(data) > 0:
                # ADM the columns to always include to form a unique ID.
                uniqid = [dt for dt in SWEEP_DTYPE.descr if
                          dt[0]=="RELEASE" or dt[0]=="BRICKID" or dt[0]=="OBJID"]
                # ADM write out separate sweeps for:
                # ADM    the SWEEP_DTYPE columns (without light-curves).
                sweepdt = [dt for dt in SWEEP_DTYPE.descr if 'LC' not in dt[0]]
                # ADM    the SWEEP_DTYPE columns (just light-curves).
                lcdt = uniqid + [dt for dt in SWEEP_DTYPE.descr if 'LC' in dt[0]]
                # ADM    the remaining "extra" columns.
                alldt = uniqid + [dt for dt in ALL_DTYPE.descr if dt[0] not in SWEEP_DTYPE.names]
                for dt, odn, end in zip([sweepdt, lcdt, alldt], outdirnames, ender):
                    fn = filename.replace(".fits", end)
                    dest = os.path.join(ns.dest, odn, fn)
                    if len(dt) > 0:
                        newdata = np.empty(len(data), dtype=dt)
                        for col in newdata.dtype.names:
                            newdata[col] = data[col]
                        save_sweep_file(dest, newdata,
                                        header, format, unitdict=unitdict)

        return filename, nbricks, len(data)

    def reduce(filename, nbricks, nobj):
        nbricks_tot[...] += nbricks
        nobj_tot[...] += nobj

        if ns.verbose and nobj > 0:
            print (
            '%s : %d bricks %d primary objects, %g bricks / sec %g objs / sec' %
            ( filename, nbricks, nobj,
              nbricks_tot / (time() - t0),
              nobj_tot / (time() - t0),
            )
            )

    with sharedmem.MapReduce(np=ns.numproc) as pool:
        pool.map(work, sweeps, reduce=reduce)


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

    if ns.bricksdesc is not None:
        bricksdesc = fitsio.read(ns.bricksdesc, 1, upper=True)
        # ADM convert from bytes_ to str_ type if fitsio version < 1.
        if bricksdesc["BRICKNAME"].dtype.type == np.bytes_:
            bricksdesc = dict([(item['BRICKNAME'].decode(), item) for item in bricksdesc])
        else:
            bricksdesc = dict([(item['BRICKNAME'], item) for item in bricksdesc])
    else:
        bricksdesc = None

    #- Load list of bricknames to use
    if ns.bricklist is not None:
        bricklist = np.loadtxt(ns.bricklist, dtype='S8')
        # TODO: skip unknown bricks?
        d = dict([(brickname.decode(), d[brickname])
                             for brickname in bricklist])

    t0 = time()

    with sharedmem.MapReduce(np=ns.numproc) as pool:
        chunksize = 1024
        keys = list(d.keys())
        def work(i):
            return [
                (brickname, d[brickname], read_region(brickname, d[brickname], bricksdesc))
                for brickname in keys[i:i+chunksize]
                ]
        bricks = sum(pool.map(work, range(0, len(keys), chunksize)), [])

    if ns.verbose:
        print('read regions of %d bricks in %g seconds' % (
            len(bricks), time() - t0))

    return bricks

def sweep_schema_ra(nstripes):
    ra = np.linspace(0, 360, nstripes + 1, endpoint=True)
    return [(ra[i], -90, ra[i+1], 90) for i in range(len(ra) - 1)]

def sweep_schema_dec(nstripes):
    dec = np.linspace(-90, 90, nstripes + 1, endpoint=True)
    return [(0, dec[i], 360, dec[i+1]) for i in range(len(dec) - 1)]

def sweep_schema_blocks(nra, ndec):
    ra = np.linspace(0, 360, nra + 1, endpoint=True)
    dec = np.linspace(-90, 90, ndec + 1, endpoint=True)

    return [(ra[i], dec[j], ra[i+1], dec[j+1]) for i in range(len(ra) - 1) for j in range(len(dec) - 1)]

class NA: pass

def make_sweep(sweep, bricks, ns, ALL_DTYPE=None):
    data = [np.empty(0, dtype=ALL_DTYPE)]
    header = {}
    ra1, dec1, ra2, dec2 = sweep
    def merge_header(header, header2):
        for key, value in header2.items():
            if key not in header:
                header[key] = value
            else:
                if header[key] is NA:
                    pass
                else:
                    if header[key] != value:
                        header[key] = NA

    with sharedmem.MapReduce(np=0) as pool:
        def filter(brickname, filename, region):
            if not intersect(sweep, region):
                return None, None
            try:
                objects = fitsio.read(filename, 1, upper=True)
                chunkheader = fitsio.read_header(filename, 0)
            except:
                if ns.ignore_errors:
                    print('IO error on %s' % filename)
                    return None, None
                else:
                    raise
            # ADM check all the column dtypes match.
            if not ns.ignore_errors:
                sflds = SWEEP_DTYPE.fields
                tflds = objects.dtype.fields
                for fld in sflds:
                    sdt, tdt = sflds[fld][0], tflds[fld][0]
                    # ADM handle the case where str_ type is converted
                    # ADM to bytes_ type by fitsio versions < 1.
                    if sdt.char=="S" and tdt.char=='U':
                        sdt = '<U{}'.format(sdt.itemsize)
                    if sdt != tdt:
                        msg = 'sweeps/Tractor dtypes differ for field '
                        msg += '{}. Sweeps: {}, Tractor: {}'.format(fld, sdt, tdt)
                        raise ValueError(msg)

            mask = objects['BRICK_PRIMARY'] != 0
            objects = objects[mask]
            mask = objects['RA'] >= ra1
            mask &= objects['RA'] < ra2
            mask &= objects['DEC'] >= dec1
            mask &= objects['DEC'] < dec2
            objects = objects[mask]

            chunk = np.empty(len(objects), dtype=ALL_DTYPE)

            for colname in chunk.dtype.names:
                if colname not in objects.dtype.names:
                    # skip missing columns
                    continue
                try:
                    chunk[colname][...] = objects[colname][...]
                except ValueError:
                    print('failed on column `%s`' % colname)
                    raise
            chunkheader = dict([(key, chunkheader[key]) for key in chunkheader.keys()])
            return chunk, chunkheader

        def reduce(chunk, chunkheader):
            if chunk is not None:
                data.append(chunk)
                merge_header(header, chunkheader)
        pool.map(filter, bricks, star=True, reduce=reduce)

    neff = len(data) - 1

    data = np.concatenate(data, axis=0)
    header = dict([(key, value) for key, value in header.items() if value is not NA])
    return data, header, neff


def save_sweep_file(filename, data, header, format, unitdict=None):
    # ADM leave the root header unchanged.
    hdr = header.copy()
    if format == 'fits':
        units = None
        # ADM derive the units from the data columns if possible.
        if unitdict is not None:
            units = [unitdict[col] for col in data.dtype.names]

        # ADM add the sweep code version header dependency.
        dep = [int(key.split("DEPNAM")[-1]) for key in hdr.keys()
               if 'DEPNAM' in key]
        if len(dep) == 0:
            nextdep = 0
        else:
            nextdep = np.max(dep) + 1
        hdr["DEPNAM{:02d}".format(nextdep)] = 'gen_sweep'
        hdr["DEPVER{:02d}".format(nextdep)] = git_version()

        hdr = [dict(name=key, value=hdr[key]) for key in sorted(hdr.keys())]
        # ADM write atomically, to a .tmp file, for extra safety.
        with fitsio.FITS(filename+".tmp", mode='rw', clobber=True) as ff:
            ff.create_image_hdu()
            ff[0].write_keys(hdr)
            ff.write_table(data, extname='SWEEP', units=units)
        os.rename(filename+'.tmp', filename)

    elif format == 'hdf5':
        import h5py
        with h5py.File(filename, 'w') as ff:
            dset = ff.create_dataset('SWEEP', data=data)
            for key in hdr:
                dset.attrs[key] = hdr[key]
    else:
        raise ValueError("Unknown format")

def intersect(region1, region2, tol=0.1):
    #  ra1, dec1, ra2, dec2 = region1
    # left top right bottom
    def pad(r):
        # this do not use periodic boundary yet
        return (r[0] - tol, r[1] - tol, r[2] + tol, r[3] + tol)

    region1 = pad(region1)
    region2 = pad(region2)

    dx = min(region1[2], region2[2]) - max(region1[0], region2[0])
    dy = min(region1[3], region2[3]) - max(region1[1], region2[1])
    return (dx > 0) & (dy > 0)

def read_region(brickname, filename, bricksdesc):
    if bricksdesc is not None:
        item = bricksdesc[brickname]
        return item['RA1'], item['DEC1'], item['RA2'], item['DEC2']

    f = fitsio.FITS(filename)
    h = f[0].read_header()
    r = h['RAMIN'], h['DECMIN'], h['RAMAX'], h['DECMAX']
    f.close()
    return r

SWEEP_DTYPE = np.dtype([
# ADM everything IN this list will be written to sweep files
# ADM with the light-curve columns spun off to their own files.
# ADM anything NOT in this list will be written to an "extra" file set.
#   ('BRICK_PRIMARY', '?'),
    ('RELEASE', '>i2'),
    ('BRICKID', '>i4'),
    ('BRICKNAME', 'S8'),
    ('OBJID', '>i4'),
    ('TYPE', 'S3'),
    ('RA', '>f8'),
    ('DEC', '>f8'),
    ('RA_IVAR', '>f4'),
    ('DEC_IVAR', '>f4'),
    ('DCHISQ', '>f4', (5,)),
    ('EBV', '>f4'),
#   ('TYCHO2INBLOB', '?'),
#   ('OUT_OF_BOUNDS', '?'),
#   ('DECAM_FLUX', '>f4', (6,)),
#   ('WISE_FLUX', '>f4', (4,)),
#    ('FLUX_U', '>f4'),
    ('FLUX_G', '>f4'),
    ('FLUX_R', '>f4'),
    ('FLUX_I', '>f4'),
    ('FLUX_Z', '>f4'),
#    ('FLUX_Y', '>f4'),
    ('FLUX_W1', '>f4'),
    ('FLUX_W2', '>f4'),
    ('FLUX_W3', '>f4'),
    ('FLUX_W4', '>f4'),
#   ('DECAM_FLUX_IVAR', '>f4', (6,)),
#   ('WISE_FLUX_IVAR', '>f4', (4,)),
#    ('FLUX_IVAR_U', '>f4'),
    ('FLUX_IVAR_G', '>f4'),
    ('FLUX_IVAR_R', '>f4'),
    ('FLUX_IVAR_I', '>f4'),
    ('FLUX_IVAR_Z', '>f4'),
#    ('FLUX_IVAR_Y', '>f4'),
    ('FLUX_IVAR_W1', '>f4'),
    ('FLUX_IVAR_W2', '>f4'),
    ('FLUX_IVAR_W3', '>f4'),
    ('FLUX_IVAR_W4', '>f4'),
#   ('DECAM_MW_TRANSMISSION', '>f4', (6,)),
#   ('WISE_MW_TRANSMISSION', '>f4', (4,)),
#    ('MW_TRANSMISSION_U', '>f4'),
    ('MW_TRANSMISSION_G', '>f4'),
    ('MW_TRANSMISSION_R', '>f4'),
    ('MW_TRANSMISSION_I', '>f4'),
    ('MW_TRANSMISSION_Z', '>f4'),
#    ('MW_TRANSMISSION_Y', '>f4'),
    ('MW_TRANSMISSION_W1', '>f4'),
    ('MW_TRANSMISSION_W2', '>f4'),
    ('MW_TRANSMISSION_W3', '>f4'),
    ('MW_TRANSMISSION_W4', '>f4'),
#   ('DECAM_NOBS', 'u1', (6,)),
#   ('WISE_NOBS', '>i2', (4,)),
#    ('NOBS_U', '>i2'),
    ('NOBS_G', '>i2'),
    ('NOBS_R', '>i2'),
    ('NOBS_I', '>i2'),
    ('NOBS_Z', '>i2'),
#    ('NOBS_Y', '>i2'),
    ('NOBS_W1', '>i2'),
    ('NOBS_W2', '>i2'),
    ('NOBS_W3', '>i2'),
    ('NOBS_W4', '>i2'),
#   ('DECAM_RCHI2', '>f4', (6,)),
#   ('WISE_RCHI2', '>f4', (4,)),
#    ('RCHISQ_U', '>f4'),
    ('RCHISQ_G', '>f4'),
    ('RCHISQ_R', '>f4'),
    ('RCHISQ_I', '>f4'),
    ('RCHISQ_Z', '>f4'),
#    ('RCHISQ_Y', '>f4'),
    ('RCHISQ_W1', '>f4'),
    ('RCHISQ_W2', '>f4'),
    ('RCHISQ_W3', '>f4'),
    ('RCHISQ_W4', '>f4'),
#   ('DECAM_FRACFLUX', '>f4', (6,)),
#   ('WISE_FRACFLUX', '>f4', (4,)),
#    ('FRACFLUX_U', '>f4'),
    ('FRACFLUX_G', '>f4'),
    ('FRACFLUX_R', '>f4'),
    ('FRACFLUX_I', '>f4'),
    ('FRACFLUX_Z', '>f4'),
#    ('FRACFLUX_Y', '>f4'),
    ('FRACFLUX_W1', '>f4'),
    ('FRACFLUX_W2', '>f4'),
    ('FRACFLUX_W3', '>f4'),
    ('FRACFLUX_W4', '>f4'),
#   ('DECAM_FRACMASKED', '>f4', (6,)),
#    ('FRACMASKED_U', '>f4'),
    ('FRACMASKED_G', '>f4'),
    ('FRACMASKED_R', '>f4'),
    ('FRACMASKED_I', '>f4'),
    ('FRACMASKED_Z', '>f4'),
#    ('FRACMASKED_Y', '>f4'),
#   ('DECAM_FRACIN', '>f4', (6,)),
#    ('FRACIN_U', '>f4'),
    ('FRACIN_G', '>f4'),
    ('FRACIN_R', '>f4'),
    ('FRACIN_I', '>f4'),
    ('FRACIN_Z', '>f4'),
#    ('FRACIN_Y', '>f4'),
#   ('DECAM_ANYMASK', '>i2', (6,)),
#    ('ANYMASK_U', '>i2'),
    ('ANYMASK_G', '>i2'),
    ('ANYMASK_R', '>i2'),
    ('ANYMASK_I', '>i2'),
    ('ANYMASK_Z', '>i2'),
#    ('ANYMASK_Y', '>i2'),
#   ('DECAM_ALLMASK', '>i2', (6,)),
#    ('ALLMASK_U', '>i2'),
    ('ALLMASK_G', '>i2'),
    ('ALLMASK_R', '>i2'),
    ('ALLMASK_I', '>i2'),
    ('ALLMASK_Z', '>i2'),
#    ('ALLMASK_Y', '>i2'),
    ('WISEMASK_W1', 'u1'),
    ('WISEMASK_W2', 'u1'),
#   ('DECAM_PSFSIZE', '>f4', (6,)),
#    ('PSFSIZE_U', '>f4'),
    ('PSFSIZE_G', '>f4'),
    ('PSFSIZE_R', '>f4'),
    ('PSFSIZE_I', '>f4'),
    ('PSFSIZE_Z', '>f4'),
#    ('PSFSIZE_Y', '>f4'),
#   ('DECAM_DEPTH', '>f4', (6,)),
#    ('PSFDEPTH_U', '>f4'),
    ('PSFDEPTH_G', '>f4'),
    ('PSFDEPTH_R', '>f4'),
    ('PSFDEPTH_I', '>f4'),
    ('PSFDEPTH_Z', '>f4'),
#    ('PSFDEPTH_Y', '>f4'),
#   ('DECAM_GALDEPTH', '>f4', (6,)),
#    ('GALDEPTH_U', '>f4'),
    ('GALDEPTH_G', '>f4'),
    ('GALDEPTH_R', '>f4'),
    ('GALDEPTH_I', '>f4'),
    ('GALDEPTH_Z', '>f4'),
#    ('GALDEPTH_Y', '>f4'),
    ('PSFDEPTH_W1', '>f4'),
    ('PSFDEPTH_W2', '>f4'),
    ('WISE_COADD_ID', 'S8'),
    ('LC_FLUX_W1', '>f4', (17,)),
    ('LC_FLUX_W2', '>f4', (17,)),
    ('LC_FLUX_IVAR_W1', '>f4', (17,)),
    ('LC_FLUX_IVAR_W2', '>f4', (17,)),
    ('LC_NOBS_W1', '>i2', (17,)),
    ('LC_NOBS_W2', '>i2', (17,)),
    ('LC_MJD_W1', '>f8', (17,)),
    ('LC_MJD_W2', '>f8', (17,)),
    ('LC_FRACFLUX_W1', '>f4', (17,)),
    ('LC_FRACFLUX_W2', '>f4', (17,)),
    ('LC_RCHISQ_W1', '>f4', (17,)),
    ('LC_RCHISQ_W2', '>f4', (17,)),
    ('LC_EPOCH_INDEX_W1', '>i2', (17,)),
    ('LC_EPOCH_INDEX_W2', '>i2', (17,)),
#    ('FRACDEV', '>f4'),
#    ('FRACDEV_IVAR', '>f4'),
    ('SHAPE_R', '>f4'),
    ('SHAPE_R_IVAR', '>f4'),
    ('SHAPE_E1', '>f4'),
    ('SHAPE_E1_IVAR', '>f4'),
    ('SHAPE_E2', '>f4'),
    ('SHAPE_E2_IVAR', '>f4'),
#    ('SHAPEDEV_R', '>f4'),
#    ('SHAPEDEV_R_IVAR', '>f4'),
#    ('SHAPEDEV_E1', '>f4'),
#    ('SHAPEDEV_E1_IVAR', '>f4'),
#    ('SHAPEDEV_E2', '>f4'),
#    ('SHAPEDEV_E2_IVAR', '>f4'),
#    ('SHAPEEXP_R', '>f4'),
#    ('SHAPEEXP_R_IVAR', '>f4'),
#    ('SHAPEEXP_E1', '>f4'),
#    ('SHAPEEXP_E1_IVAR', '>f4'),
#    ('SHAPEEXP_E2', '>f4'),
#    ('SHAPEEXP_E2_IVAR', '>f4'),
#    ('FIBERFLUX_U', '>f4'),
    ('FIBERFLUX_G', '>f4'),
    ('FIBERFLUX_R', '>f4'),
    ('FIBERFLUX_I', '>f4'),
    ('FIBERFLUX_Z', '>f4'),
#    ('FIBERFLUX_Y', '>f4'),
#    ('FIBERTOTFLUX_U', '>f4'),
    ('FIBERTOTFLUX_G', '>f4'),
    ('FIBERTOTFLUX_R', '>f4'),
    ('FIBERTOTFLUX_I', '>f4'),
    ('FIBERTOTFLUX_Z', '>f4'),
#    ('FIBERTOTFLUX_Y', '>f4'),
    ('REF_CAT', '|S2'),
    ('REF_ID', '>i8'),
    ('REF_EPOCH', '>f4'),
    ('GAIA_PHOT_G_MEAN_MAG', '>f4'),
    ('GAIA_PHOT_G_MEAN_FLUX_OVER_ERROR', '>f4'),
    ('GAIA_PHOT_BP_MEAN_MAG', '>f4'),
    ('GAIA_PHOT_BP_MEAN_FLUX_OVER_ERROR', '>f4'),
    ('GAIA_PHOT_RP_MEAN_MAG', '>f4'),
    ('GAIA_PHOT_RP_MEAN_FLUX_OVER_ERROR', '>f4'),
    ('GAIA_ASTROMETRIC_EXCESS_NOISE', '>f4'),
    ('GAIA_DUPLICATED_SOURCE', '>b1'),
    ('GAIA_PHOT_BP_RP_EXCESS_FACTOR', '>f4'),
    ('GAIA_ASTROMETRIC_SIGMA5D_MAX', '>f4'),
    ('GAIA_ASTROMETRIC_PARAMS_SOLVED', '|u1'),
    ('PARALLAX', '>f4'),
    ('PARALLAX_IVAR', '>f4'),
    ('PMRA', '>f4'),
    ('PMRA_IVAR', '>f4'),
    ('PMDEC', '>f4'),
    ('PMDEC_IVAR', '>f4'),
    ('MASKBITS', '>i4'),
    ('FITBITS', '>i2'),
    ('SERSIC', '>f4'),
    ('SERSIC_IVAR', '>f4')]
)

def parse_args():
    ap = argparse.ArgumentParser(
    description="""Create Sweep files for DECALS.
        This tool ensures each sweep file contains roughly '-n' objects. HDF5 and FITS formats are supported.
        Columns contained in the main sets of sweep file are:
        [%(columns)s]. Light-curve columns in this list are spun off to their own files, as are all remaining columns.
    """ % dict(columns=str(SWEEP_DTYPE.names)),
        )

    ### ap.add_argument("--type", choices=["tractor"], default="tractor", help="Assume a type for src files")

    ap.add_argument("src", help="Path to the root directory contains all tractor files")
    ap.add_argument("dest", help="Path to the Output sweep file")

    ap.add_argument('-f', "--format", choices=['fits'], nargs='+', default=["fits"],
        help="Format of the output sweep files")

    ap.add_argument('-F', "--filelist", default=None,
        help="list of tractor brickfiles to use; this will avoid expensive walking of the path.")

    ap.add_argument('-d', "--bricksdesc", default=None,
        help="location of decals-bricks.fits, speeds up the scanning")

    ap.add_argument('-v', "--verbose", action='store_true')
    ap.add_argument('-m', "--mopup", action='store_true',
        help="if set, don't overwrite existing files (as a speed-up)")
    ap.add_argument('-I', "--ignore-errors", action='store_true')

    ap.add_argument('-S', "--schema", choices=['blocks', 'blocksdr10', 'dec', 'ra'],
            default='blocks',
            help="""Decomposition schema. Still being tuned. """)

    ap.add_argument('-b', "--bricklist",
        help="""Filename with list of bricknames to include.
                If not set, all bricks in src are included, sorted by brickname.
            """)

    ap.add_argument("--numproc", type=int, default=None,
        help="""Number of concurrent processes to use. 0 for sequential execution.
            Default is to use OMP_NUM_THREADS, or the number of cores on the node.""")

    return ap.parse_args()

if __name__ == "__main__":
    main()
