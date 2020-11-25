import pylab as plt
import numpy as np
from astrometry.util.fits import *
from glob import glob
import os
import subprocess
import fitsio
import hashlib

base = '/global/cfs/cdirs/cosmo/work/legacysurvey/dr9m/south/'
outdir = 'fix-wise-psfdepth'
pat = base + 'tractor/*/tractor-*.fits'
print('Searching for', pat)
fns = glob(pat)
fns.sort()
print('Found', len(fns))

badfn = []
for i,fn in enumerate(fns):
    if i % 1000 == 0:
        print(i, fn)
    T = fits_table(fn, columns=['psfdepth_w3', 'psfdepth_w4'])
    if (np.any(np.logical_not(np.isfinite(T.psfdepth_w3))) or 
        np.any(np.logical_not(np.isfinite(T.psfdepth_w4)))):
        print(fn, ': some non-finite psfdepth_w[34]')
        badfn.append(fn)
print('Found', len(badfn), 'bad files')

for fn in badfn:
    brick = fn.split('-')[-1].replace('.fits','')
    print(fn, 'brick', brick)
    chk = base + 'tractor/%s/brick-%s.sha256sum' % (brick[:3], brick)
    cmd = ('(cd %s && grep tractor/.*/tractor-%s.fits %s | sha256sum -c -)' %
           (base, brick, chk))
    print(cmd)
    r = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True, check=True, encoding='utf-8')
    print('sha256sum output:', r.stdout, 'err', r.stderr)
    if not 'OK' in r.stdout:
        raise RuntimeError('checksum fail')
    T = fits_table(fn)

    hdr = T.get_header()
    dirnm = os.path.join(outdir, 'tractor', brick[:3])
    try:
        os.makedirs(dirnm)
    except:
        pass
    outfn = os.path.join(dirnm, 'tractor-%s.fits' % brick)
    T.psfdepth_w3[np.logical_not(np.isfinite(T.psfdepth_w3))] = 0.
    T.psfdepth_w4[np.logical_not(np.isfinite(T.psfdepth_w4))] = 0.

    columns = T.get_columns()
    units = []
    for i,col in enumerate(columns):
        typekey = 'TTYPE%i' % (i+1)
        assert(hdr[typekey].strip() == col)
        unitkey = 'TUNIT%i' % (i+1)
        if unitkey in hdr:
            unit = hdr[unitkey]
        else:
            unit = ''
        units.append(unit)

    # memory-buffer
    fits = fitsio.FITS('mem://', 'rw')

    # We do some messy FITS file manipulation due to a fitsio bug
    # where it fails to preserve the ordering of COMMENT cards...
    cmd = 'fitsgetext -i %s -o %s -e 0' % (fn, outfn)
    subprocess.run(cmd, check=True, shell=True)
    
    T.writeto(None, header=hdr, fits_object=fits, units=units)

    hashfunc = hashlib.sha256
    sha = hashfunc()
    # Read back the data written into memory by the
    # fitsio library
    rawdata = fits.read_raw()
    # close the fitsio file
    fits.close()

    phdr = open(outfn, 'rb').read()
    sha.update(phdr)
    
    sha.update(rawdata[2880:])
    hashcode = sha.hexdigest()
    del sha

    open(outfn, 'ab').write(rawdata[2880:])
    
    outchk = os.path.join(dirnm, 'brick-%s.sha256sum' % brick)
    cmd = ('(grep -v tractor/.*/tractor-%s.fits %s; echo "%s *tractor/%s/tractor-%s.fits") > %s' %
           (brick, chk, hashcode, brick[:3], brick, outchk))
    print(cmd)
    r = subprocess.run(cmd, shell=True, check=True)
