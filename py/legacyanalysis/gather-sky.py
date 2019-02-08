import fitsio
from glob import glob
from astrometry.util.fits import *

def update():
    fns = glob('/global/cscratch1/sd/dstn/dr8sky/calib/decam/splinesky/*/*/decam-*.fits')
    fns.sort()
    print(len(fns), 'splinesky files')
    keys = ['/'.join(fn.split('/')[-5:]) for fn in fns]

    P = fits_table('/global/cscratch1/sd/dstn/dr8/sky-hdrs.fits')
    print('Read', len(P), 'existing results')
    file_indices = dict([(fn.strip(), i) for i,fn in enumerate(P.filename)])

    skeys = ['legpipev', 'plver', 'imgdsum', 'procdate']
    fkeys = ['sig1',
            's_mode', 's_med', 's_cmed', 's_john',
            's_p0', 's_p10', 's_p20', 's_p30', 's_p40', 's_p50',
            's_p60', 's_p70', 's_p80', 's_p90', 's_p100',
            's_fmaskd', 's_fine']

    for j,(k,fn) in enumerate(zip(keys, fns)):
        print(j, 'of', len(fns), fn)
        i = file_indices[k]

        hdr = read_primary_header(fn)
        words = fn.split('/')
        words = words[-1].replace('.fits','').split('-')
        P.camera[i] = words[0]
        P.expnum[i] = int(words[1], 10)
        P.ccdname[i] = words[2]
        for k in skeys:
            getattr(P, k)[i] = hdr[k]
        for k in fkeys:
            P.get(k)[i] = hdr[k]

    P.writeto('/global/cscratch1/sd/dstn/dr8/sky-hdrs-update.fits')


def main():

    fns = glob('/global/cscratch1/sd/dstn/dr8/calib/decam/splinesky/*/*/decam-*.fits')
    #fns = glob('/global/cscratch1/sd/dstn/dr8/calib/decam/splinesky/002*/*/*.fits')
    fns.sort()

    #fns = fns[:1000]

    print(len(fns), 'splinesky files')

    P = fits_table('/global/cscratch1/sd/dstn/dr8/sky-hdrs-resume.fits')
    #columns=['filename'])
    print('Read', len(P), 'existing results')
    donefns = set([fn.strip() for fn in P.filename])

    keys = ['/'.join(fn.split('/')[-5:]) for fn in fns]

    print('Donefns:', list(donefns)[:5])
    print('keys:', keys[:5])

    fns = [full for full,key in zip(fns,keys) if key not in donefns]
    print(len(fns), 'new splinesky filenames to do')

    N = len(fns)
    T = fits_table()
    T.filename = np.array(['/'.join(fn.split('/')[-5:]) for fn in fns])
    T.camera = []
    T.ccdname = []
    T.expnum = np.zeros(N, np.int32)

    skeys = ['legpipev', 'plver', 'imgdsum', 'procdate']
    fkeys = ['sig1',
            's_mode', 's_med', 's_cmed', 's_john',
            's_p0', 's_p10', 's_p20', 's_p30', 's_p40', 's_p50',
            's_p60', 's_p70', 's_p80', 's_p90', 's_p100',
            's_fmaskd', 's_fine']

    for k in skeys:
        setattr(T, k, [])
    for k in fkeys:
        T.set(k, np.zeros(N, np.float32))

    for i,fn in enumerate(fns):
        print(i, 'of', len(fns), fn)
        hdr = read_primary_header(fn)
        words = fn.split('/')
        words = words[-1].replace('.fits','').split('-')
        T.camera.append(words[0])
        T.expnum[i] = int(words[1], 10)
        T.ccdname.append(words[2])

        for k in skeys:
            getattr(T, k).append(hdr[k])
        for k in fkeys:
            T.get(k)[i] = hdr[k]

        if (i+1) % 10000 == 0:
            merge_tables([P,T[:(i+1)]]).writeto('/global/cscratch1/sd/dstn/dr8/sky-hdrs-interim.fits')

    merge_tables([P,T]).writeto('/global/cscratch1/sd/dstn/dr8/sky-hdrs.fits')


def read_primary_header(fn):
    '''
    Reads the FITS primary header (HDU 0) from the given filename.
    This is just a faster version of fitsio.read_header(fn).
    '''
    if fn.endswith('.gz'):
        return fitsio.read_header(fn)

    # Weirdly, this can be MUCH faster than letting fitsio do it...
    hdr = fitsio.FITSHDR()
    foundEnd = False
    ff = open(fn, 'rb')
    h = b''
    while True:
        hnew = ff.read(32768)
        if len(hnew) == 0:
            # EOF
            ff.close()
            raise RuntimeError('Reached end-of-file in "%s" before finding end of FITS header.' % fn)
        h = h + hnew
        while True:
            line = h[:80]
            h = h[80:]
            #print('Header line "%s"' % line)
            # HACK -- fitsio apparently can't handle CONTINUE.
            # It also has issues with slightly malformed cards, like
            # KEYWORD  =      / no value
            if line[:8] != b'CONTINUE':
                try:
                    hdr.add_record(line.decode())
                except OSError as err:
                    print('Warning: failed to parse FITS header line: ' +
                          ('"%s"; error "%s"; skipped' % (line.strip(), str(err))))
                          
            if line == (b'END' + b' '*77):
                foundEnd = True
                break
            if len(h) < 80:
                break
        if foundEnd:
            break
    ff.close()
    return hdr



if __name__ == '__main__':
    #main()
    update()
