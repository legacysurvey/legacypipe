import requests
from astrometry.util.fits import *
from astrometry.util.multiproc import *

def get_cell((skycell, subcell)):
    url = 'http://ps1images.stsci.edu/cgi-bin/ps1filenames.py?skycell=%i.%03i&type=stack,stack.wt,stack.mask,stack.exp,stack.num,stack.expwt,stack.psf,stack.mdc,stack.cmf' % (skycell, subcell)
    print('Getting', url)
    r = requests.get(url)
    lines = r.text.split('\n')
    #assert(len(lines) == 6)
    cols = 'projcell subcell ra dec filter mjd type filename shortname'
    assert(lines[0] == cols)
    lines = lines[1:]
    lines = [l.split() for l in lines]
    T = fits_table()
    types = dict(projcell=np.int16, subcell=np.int16, ra=np.float64, dec=np.float64, mjd=None)
    for i,col in enumerate(cols.split()):
        tt = types.get(col, str)
        if tt is None:
            continue
        vals = [words[i] for words in lines]
        #print('Values for', col, ':', vals)
        # HACK -- base-10 parsing for integer subcell
        if col == 'subcell':
            vals = [int(v, 10) for v in vals]
        T.set(col, np.array([tt(v) for v in vals], dtype=tt))
    return T
        
mp = multiproc(8)

TT = []
for skycell in range(635, 2643+1):
    fn = 'ps1skycells-%i.fits' % skycell
    if os.path.exists(fn):
        print('Reading cached', fn)
        TT.append(fits_table(fn))
        continue

    args = []
    for subcell in range(100):
        args.append((skycell, subcell))
    TTi = mp.map(get_cell, args)

    Ti = merge_tables(TTi)
    Ti.writeto(fn)
    TT.extend(TTi)
T = merge_tables(TT)
T.writeto('ps1skycells.fits')
        
