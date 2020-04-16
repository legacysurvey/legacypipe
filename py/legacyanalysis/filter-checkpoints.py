from glob import glob
from astrometry.util.file import *
import numpy as np
from legacypipe.bits import IN_BLOB

indir = '/global/cscratch1/sd/dstn/dr9.3'
outdir = '/global/cscratch1/sd/dstn/dr9.3.1'

fns = glob(os.path.join(indir, 'checkpoints', '*', 'checkpoint-*.pickle'))
fns.sort()
for fn in fns:
    outfn = fn.replace(indir, outdir)
    chk = unpickle_from_file(fn)
    print(len(chk), fn, '->', outfn)
    keep = []
    for c in chk:
        r = c['result']
        if r is None or len(r) == 0:
            keep.append(c)
            continue
        if np.any(r.brightblob & IN_BLOB['BRIGHT']):
            #print('Skipping blob with BRIGHT bits')
            continue
        keep.append(c)
    trymakedirs(outfn, dir=True)
    pickle_to_file(keep, outfn)
    print('Wrote', len(keep), 'of', len(chk), 'to', outfn)
