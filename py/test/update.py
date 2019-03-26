from glob import glob
import os
import fitsio
from astrometry.util.fits import *

for dirnm in ['test/testcase6',
              'test/testcase4',
              'test/testcase7',
              'test/testcase3',
              ]:
    ccdfns = glob(os.path.join(dirnm, 'survey-ccds*.fits.gz'))
    assert(len(ccdfns) == 1)
    ccdfn = ccdfns[0]
    T = fits_table(ccdfn)
    plprocid = []
    for i,t in enumerate(T):
        fn = os.path.join(dirnm, 'images', t.image_filename.strip())
        hdr = fitsio.read_header(fn)
        pid = hdr['PLPROCID']
        pid = str(pid)
        plprocid.append(pid)

        enum = '%08i' % t.expnum

        for caltype in ['psfex', 'splinesky']:
            fn = os.path.join(dirnm, 'calib', t.camera.strip(), caltype,
                              enum[:5], enum,
                              '%s-%s-%s.fits' % (t.camera.strip(), enum, t.ccdname.strip()))
            F = fitsio.FITS(fn, 'rw')
            F[0].write_key('PLPROCID', pid, comment='CP processing id')
            F.close()
            #cmd = "modhead %s PLPROCID '%s '" % (fn, pid)
            #print(cmd)
            #os.system(cmd)
        
    T.plprocid = np.array(plprocid)
    T.writeto(ccdfn)
