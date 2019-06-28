from astrometry.util.fits import *
from astrometry.util.file import *

dr8 = fits_table('/global/cscratch1/sd/djschleg/dr8/DECam-dr8list-ondisk.fits')
print(len(dr8), 'DR8')
dr8.cut(dr8.qkeep == 1)
print(len(dr8), 'to keep')

# expnum, procdate
procdates = dict([(e,d) for e,d in zip(dr8.expnum, dr8.procdate)])

sky = fits_table('/global/cscratch1/sd/dstn/dr8/sky-hdrs.fits')
#sky = fits_table('/global/cscratch1/sd/dstn/dr8/sky-hdrs-update.fits')
print(len(sky), 'sky')

sourcedir = '/global/cscratch1/sd/dstn/dr8/calib/'

extradir  = '/global/cscratch1/sd/dstn/dr8-sky-unneeded/'
olddir    = '/global/cscratch1/sd/dstn/dr8-sky-old/'

redo = []
expnumccd = []
nskip = 0
for i,(e,d,c,fn) in enumerate(zip(sky.expnum, sky.procdate, sky.ccdname, sky.filename)):
    fn = fn.strip()
    if not e in procdates:
        
        # source = os.path.join(sourcedir, fn)
        # dest   = os.path.join(extradir, fn)
        # destdir = os.path.dirname(dest)
        # trymakedirs(destdir)
        # print('Move', source, dest)
        # os.rename(source, dest)
        
        nskip += 1
        continue
    if procdates[e] != d:
        #print('Expnum', e, c, ': sky procdate', d, 'vs DR8 list', procdates[e], fn)

        # source = os.path.join(sourcedir, fn)
        # dest   = os.path.join(olddir, fn)
        # destdir = os.path.dirname(dest)
        # trymakedirs(destdir)
        # print('Move', source, dest)
        # os.rename(source, dest)

        redo.append(i)
        expnumccd.append('%i-%s' % (e, c.strip()))

print(nskip, 'CCDs not in DR8 expnum list')
print(len(redo), 'to redo')

nper = 100

while len(expnumccd) > 0:
    thischunk = expnumccd[:nper]
    expnumccd = expnumccd[nper:]
    #for s in thischunk:
    print(' '.join(thischunk))
