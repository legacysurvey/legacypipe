from __future__ import print_function
from glob import glob
import os

if __name__ == '__main__':
    #basedir = '/global/projecta/projectdirs/cosmo/work/dr4c'
    basedir = '/global/projecta/projectdirs/cosmo/work/dr4_backup/brick_sha1sums'

    outdir = 'sha1sums'

    pat = os.path.join(basedir, '*', 'brick*.sha*')
    #pat = os.path.join(basedir, 'tractor', '*', 'brick*.sha*')
    #pat = os.path.join(basedir, 'tractor', '055', 'brick*.sha*')
    print('Pattern:', pat)
    fns = glob(pat)
    fns.sort()
    print('Found', len(fns), 'files')

    dirs = {}

    for fn in fns:
        print('Reading', fn)
        for line in open(fn).readlines():
            words = line.strip().split(' ')
            #print('Line', line, '-> words', words)
            sha,path = words

            dirnm = os.path.dirname(path)
            filenm = os.path.basename(path)
            if not dirnm in dirs:
                dirs[dirnm] = [(filenm, sha)]
            else:
                dirs[dirnm].append((filenm, sha))

    for dirnm,files in dirs.items():
        print('Directory', dirnm)
        #print('  files:', files)
        shadir = os.path.join(outdir, dirnm)
        os.makedirs(shadir)
        shafn = os.path.join(shadir, 'legacysurvey_dr4_' + dirnm.replace('/','_') + '.sha1sum')
        f = open(shafn, 'w')
        for fn,sha in files:
            f.write('%s  %s\n' % (sha, fn))
        f.close()
        print('  wrote', shafn)

