'''
A tiny script to grab cutouts from the Legacy Survey Image Viewer from a table of targets.

Downloads all the cutouts, and writes an HTML file pointing to them.
'''

from __future__ import print_function
from astrometry.util.fits import fits_table
import os
import numpy as np

if __name__ == '__main__':
    # From Christophe, 2016-06-02, Results.fits
    T = fits_table('mmt.fits')
    T.is_galaxy = (T.id == 4)
    T.is_qso = np.logical_or(T.id == 3, T.id == 30)
    #print(len(T), 'targets')
    #print(sum(T.is_galaxy), 'galaxies')
    #print(sum(T.is_qso), 'quasars')

    urls = []
    # If you don't want to download all the images, you can just print the URLs...
    # but it's faster (and kinder to the server) to download all the images once.
    #print('<html><body>')
    for i in np.flatnonzero(T.is_galaxy):
        url = 'http://legacysurvey.org/viewer/jpeg-cutout-decals-dr2/?ra=%.4f&dec=%.4f&pixscale=0.262&size=100' % (T.ra[i], T.dec[i])
        #print('<img src="%s">' % url)
        urls.append((T.ra[i], T.dec[i], url))
    #print('</body></html>')
    
    outfns = []
    for i,(ra,dec,url) in enumerate(urls):
        outfn = 'gal-%03i.png' % i
        if not os.path.exists(outfn):
            cmd = 'wget --continue -O "%s" "%s"' % (outfn, url)
            print(cmd)
            os.system(cmd)
        outfns.append(outfn)
        
    print('<html><body>')
    for i,((ra,dec,url),fn) in enumerate(zip(urls, outfns)):
        print('<a href="http://legacysurvey.org/viewer/?ra=%.4f&dec=%.4f"><img src="%s"></a>' % (ra, dec, fn))
    print('</body></html>')
