import pylab as plt
import numpy as np
from astrometry.util.util import *
from astrometry.util.starutil_numpy import *
from astrometry.util.fits import *
from legacypipe.reference import *
from astrometry.util.util import radecdegtohealpix, healpix_xy_to_ring
from glob import glob
from legacypipe.survey import wcs_for_brick
from legacypipe.reference import read_gaia, get_reference_sources
from legacypipe.survey import LegacySurveyData

from astrometry.util.multiproc import multiproc

def main():

    B = fits_table('/global/cfs/cdirs/cosmo/data/legacysurvey/dr8/survey-bricks.fits.gz')
    B.ll,B.bb = radectolb(B.ra, B.dec)
    I = np.flatnonzero((B.dec > -70) * (np.abs(B.bb) > 10))
    B[I].writeto('bricks-for-gaia.fits')
    BG = B[I]
    BG = BG[np.argsort(-BG.dec)]

# healpixes = set()
# nside = 32
# for r,d in zip(BG.ra,BG.dec):
#     hpxy = radecdegtohealpix(r, d, nside)
#     hpring = healpix_xy_to_ring(hpxy, nside)
#     healpixes.add(hpring)
# hr,hd = [],[]
# for hp in healpixes:
#     hp = healpix_ring_to_xy(hp, nside)
#     r,d = healpix_to_radecdeg(hp, nside, 0.5, 0.5)
#     hr.append(r)
#     hd.append(d)
# plt.plot(hr, hd, 'b.', alpha=0.1);

    survey = LegacySurveyData('/global/cfs/cdirs/cosmo/work/legacysurvey/dr9')

    #BG = BG[:100]
    
    # GG = []
    # for i,brick in enumerate(BG):
    #     G = one_brick(brick, survey)
    #     GG.append(G)

    mp = multiproc(32)

    GG = []
    iset = 0
    while len(BG):
        N = 10000
        outfn = '/global/cscratch1/sd/dstn/gaia-mask-dr9-set%i.fits' % iset
        if os.path.exists(outfn):
            Gset = fits_table(outfn)
            print('Read', outfn)
            nb = len(set(Gset.brickname))
            if nb != N:
                print('Warning: file contains', nb, 'bricks, vs', N)
        else:
            Gset = mp.map(bounce_one_brick, [(brick,survey) for brick in BG[:N]])
            Gset = [G for G in Gset if G is not None]
            Gset = merge_tables(Gset, columns='fillzero')
            Gset.writeto(outfn)
        GG.append(Gset)
        iset += 1
        BG = BG[N:]

    G = merge_tables(GG, columns='fillzero')
    G.writeto('/global/cscratch1/sd/dstn/gaia-mask-dr9.fits')

def bounce_one_brick(X):
    return one_brick(*X)
    
def one_brick(brick, survey):
    wcs = wcs_for_brick(brick)
    # gaia_margin: don't retrieve sources outside the brick (we'll get them
    # in the neighbouring brick!)
    G,_ = get_reference_sources(survey, wcs, 0.262, None,
                                star_clusters=False,
                                gaia_margin=0.,
                                galaxy_margin=0.)
    G.cut((G.ra  >= brick.ra1 ) * (G.ra  < brick.ra2) *
          (G.dec >= brick.dec1) * (G.dec < brick.dec2))
    I = np.flatnonzero(np.logical_or(G.isbright, G.ismedium))
    #print('%i of %i: Brick' % (i+1, len(BG)), brick.brickname, len(G), len(I))
    print('Brick', brick.brickname, len(G), len(I))
    if len(I) == 0:
        return None
    G.cut(I)
    G.brickname = np.array([brick.brickname] * len(G))
    return G

if __name__ == '__main__':
    main()
    
