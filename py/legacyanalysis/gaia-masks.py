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
from collections import Counter

from desimodel.io import load_tiles
from desimodel.footprint import is_point_in_desi

from astrometry.util.multiproc import multiproc

def main():
    B = fits_table('/global/cfs/cdirs/cosmo/work/legacysurvey/dr11/survey-bricks.fits.gz')
    B.ll,B.bb = radectolb(B.ra, B.dec)
    #I = np.flatnonzero(np.abs(B.bb) > 10)
    I = np.flatnonzero(np.abs(B.bb) >= 8)
    B[I].writeto('bricks-for-gaia.fits')

    # I initially ran |b|<10, but in DR11 we have bricks down to |B|=8.87 deg; add a second set.
    I1 = np.flatnonzero(np.abs(B.bb) > 10)
    I2 = np.flatnonzero((np.abs(B.bb) >= 8) * (np.abs(B.bb) <= 10))
    assert(len(I) == len(I1) + len(I2))

    #BG = B[I]
    #BG = BG[np.argsort(-BG.dec)]

    BG1 = B[I1]
    BG1 = BG1[np.argsort(-BG1.dec)]
    BG2 = B[I2]
    BG2 = BG2[np.argsort(-BG2.dec)]

    survey = LegacySurveyData('/dvs_ro/cfs/cdirs/cosmo/work/legacysurvey/dr11')

    mp = multiproc(16)

    GG = []
    iset = 0
    for BG in [BG1, BG2]:
        while len(BG):
            print('Set', iset)
            N = 10000
            outfn = '/pscratch/sd/d/dstn/gaia-mask-dr11-set%03i.fits' % iset
            if os.path.exists(outfn):
                Gset = fits_table(outfn)
                print('Read', outfn)
                nb = len(set(Gset.brickname))
                if nb != N:
                    print('Warning: file contains', nb, 'bricks, vs', N)
            else:
                print('Creating arg list...')
                args = [(brick,survey) for brick in BG[:N]]
                print('mp.map...')
                it = mp.imap(bounce_one_brick, args)
                Gset = []
                for G in it:
                    if G is not None:
                        Gset.append(G)
                print('Got', len(Gset), 'good bricks')
                Gset = merge_tables(Gset, columns='fillzero')
                Gset.writeto(outfn)
            GG.append(Gset)
            iset += 1
            BG = BG[N:]

    G = merge_tables(GG, columns='fillzero')
    outfn = '/pscratch/sd/d/dstn/gaia-mask-dr11-all.fits'
    G.writeto(outfn)
    print('Wrote', outfn)

    print('Looking up in_desi')
    desitiles = load_tiles()
    G.in_desi = is_point_in_desi(desitiles, G.ra, G.dec)

    # Rename Gaia columns
    gaia_cols = [
        'phot_g_mean_mag',  'phot_g_mean_flux_over_error',  'phot_g_n_obs',
        'phot_bp_mean_mag', 'phot_bp_mean_flux_over_error',
        'phot_rp_mean_mag', 'phot_rp_mean_flux_over_error',
        'astrometric_excess_noise', 'astrometric_excess_noise_sig',
        'duplicated_source',
        'phot_bp_rp_excess_factor',
        'astrometric_sigma5d_max',
        'astrometric_params_solved']
    for col in gaia_cols:
        G.rename(col, 'gaia_'+col)

    # Tidy up...
    cols = ['ra', 'dec', 'ref_cat', 'ref_id', 'ref_epoch', 'mag', 'mask_mag', 'radius', 'radius_pix',
            'pmra', 'pmdec', 'parallax', 'ra_ivar', 'dec_ivar', 'pmra_ivar', 'pmdec_ivar',
            'parallax_ivar', 'in_desi', 'istycho', 'isgaia', 'isbright', 'ismedium', 'pointsource',
            'decam_mag_g', 'decam_mag_r', 'decam_mag_i', 'decam_mag_z', 'zguess',
            'brickname', 'ibx', 'iby',] + ['gaia_'+c for c in gaia_cols]

    outfn = '/pscratch/sd/d/dstn/gaia-mask-dr11.fits'
    G.writeto(outfn, columns=cols)
    print('Wrote', outfn)

def bounce_one_brick(X):
    return one_brick(*X)
    
def one_brick(brick, survey):
    wcs = wcs_for_brick(brick)
    # gaia_margin: don't retrieve sources outside the brick (we'll get them
    # in the neighbouring brick!)
    G,_ = get_reference_sources(survey, wcs, None,
                                star_clusters=False,
                                large_galaxies=False,
                                gaia_margin=0.,
                                galaxy_margin=0.)
    G.cut((G.ra  >= brick.ra1 ) * (G.ra  < brick.ra2) *
          (G.dec >= brick.dec1) * (G.dec < brick.dec2))
    I = np.flatnonzero(np.logical_or(G.isbright, G.ismedium))
    print('Brick %s: %i ref sources, %i for masks, ref_cat %s, bright: %i, medium: %i' %
          (brick.brickname, len(G), len(I), Counter(G.ref_cat).most_common(),
           np.sum(G.isbright), np.sum(G.ismedium)))
    if len(I) == 0:
        return None
    G.cut(I)
    G.brickname = np.array([brick.brickname] * len(G))
    return G

if __name__ == '__main__':
    main()
