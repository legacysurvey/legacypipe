from astrometry.util.fits import fits_table
import numpy as np

'''
This is a little script for merging Aaron's astrometric offsets into our
WISE tile file.
'''

#/project/projectdirs/cosmo/work/wise/outputs/merge/neo4/fulldepth/fulldepth_neo4_index.fits

W = fits_table('legacypipe/data/wise-tiles.fits')

offsets = fits_table('fulldepth_neo4_index.fits')

off1 = offsets[offsets.band == 1]
off2 = offsets[offsets.band == 2]

name_map_1 = dict([(tile,i) for i,tile in enumerate(off1.coadd_id)])
W.crpix_w1 = off1.crpix[np.array([name_map_1[tile] for tile in W.coadd_id])]
ra  = off1.ra [np.array([name_map_1[tile] for tile in W.coadd_id])]
dec = off1.dec[np.array([name_map_1[tile] for tile in W.coadd_id])]
diff = np.mean(np.hypot(W.ra - ra, W.dec - dec))
print('Mean difference RA,Dec:', diff)

name_map_2 = dict([(tile,i) for i,tile in enumerate(off2.coadd_id)])
W.crpix_w2 = off2.crpix[np.array([name_map_2[tile] for tile in W.coadd_id])]
ra  = off2.ra [np.array([name_map_2[tile] for tile in W.coadd_id])]
dec = off2.dec[np.array([name_map_2[tile] for tile in W.coadd_id])]
diff = np.mean(np.hypot(W.ra - ra, W.dec - dec))
print('Mean difference RA,Dec:', diff)

W.crpix_w1 = W.crpix_w1.astype(np.float32)
W.crpix_w2 = W.crpix_w2.astype(np.float32)

W.writeto('wise-tiles.fits')
