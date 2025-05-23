import os
from legacypipe.ps1cat import HealpixedCatalog
import numpy as np

def gaia_to_decam(gaia, bands,
                  average_color=1.4,
                  color_clip=None,
                  only_color_term=False):
    from functools import reduce
    G  = gaia.phot_g_mean_mag .astype(np.float32)
    BP = gaia.phot_bp_mean_mag.astype(np.float32)
    RP = gaia.phot_rp_mean_mag.astype(np.float32)
    color = BP - RP

    # From Rongpu, 2020-04-12
    # no BP-RP color: use average color
    badcolor = reduce(np.logical_or,
                      [np.logical_not(np.isfinite(color)),
                       BP == 0, RP == 0,])
    color[badcolor] = average_color

    coeffs = {
        'g': ('G', [-0.1178631039, 0.3650113495, 0.5608615360, -0.2850687702,
                -1.0243473939, 1.4378375491, 0.0679401731, -1.1713172509,
                0.9107811975, -0.3374324004, 0.0683946390, -0.0073089582,
                0.0003230170]),
        'r': ('G', [0.1139078673, -0.2868955307, 0.0013196434, 0.1029151074,
                0.1196710702, -0.3729031390, 0.1859874242, 0.1370162451,
                -0.1808580848, 0.0803219195, -0.0180218196, 0.0020584707,
                -0.0000953486]),
        'i': ('G', [0.3396481660, -0.6491867119, -0.3330769819, 0.4381097294,
                0.5752125977, -1.4746570523, 1.2979140762, -0.6371018151,
                0.1948940062, -0.0382055596, 0.0046907449, -0.0003296841,
                0.0000101480]),
        'z': ('G', [0.4811198057, -0.9990015041, 0.1403990019, 0.2150988888,
                -0.2917655866, 0.1326831887, -0.0259205004, 0.0018548776]),
        # IBIS - Arjun
        # 'M411': ('G', [-0.3464, 1.9527,-2.8314, 3.7463,-1.7361, 0.2621]),
        # 'M438': ('G', [-0.1806, 0.8371,-0.2328, 0.6813,-0.3504, 0.0527]),
        # 'M464': ('G', [-0.3263, 1.4027,-1.3349, 1.1068,-0.3669, 0.0424]),
        # 'M490': ('G', [-0.2287, 1.6287,-2.7733, 2.6698,-1.0101, 0.1330]),
        # 'M517': ('G', [-0.1937, 1.2866,-2.4744, 2.7437,-1.1472, 0.1623]),
        'N395': ('G', [ 1.8757,-7.2503,11.9664,-5.8214, 0.9271]),
        # IBIS - Rongpu, 2025-04-07
        'M411': ('G', [-0.121135807, 0.214202972, -0.432275297, 6.33517298, -6.16359698, -10.5837512, 30.6209588, -32.3682815, 18.961989, -6.68321801, 1.41269689, -0.165375832, 0.00825599733]),
        'M438': ('G', [-0.0366575264, 0.148435846, -1.73012913, 7.94291264, -2.59147985, -19.9128805, 36.7698794, -31.5884298, 15.9377805, -4.98555542, 0.954462291, -0.102772045, 0.0047791309]),
        'M464': ('G', [-0.159291847, 0.404280833, 0.216697525, 1.33091203, -1.3403765, -3.65518804, 8.69839989, -8.05791299, 4.14722148, -1.28622435, 0.240118347, -0.0249686426, 0.001115772]),
        'M490': ('G', [-0.0288551549, 0.102359685, 0.413203971, 1.58167708, -3.59089787, -0.815820939, 9.10026694, -11.3422226, 7.04242478, -2.53797638, 0.539870363, -0.0631013162, 0.003132881]),
        'M517': ('G', [-0.118259758, 0.328822667, 1.34465039, -2.35947092, -1.99802997, 7.72242161, -5.60664076, -0.265119231, 2.37946742, -1.39398479, 0.383994131, -0.0532835762, 0.00300229245]),
    }

    base_mag = dict(G=G, BP=BP, RP=RP)
    mags = []
    for b in bands:
        if not b in coeffs:
            mags.append(None)
            continue

        # clip to reasonable range for the polynomial fit
        if color_clip is None and b in ['M411','M438','M464','M490','M517','N395']:
            lo,hi = (0.0, 3.3)
        elif color_clip is None: # and b in ['g','r','i','z']:
            lo,hi = (-0.6, 4.1)
        cc = np.clip(color, lo, hi)

        base, co = coeffs[b]
        mag = base_mag[base].copy()
        # Zero out the mags if we don't have the base mag measured
        # (eg, some stars in EDR3 have G=0(nan), BP=0(nan) but RP~20)
        nomags = (mag == 0.)
        if only_color_term:
            mag[:] = 0.
        for order,c in enumerate(co):
            mag += c * cc**order
        mag[nomags] = 0.
        mags.append(mag)
    return mags

    #  For possible future use:
    #  BASS/MzLS:
    #  coeffs = dict(
    #  g = [-0.1299895823, 0.3120393968, 0.5989482686, 0.3125882487,
    #      -1.9401592247, 1.1011670449, 2.0741304659, -3.3930306403,
    #      2.1857291197, -0.7674676232, 0.1542300648, -0.0167007725,
    #      0.0007573720],
    #  r = [0.0901464643, -0.2463711147, 0.0094963025, -0.1187138789,
    #      0.4131107392, -0.1832183301, -0.6015486252, 0.9802538471,
    #      -0.6613809948, 0.2426395251, -0.0505867727, 0.0056462458,
    #      -0.0002625789],
    #  z = [0.4862049092, -1.0278704657, 0.1220984456, 0.3000129189,
    #      -0.3770662617, 0.1696090596, -0.0331679127, 0.0023867628])

class GaiaCatalog(HealpixedCatalog):
    def __init__(self, file_prefix=None, indexing=None, **kwargs):
        self.gaiadir = os.getenv('GAIA_CAT_DIR')
        if self.gaiadir is None:
            raise ValueError('You must have the GAIA_CAT_DIR environment variable set to point to healpixed Gaia catalogs')
        if indexing is None:
            indexing = os.getenv('GAIA_CAT_SCHEME', 'ring')
        if not indexing in ['nested', 'ring']:
            raise ValueError('Supported values for the GAIA_CAT_SCHEME environment variable or healpix indexing scheme are "nested" or "ring"')
        if file_prefix is None:
            file_prefix = os.getenv('GAIA_CAT_PREFIX', 'chunk')
        #
        fnpattern = os.path.join(self.gaiadir, file_prefix + '-%(hp)05d.fits')
        super(GaiaCatalog, self).__init__(fnpattern, indexing=indexing, **kwargs)

    def get_catalog_radec_box(self, ralo, rahi, declo, dechi):
        import numpy as np

        wrap = False
        if rahi < ralo:
            # wrap-around?
            rahi += 360.
            wrap = True

        # Prepare RA,Dec grid to pick up overlapping healpixes
        rr,dd = np.meshgrid(np.linspace(ralo,  rahi,  2+int(( rahi- ralo)/0.1)),
                            np.linspace(declo, dechi, 2+int((dechi-declo)/0.1)))
        healpixes = set()
        for r,d in zip(rr.ravel(), dd.ravel()):
            healpixes.add(self.healpix_for_radec(r, d))
        # Read catalog in those healpixes
        cat = self.get_healpix_catalogs(healpixes)
        #print('Read', len(cat), 'Gaia catalog entries.  RA range', cat.ra.min(), cat.ra.max(),
        #      'Dec range', cat.dec.min(), cat.dec.max())
        cat.cut((cat.dec >= declo) * (cat.dec <= dechi))
        if wrap:
            cat.cut(np.logical_or(cat.ra <= ralo, cat.ra >= (rahi - 360.)))
        else:
            cat.cut((cat.ra  >= ralo ) * (cat.ra  <= rahi))
        return cat

    @staticmethod
    def catalog_nantozero(gaia):
        gaia.pmra = nantozero(gaia.pmra)
        gaia.pmdec = nantozero(gaia.pmdec)
        gaia.parallax = nantozero(gaia.parallax)
        return gaia

def nantozero(x):
    import numpy as np
    x = x.copy()
    x[np.logical_not(np.isfinite(x))] = 0.
    return x
