import numpy as np

## These are the mask bits in ANYMASK / ALLMASK.
#
# From: http://www.noao.edu/noao/staff/fvaldes/CPDocPrelim/PL201_3.html
# 1   -- detector bad pixel           InstCal
# 1   -- detector bad pixel/no data   Resampled
# 1   -- No data                      Stacked
# 2   -- saturated                    InstCal/Resampled
# 4   -- interpolated                 InstCal/Resampled
# 16  -- single exposure cosmic ray   InstCal/Resampled
# 64  -- bleed trail                  InstCal/Resampled
# 128 -- multi-exposure transient     InstCal/Resampled
DQ_BITS = dict(badpix  =    1,
               satur   =    2,
               interp  =    4,
               cr      =   16,   # 0x 10
               bleed   =   64,   # 0x 40
               trans   =  128,   # 0x 80
               edge    =  256,   # 0x100
               edge2   =  512,   # 0x200
               # Added by our stage_outliers rejection
               outlier = 2048,   # 0x800
    )

# Bit codes for why a CCD got cut (survey-ccds file, ccd_cuts column)
CCD_CUTS = dict(
    err_legacyzpts = 0x1,
    not_grz = 0x2,
    # DR10 alias
    not_griz = 0x2,
    not_third_pix = 0x4, # Mosaic3 one-third-pixel interpolation problem
    exptime = 0x8,
    ccdnmatch = 0x10,
    zpt_diff_avg = 0x20,
    zpt_small = 0x40,
    zpt_large = 0x80,
    sky_is_bright = 0x100,
    badexp_file = 0x200,
    phrms = 0x400,
    radecrms = 0x800,
    seeing_bad = 0x1000,
    early_decam = 0x2000,
    depth_cut = 0x4000,
    too_many_bad_ccds = 0x8000,
    flagged_in_des = 0x10000,
    phrms_s7 = 0x20000,
    # DR10 late depth-cut
    depth_cut_2 = 0x40000,
    plver = 0x80000,
)

FITBITS = dict(
    FORCED_POINTSOURCE = 0x1,
    FIT_BACKGROUND     = 0x2,
    HIT_RADIUS_LIMIT   = 0x4,
    HIT_SERSIC_LIMIT   = 0x8,
    FROZEN             = 0x10, # all source parameters were frozen at ref-cat values
    BRIGHT             = 0x20,
    MEDIUM             = 0x40,
    GAIA               = 0x80,
    TYCHO2             = 0x100,
    LARGEGALAXY        = 0x200,
    WALKER             = 0x400,
    RUNNER             = 0x800,
    GAIA_POINTSOURCE   = 0x1000,
    ITERATIVE          = 0x2000,
)

# Outlier mask bit values
OUTLIER_POS = 1
OUTLIER_NEG = 2

# Bits in the "maskbits" data product.
# NOTE, however, that a LegacySurveyData may override this
# (eg, to add more bands for ALLMASK_); the most reliable approach
# is to read the FITS headers to find the bit values -- eg
# MB_SAT_G=                    4 / maskbits: g band saturated
# MBIT_2  = 'SATUR_G '           / maskbits bit 2 (0x4): g band saturated
MASKBITS = dict(
    NPRIMARY   = 0x1,   # not PRIMARY
    BRIGHT     = 0x2,
    SATUR_G    = 0x4,
    SATUR_R    = 0x8,
    SATUR_Z    = 0x10,
    ALLMASK_G  = 0x20,
    ALLMASK_R  = 0x40,
    ALLMASK_Z  = 0x80,
    WISEM1     = 0x100, # WISE masked
    WISEM2     = 0x200,
    BAILOUT    = 0x400, # bailed out of processing
    MEDIUM     = 0x800, # medium-bright star
    GALAXY     = 0x1000, # SGA large galaxy
    CLUSTER    = 0x2000, # Cluster catalog source
    SATUR_I    = 0x4000,
    ALLMASK_I  = 0x8000,
    SUB_BLOB   = 0x10000,
    RESOLVED   = 0x20000, # Within a local dwarf galaxy whose stars are resolved
    CLOUDS     = 0x40000, # Within one of the Magellanic Clouds
)

# Bits used internally for reference masks; these fit in a uint8,
# and get transferred to MASKBITS at the end of the pipeline.
REF_MAP_BITS = dict(
    BRIGHT  = 0x1,   # "bright" star
    MEDIUM  = 0x2,   # "medium-bright" star
    CLUSTER = 0x4,   # Globular cluster
    GALAXY  = 0x8,   # large SGA galaxy
    RESOLVED = 0x10, # nearby dwarf galaxy resolved into stars
    CLOUDS   = 0x20, # SMC, LMC
)

MASKBITS_DESCRIPTIONS = [
    ('NPRIMARY',  'not primary brick area'),
    ('BRIGHT',    'bright star nearby'),
    ('SATUR_G',   'g band saturated'),
    ('SATUR_R',   'r band saturated'),
    ('SATUR_Z',   'z band saturated'),
    ('ALLMASK_G', 'any ALLMASK_G bit set'),
    ('ALLMASK_R', 'any ALLMASK_R bit set'),
    ('ALLMASK_Z', 'any ALLMASK_Z bit set'),
    ('WISEM1',    'WISE W1 (all masks)'),
    ('WISEM2',    'WISE W2 (all masks)'),
    ('BAILOUT',   'Bailed out processing'),
    ('MEDIUM',    'medium-bright star'),
    ('GALAXY',    'SGA large galaxy'),
    ('CLUSTER',   'Globular cluster'),
    ('SATUR_I',   'i band saturated'),
    ('ALLMASK_I', 'any ALLMASK_I bit set'),
    ('SUB_BLOB',  'large blobs broken up'),
    ('RESOLVED',  'local dwarf galaxy'),
    ('CLOUDS',    'Magellanic Cloud'),
]

# This is signed, not unsigned, because int32 is a supported FITS data type, uint32 isn't really
maskbits_type = np.int32

FITBITS_DESCRIPTIONS = [
    ('FORCED_POINTSOURCE',  'forced to be PSF'),
    ('FIT_BACKGROUND',      'background levels fit'),
    ('HIT_RADIUS_LIMIT',    'hit radius limit during fit'),
    ('HIT_SERSIC_LIMIT',    'hit Sersic index limit during fit'),
    ('FROZEN',              'parameters were not fit'),
    ('BRIGHT',              'bright star'),
    ('MEDIUM',              'medium-bright star'),
    ('GAIA',                'Gaia source'),
    ('TYCHO2',              'Tycho-2 star'),
    ('LARGEGALAXY',         'SGA large galaxy'),
    ('WALKER',              'fitting moved pos > 1 arcsec'),
    ('RUNNER',              'fitting moved pos > 2.5 arcsec'),
    ('GAIA_POINTSOURCE',    'Gaia source treated as point source'),
    ('ITERATIVE',           'source detected during iterative detection'),
]

# This is how the other bit masks should be written...
WISE_MASK_BITS = [
    (0, 'BRIGHT',  'Bright star core/wings'),
    (1, 'SPIKE',   'PSF-based diffraction spike'),
    (2, 'GHOST',   'Optical ghost'),
    (3, 'LATENT',  'First latent'),
    (4, 'LATENT2', 'Second latent image'),
    (5, 'HALO',    'AllWISE-like circular halo'),
    (6, 'SATUR',   'Bright star saturation'),
    (7, 'SPIKE2',  'Geometric diffraction spike'),
]
