
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
DQ_BITS = dict(badpix=1,
               satur=2,
               interp=4,
               cr=16,
               bleed=64,
               trans=128,
               edge = 256,
               edge2 = 512,
               # Added by our stage_outliers rejection
               outlier = 2048,
    )

# Outlier mask bit values
OUTLIER_POS = 1
OUTLIER_NEG = 2

# Bits in the "maskbits" data product
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
    GALAXY     = 0x1000, # LSLGA large galaxy
    CLUSTER    = 0x2000, # Cluster catalog source
)

# Bits in the "brightblob" bitmask
IN_BLOB = dict(
    BRIGHT  = 0x1,   # "bright" star
    MEDIUM  = 0x2,   # "medium-bright" star
    CLUSTER = 0x4,   # Globular cluster
    GALAXY  = 0x8,   # large LSLGA galaxy
)

