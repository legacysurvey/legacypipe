from legacyzpts import legacy_zeropoints_merge
from astrometry.util.fits import fits_table
import argparse

"""
Merges the ccds-annnotated files and generate a survey-ccds file as an after-burner.
Uses the following naming convention:
    ccds-annotated-<camera>-<drN>.fits
    survey-ccds-<camera>-<drN>.fits
where camera and the DR number are inputs.
Note that there are no checks on whether the input file list consists of annotated files.

M. Landriau, September 2019
"""

if __name__ == "__main__":
    parser0 = argparse.ArgumentParser(description='Generates an annotated CCDs file and a legacypipe-compatible CCDs file from a set of reduced imaging.')
    parser0.add_argument('--file_list',help='List of zeropoint fits files to concatenate')
    parser0.add_argument('--camera', type=str, help='decam, mosaic or 90prime')
    parser0.add_argument('--data_release', type=int, help='Number of LS DR')
    opt = parser0.parse_args()

    fna = "ccds-annotated-"+opt.camera+"-dr"+str(opt.data_release)+".fits"
    fns = fna.replace("ccds-annotated", "survey-ccds")

    arg1 = "--file_list"
    arg2 = "--outname"

    legacy_zeropoints_merge.main([arg1, opt.file_list, arg2, fna])

    # List of keys to cut to
    """
    keys = ['image_filename',
            'image_hdu',
            'camera  ',
            'expnum  ',
            'plver   ',
            'procdate',
            'plprocid',
            'ccdname ',
            'object  ',
            'propid  ',
            'filter  ',
            'exptime ',
            'mjd_obs ',
            'airmass ',
            'fwhm    ',
            'width   ',
            'height  ',
            'ra_bore ',
            'dec_bore',
            'crpix1  ',
            'crpix2  ',
            'crval1  ',
            'crval2  ',
            'cd1_1   ',
            'cd1_2   ',
            'cd2_1   ',
            'cd2_2   ',
            'yshift  ',
            'ra      ',
            'dec     ',
            'skyrms  ',
            'sig1    ',
            'ccdzpt  ',
            'zpt     ',
            'ccdraoff',
            'ccddecoff',
            'ccdskycounts',
            'ccdskysb',
            'ccdrarms',
            'ccddecrms',
            'ccdphrms',
            'ccdnastrom',
            'ccdnphotom',
            'ccd_cuts']
        """
    keys = ['image_filename',
            'image_hdu',
            'camera',
            'expnum',
            'plver',
            'procdate',
            'plprocid',
            'ccdname',
            'object',
            'propid',
            'filter',
            'exptime',
            'mjd_obs',
            'airmass',
            'fwhm',
            'width',
            'height',
            'ra_bore',
            'dec_bore',
            'crpix1',
            'crpix2',
            'crval1',
            'crval2',
            'cd1_1',
            'cd1_2',
            'cd2_1',
            'cd2_2',
            'yshift',
            'ra',
            'dec',
            'skyrms',
            'sig1',
            'ccdzpt',
            'zpt',
            'ccdraoff',
            'ccddecoff',
            'ccdskycounts',
            'ccdskysb',
            'ccdrarms',
            'ccddecrms',
            'ccdphrms',
            'ccdnastrom',
            'ccdnphotom',
            'ccd_cuts']

    t = fits_table(fna)
    t.writeto(fns, columns=keys)


