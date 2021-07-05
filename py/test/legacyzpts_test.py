from __future__ import print_function
import os
import numpy as np
from astrometry.util.fits import fits_table
from astrometry.libkd.spherematch import match_radec
from legacyzpts.legacy_zeropoints import main as lzmain
from legacyzpts.merge_annotated import main as mamain
from legacyzpts.update_ccd_cuts import main as upmain
from legacypipe.runbrick import main as rbmain

import astropy.utils.iers
astropy.utils.iers.conf.auto_download = False

def main():
    survey_dir = os.environ['LEGACYPIPE_TEST_DATA']
    lzmain(args=['--survey-dir', survey_dir, '--camera', 'decam', '--image',
            'decam/CP/V4.8.2a/CP20181208/c4d_181209_065355_N23_ooi_g_ls9.fits.fz'])
    lzmain(args=['--survey-dir', survey_dir, '--camera', 'mosaic', '--image',
            'mosaic/CP/V4.4/CP20161016/k4m_161017_115416_CCD3_ooi_zd_ls9.fits.fz'])
    lzmain(args=['--survey-dir', survey_dir, '--camera', '90prime', '--image',
            '90prime/CP/V2.3/CP20170202/ksb_170203_044105_CCD2_ooi_r_ls9.fits.fz'])
    cmd = 'find %s/zpt/decam/ -name "*-ann*" > decam.txt' % survey_dir
    rtn = os.system(cmd)
    assert(rtn == 0)
    cmd = 'find %s/zpt/mosaic/ -name "*-ann*" > mosaic.txt' % survey_dir
    rtn = os.system(cmd)
    assert(rtn == 0)
    cmd = 'find %s/zpt/90prime/ -name "*-ann*" > 90prime.txt' % survey_dir
    rtn = os.system(cmd)
    assert(rtn == 0)

    mamain(args=['--camera', 'decam', '--file_list', 'decam.txt', '--data_release', '100'])
    mamain(args=['--camera', 'mosaic', '--file_list', 'mosaic.txt', '--data_release', '100'])
    mamain(args=['--camera', '90prime', '--file_list', '90prime.txt', '--data_release', '100'])

    upmain(args=['--imlist', 'decam.txt', 'decam',
                 'survey-ccds-decam-dr100.fits', 'ccds-annotated-decam-dr100.fits',
                 os.path.join(survey_dir, 'survey-ccds-decam-test.fits.gz'), 'ccds-annotated-decam-test.fits',
                 '--tilefile', 'legacyzpts/data/decam-tiles_obstatus.fits', '--depth-cut', '--good-ccd-fraction', '0'])
    upmain(args=['--imlist', 'mosaic.txt', 'mosaic',
                 'survey-ccds-mosaic-dr100.fits', 'ccds-annotated-mosaic-dr100.fits',
                 os.path.join(survey_dir, 'survey-ccds-mosaic-test.fits.gz'), 'ccds-annotated-mosaic-test.fits',
                 '--tilefile', 'legacyzpts/data/mosaic-tiles_obstatus.fits', '--depth-cut', '--good-ccd-fraction', '0'])
    upmain(args=['--imlist', '90prime.txt', '90prime',
                 'survey-ccds-90prime-dr100.fits', 'ccds-annotated-90prime-dr100.fits',
                 os.path.join(survey_dir, 'survey-ccds-90prime-test.fits.gz'), 'ccds-annotated-90prime-test.fits',
                 '--depth-cut', '--good-ccd-fraction', '0'])

    rbmain(args=['--brick', '1110p322', '--survey-dir', survey_dir,
                 '--zoom', '1500', '1900', '2500', '2900', '--no-outliers',
                 '--no-wise', '--no-large-galaxies', '--threads', '4'])

    # Cross-match to Gaia and check mags -- this is an end-to-end check on the calibs.
    T = fits_table('tractor/111/tractor-1110p322.fits')
    G = fits_table(os.path.join(os.environ['GAIA_CAT_DIR'], 'chunk-02791.fits'))
    I,J,_ = match_radec(G.ra, G.dec, T.ra, T.dec, 1./3600., nearest=True)
    assert(len(I) == 5)
    K = np.argsort(G.phot_g_mean_mag[I])
    I = I[K]
    J = J[K]
    tmags = -2.5*(np.log10(np.vstack((T.flux_g[J], T.flux_r[J], T.flux_z[J]))) - 9).T
    gmags = np.vstack((G.phot_bp_mean_mag[I], G.phot_g_mean_mag[I], G.phot_rp_mean_mag[I])).T
    dmags = tmags - gmags
    # tmags:
    # array([[12.413096, 12.612314, 17.731936],
    #        [14.27177 , 14.24223 , 15.958183],
    #        [15.427404, 15.12024 , 16.319664],
    #        [21.58627 , 20.125734, 18.591759],
    #        [21.111994, 20.547356, 20.352291]], dtype=float32)
    # gmags:
    # array([[12.404422, 12.11557 , 11.675267],
    #        [14.26786 , 13.915597, 13.40137 ],
    #        [15.367546, 14.982276, 14.428192],
    #        [21.27354 , 19.981428, 18.877398],
    #        [20.961052, 20.722582, 20.46425 ]], dtype=float32)
    # diffs:
    # array([[ 0.009,  0.497,  6.057],
    #        [ 0.004,  0.327,  2.557],
    #        [ 0.060,  0.138,  1.891],
    #        [ 0.313,  0.144, -0.286],
    #        [ 0.151, -0.175, -0.112]], dtype=<U6)

    # Looks like the z-band in particular is pretty bad on the highly saturated stars.

    assert(np.all(np.abs(dmags)[:, :2] < 0.5))
    assert(np.all(np.abs(dmags)[3:, :] < 0.5))

    # Cross-match to Pan-STARRS1 too
    P = fits_table('legacypipe-test-data/ps1-qz-star-v3/ps1-02791.fits')
    I,J,d = match_radec(P.ra, P.dec, T.ra, T.dec, 1.0/3600., nearest=True)
    assert(len(I) == 3)
    K = np.argsort(P.median[I,1])
    I = I[K]
    J = J[K]
    tmags = -2.5*(np.log10(np.vstack((T.flux_g[J], T.flux_r[J], T.flux_z[J]))) - 9).T
    # array([[14.27177 , 14.24223 , 15.958183],
    #        [15.427404, 15.12024 , 16.319664],
    #        [21.58627 , 20.125734, 18.591759]], dtype=float32)
    pmags = P.median[I]
    # array([[14.2699375, 13.848834 , 13.763675 , 13.691722 , 13.696827 ],
    #        [15.400131 , 14.924227 , 14.756025 , 14.679508 , 14.643507 ],
    #        [21.471273 , 20.312788 , 19.22419  , 18.715921 , 18.467852 ]],
    #       dtype=float32)
    dmags = tmags - pmags[:, np.array([0, 1, 3])]
    # array([[ 0.002,  0.393,  2.266],
    #        [ 0.027,  0.196,  1.640],
    #        [ 0.115, -0.187, -0.124]], dtype='<U6')
    assert(np.all(np.abs(dmags[:, :2]) < 0.4))
    assert(np.all(np.abs(dmags[2, :])  < 0.2))

if __name__ == '__main__':
    main()
