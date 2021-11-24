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
    os.environ['GAIA_CAT_DIR'] = os.path.join(survey_dir, 'gaia-dr2')
    os.environ['GAIA_CAT_VER'] = '2'
    for k in ['GAIA_CAT_SCHEME','GAIA_CAT_PREFIX']:
        if k in os.environ:
            del os.environ[k]
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
                 '--no-wise', '--no-large-galaxies', '--force-all', #'--no-write',
                 '--threads', '4'])

    # Cross-match to Gaia and check mags -- this is an end-to-end check on the calibs.
    T = fits_table('tractor/111/tractor-1110p322.fits')
    #G = fits_table(os.path.join(survey_dir, 'gaia-dr2', 'chunk-02791.fits'))
    G = fits_table('metrics/111/reference-1110p322.fits')

    I,J,_ = match_radec(G.ra, G.dec, T.ra, T.dec, 1./3600., nearest=True)
    assert(len(I) == 5)
    K = np.argsort(G.phot_g_mean_mag[I])
    I = I[K]
    J = J[K]
    tmags = -2.5*(np.log10(np.vstack((T.flux_g[J], T.flux_r[J], T.flux_z[J]))) - 9).T
    gmags = np.vstack((G.decam_mag_g[I], G.decam_mag_r[I], G.decam_mag_z[I])).T    
    dmags = tmags - gmags

    np.set_printoptions(precision=3, suppress=True)
    print('tmags:')
    print(tmags)
    print('Gaia-decam mags:')
    print(gmags)
    print('dmags:')
    print(dmags)

    # tmags:
    # [[12.386 12.528 17.783]
    #  [14.291 14.576 16.308]
    #  [15.439 15.236 15.976]
    #  [21.586 20.126 18.59 ]
    #  [21.111 20.545 20.351]]
    # Gaia-decam mags:
    # [[12.396 12.05  11.967]
    #  [14.285 13.824 13.666]
    #  [15.404 14.877 14.68 ]
    #  [21.63  20.189 18.632]
    #  [20.866 20.706 20.754]]
    # dmags:
    # [[-0.01   0.478  5.815]
    #  [ 0.006  0.752  2.641]
    #  [ 0.035  0.359  1.296]
    #  [-0.044 -0.063 -0.042]
    #  [ 0.246 -0.161 -0.404]]

    # Looks like the z-band in particular is pretty bad on the highly saturated stars!

    assert(np.all(np.abs(dmags)[:, 0] < 0.25))
    assert(np.all(np.abs(dmags)[:, 1] < 0.8))
    assert(np.all(np.abs(dmags)[3:, :] < 0.5))

    gmags = np.vstack((G.phot_bp_mean_mag[I], G.phot_g_mean_mag[I], G.phot_rp_mean_mag[I])).T
    dmags = tmags - gmags
    print('Gaia mags:')
    print(gmags)
    print('dmags:')
    print(dmags)

    # Gaia mags:
    # [[12.404 12.116 11.675]
    #  [14.268 13.916 13.401]
    #  [15.368 14.982 14.428]
    #  [21.274 19.981 18.877]
    #  [20.961 20.723 20.464]]
    # dmags:
    # [[-0.019  0.413  6.107]
    #  [ 0.023  0.661  2.906]
    #  [ 0.071  0.254  1.548]
    #  [ 0.313  0.145 -0.287]
    #  [ 0.15  -0.177 -0.114]]

    assert(np.all(np.abs(dmags)[:, 0] < 0.6))
    assert(np.all(np.abs(dmags)[:, 1] < 0.7))
    assert(np.all(np.abs(dmags)[3:, :] < 0.6))

    lptdir = os.environ.get('LEGACYPIPE_TEST_DATA', 'legacypipe-test-data')
    # Cross-match to Pan-STARRS1 too
    P = fits_table(os.path.join(lptdir, 'ps1-qz-star-v3/ps1-02791.fits'))
    I,J,_ = match_radec(P.ra, P.dec, T.ra, T.dec, 1.0/3600., nearest=True)
    assert(len(I) == 3)
    K = np.argsort(P.median[I,1])
    I = I[K]
    J = J[K]
    tmags = -2.5*(np.log10(np.vstack((T.flux_g[J], T.flux_r[J], T.flux_z[J]))) - 9).T
    pmags = P.median[I]
    dmags = tmags - pmags[:, np.array([0, 1, 3])]
    print('tmags:')
    print(tmags)
    print('Pan-STARRS mags (grz):')
    print(pmags)
    print('dmags:')
    print(dmags)

    # tmags:
    # [[14.291 14.576 16.308]
    #  [15.439 15.236 15.976]
    #  [21.586 20.126 18.59 ]]
    # Pan-STARRS mags (grz):
    # [[14.27  13.849 13.764 13.692 13.697]
    #  [15.4   14.924 14.756 14.68  14.644]
    #  [21.471 20.313 19.224 18.716 18.468]]
    # dmags:
    # [[ 0.021  0.727  2.616]
    #  [ 0.038  0.312  1.297]
    #  [ 0.115 -0.187 -0.126]]

    # g,r mags
    assert(np.all(np.abs(dmags[:, 0]) < 0.2))
    assert(np.all(np.abs(dmags[1:, 1]) < 0.4))
    # final star
    assert(np.all(np.abs(dmags[2, :])  < 0.2))

if __name__ == '__main__':
    main()
