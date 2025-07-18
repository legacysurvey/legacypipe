if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import os
import sys
import time
import tempfile
import numpy as np
import fitsio
from legacypipe.runbrick import main
from astrometry.util.fits import fits_table

def reset_env():
    for v in ['UNWISE_COADDS_TIMERESOLVED_DIR', 'SKY_TEMPLATE_DIR',
              'LARGEGALAXIES_CAT', 'GAIA_CAT_DIR', 'TYCHO2_KD_DIR',
              'UNWISE_MODEL_SKY_DIR']:
        if v in os.environ:
            del os.environ[v]

def set_ref_env(surveydir):
    os.environ['TYCHO2_KD_DIR'] = surveydir
    os.environ['GAIA_CAT_DIR'] = os.path.join(surveydir, 'gaia')
    os.environ['GAIA_CAT_VER'] = '2'
    os.environ['GAIA_CAT_PREFIX'] = 'chunk'
    os.environ['GAIA_CAT_SCHEME'] = 'ring'

def rbmain():
    from legacypipe.catalog import read_fits_catalog
    from legacypipe.survey import LegacySurveyData, wcs_for_brick
    from tractor.galaxy import DevGalaxy
    from tractor import PointSource, Catalog
    from tractor import GaussianMixturePSF
    from legacypipe.survey import BrickDuck
    from legacypipe.forced_photom import main as forced_main
    from astrometry.util.file import trymakedirs
    import shutil

    ceres  = 'ceres'  in sys.argv
    psfex  = 'psfex'  in sys.argv

    reset_env()

    test_4(ceres, psfex)
    test_4b()
    sys.exit(0)
    
    oldargs = sys.argv
    sys.argv = [sys.argv[0]]
    main()
    sys.argv = oldargs

    # Test create_kdtree and (reading CCD kd-tree)!
    indir = os.path.join(os.path.dirname(__file__), 'testcase6')
    with tempfile.TemporaryDirectory() as surveydir:
        files = ['calib', 'gaia', 'images', 'survey-bricks.fits.gz',
                 'tycho2.kd.fits']
        for fn in files:
            src = os.path.join(indir, fn)
            dst = os.path.join(surveydir, fn)
            #trymakedirs(dst, dir=True)
            print('Copy', src, dst)
            if os.path.isfile(src):
                shutil.copy(src, dst)
            else:
                shutil.copytree(src, dst)

        from legacypipe.create_kdtrees import create_kdtree
        infn = os.path.join(indir, 'survey-ccds-1.fits.gz')
        outfn = os.path.join(surveydir, 'survey-ccds-1.kd.fits')
        create_kdtree(infn, outfn, False)

        os.environ['TYCHO2_KD_DIR'] = surveydir
        outdir = 'out-testcase6-kd'
        main(args=['--brick', '1102p240', '--zoom', '500', '600', '650', '750',
                   '--force-all', '--no-write', '--no-wise', '--no-gaia',
                   '--survey-dir', surveydir,
                   '--outdir', outdir])
        fn = os.path.join(outdir, 'tractor', '110', 'tractor-1102p240.fits')
        assert(os.path.exists(fn))
        T = fits_table(fn)
        assert(len(T) == 2)
        # Since there is a Tycho-2 star in the blob, forced to be PSF.
        assert(T.type[0].strip() == 'PSF')
        assert(T.type[1].strip() == 'PSF')
        # There is a Tycho-2 star in the blob.
        I = np.flatnonzero(T.ref_cat == 'T2')
        assert(len(I) == 1)
        assert(T.ref_id[I][0] == 1909016711)

        cat = read_fits_catalog(T)
        assert(len(cat) == 2)
        assert(isinstance(cat[0], PointSource))
        assert(isinstance(cat[1], PointSource))
        cat,ivs = read_fits_catalog(T, invvars=True)
        assert(len(cat) == 2)
        assert(isinstance(cat[0], PointSource))
        assert(isinstance(cat[1], PointSource))
        cat2 = Catalog(*cat)
        assert(len(ivs) == len(cat2.getParams()))

        # test --fit-on-coadds
        outdir = 'out-testcase6-coadds'
        main(args=['--brick', '1102p240', '--zoom', '500', '600', '650', '750',
                   '--force-all', '--no-write', '--no-wise', '--no-gaia',
                   '--survey-dir', surveydir, '--fit-on-coadds',
                   '--outdir', outdir])

        fn = os.path.join(outdir, 'tractor', '110', 'tractor-1102p240.fits')
        assert(os.path.exists(fn))
        T = fits_table(fn)
        assert(len(T) == 2)
        # Since there is a Tycho-2 star in the blob, forced to be PSF.
        assert(T.type[0].strip() == 'PSF')
        assert(T.type[1].strip() == 'PSF')
        # There is a Tycho-2 star in the blob.
        I = np.flatnonzero(T.ref_cat == 'T2')
        assert(len(I) == 1)
        assert(T.ref_id[I][0] == 1909016711)
        del os.environ['TYCHO2_KD_DIR']

        # test --skip-coadds
        r = main(args=['--brick', '1102p240',
                       '--zoom', '500', '600', '650', '750',
                       '--force-all', '--no-write', '--no-wise', '--no-gaia',
                       '--survey-dir', surveydir,
                       '--outdir', outdir, '--skip-coadd'])
        assert(r == 0)

        # test --skip
        r = main(args=['--brick', '1102p240',
                       '--zoom', '500', '600', '650', '750',
                       '--force-all', '--no-write', '--no-wise', '--no-gaia',
                       '--survey-dir', surveydir,
                       '--outdir', outdir, '--skip'])
        assert(r == 0)

        # NothingToDoError (neighbouring brick)
        r = main(args=['--brick', '1102p240', '--zoom', '0','100','0','100',
                       '--force-all', '--no-write', '--no-wise', '--no-gaia',
                       '--survey-dir', surveydir,
                       '--outdir', outdir])
        assert(r == 0)

    surveydir = os.path.join(os.path.dirname(__file__), 'testcase9')

    # Test for some get_tractor_image kwargs
    survey = LegacySurveyData(surveydir)
    fakebrick = BrickDuck(9.1228, 3.3975, 'quack')
    wcs = wcs_for_brick(fakebrick, W=100, H=100)
    ccds = survey.ccds_touching_wcs(wcs)
    ccd = ccds[0]
    im = survey.get_image_object(ccd)
    H,W = wcs.shape
    targetrd = np.array([wcs.pixelxy2radec(x,y) for x,y in
                         [(1,1),(W,1),(W,H),(1,H),(1,1)]])
    tim = im.get_tractor_image(radecpoly=targetrd)
    assert(tim.getImage() is not None)
    assert(tim.getInvError() is not None)
    assert(tim.dq is not None)
    tim2 = im.get_tractor_image(radecpoly=targetrd, pixels=False)
    assert(np.all(tim2.getImage() == 0.))
    tim4 = im.get_tractor_image(radecpoly=targetrd, invvar=False)
    u = np.unique(tim4.inverr)
    # inverr pixels can be zeroed out
    assert((len(u) == 1) or (len(u) == 2))
    u = u[-1]
    target = tim4.zpscale / tim4.sig1
    assert(np.abs(u / target - 1.) < 0.001)
    tim3 = im.get_tractor_image(radecpoly=targetrd, invvar=False, dq=False)
    assert(not hasattr(tim3, 'dq'))
    tim5 = im.get_tractor_image(radecpoly=targetrd, gaussPsf=True)
    print(tim5.getPsf())
    assert(isinstance(tim5.getPsf(), GaussianMixturePSF))

    surveydir = os.path.join(os.path.dirname(__file__), 'testcase12')
    set_ref_env(surveydir)
    os.environ['UNWISE_MODEL_SKY_DIR'] = os.path.join(surveydir, 'images', 'unwise-mod')
    #python legacypipe/runbrick.py --radec  --width 100 --height 100 --outdir dup5b --survey-dir test/testcase12 --force-all --no-wise
    unwdir = os.path.join(surveydir, 'images', 'unwise')

    main(args=['--radec', '346.684', '12.791', '--width', '100',
               '--height', '100', '--no-wise-ceres',
               '--unwise-dir', unwdir, '--survey-dir', surveydir,
               '--outdir', 'out-testcase12', '--skip-coadd', '--force-all',
               '--stage', 'image_coadds'])

    # Check that the dither_seed gets set the same each time.
    hdr = fitsio.read_header('out-testcase12/coadd/cus/custom-346684p12791/legacysurvey-custom-346684p12791-image-r.fits.fz', ext=1)
    dither = hdr['ZDITHER0']
    #assert(dither == 1652)

    main(args=['--radec', '346.684', '12.791', '--width', '100',
               '--height', '100', '--no-wise-ceres',
               '--unwise-dir', unwdir, '--survey-dir', surveydir,
               '--outdir', 'out-testcase12', '--skip-coadd', '--force-all'])

    # --plots for stage_wise_forced
    main(args=['--radec', '346.684', '12.791', '--width', '100',
               '--height', '100', '--no-wise-ceres',
               '--unwise-dir', unwdir, '--survey-dir', surveydir,
               '--outdir', 'out-testcase12', '--stage', 'wise_forced',
               '--plots'])
    reset_env()

    M = fitsio.read('out-testcase12/coadd/cus/custom-346684p12791/legacysurvey-custom-346684p12791-maskbits.fits.fz')
    # Count masked & unmasked bits (the cluster splits this 100x100 field)
    from collections import Counter
    c = Counter(M.ravel())
    from legacypipe.bits import MASKBITS
    assert(c[0] >= 4000)
    assert(c[MASKBITS['CLUSTER']] >= 4000)

    surveydir = os.path.join(os.path.dirname(__file__), 'testcase9')
    set_ref_env(surveydir)
    os.environ['LARGEGALAXIES_CAT'] = os.path.join(surveydir,
                                                   'sga-sub.kd.fits')
    main(args=['--radec', '9.1228', '3.3975', '--width', '100',
               '--height', '100', '--old-calibs-ok', '--no-wise-ceres',
               '--no-wise', '--survey-dir', surveydir,
               '--outdir', 'out-testcase9', '--skip', '--force-all',
               '--ps', 'tc9-ps.fits', '--ps-t0', str(int(time.time()))])
    # (omit --force-all --no-write... reading from pickles below!)

    # Test with --apodize
    main(args=['--radec', '9.1228', '3.3975', '--width', '100',
               '--height', '100', '--old-calibs-ok', '--no-wise',
               '--force-all', '--no-write', '--survey-dir', surveydir,
               '--outdir', 'out-testcase9-ap', '--apodize'])

    main(args=['--radec', '9.1228', '3.3975', '--width', '100',
               '--height', '100', '--old-calibs-ok', '--no-wise-ceres',
               '--no-wise', '--survey-dir',
               surveydir, '--outdir', 'out-testcase9',
               '--plots', '--stage', 'halos'])

    main(args=['--radec', '9.1228', '3.3975', '--width', '100',
               '--height', '100', '--old-calibs-ok', '--no-wise-ceres',
               '--no-wise', '--survey-dir',
               surveydir, '--outdir', 'out-testcase9-coadds',
               '--stage', 'image_coadds', '--blob-image'])

    T = fits_table('out-testcase9/tractor/cus/tractor-custom-009122p03397.fits')
    assert(len(T) == 4)
    # Gaia star becomes a DUP!
    assert(np.sum([t == 'DUP' for t in T.type]) == 1)
    # LSLGA galaxy exists!
    Igal = np.flatnonzero([r == 'L3' for r in T.ref_cat])
    assert(len(Igal) == 1)
    assert(np.all(T.ref_id[Igal] > 0))
    assert(T.type[Igal[0]] == 'SER')

    # --brick and --zoom rather than --radec --width --height
    main(args=['--survey-dir', surveydir, '--outdir', 'out-testcase9b',
               '--zoom', '1950', '2050', '340', '440', '--brick', '0091p035', '--force-all'])

    # test forced phot??
    shutil.copy('test/testcase9/survey-bricks.fits.gz', 'out-testcase9b')

    forced_main(args=['--survey-dir', surveydir,
                      '--no-ceres',
                      '--catalog-dir', 'out-testcase9b',
                      '--expnum', '372546', '--ccdname', 'N26', '--out', 'forced1.fits'])
    assert(os.path.exists('forced1.fits'))
    _ = fits_table('forced1.fits')
    # ... more tests...!

    forced_main(args=['--survey-dir', surveydir,
                      '--no-ceres',
                      '--catalog-dir', 'out-testcase9b',
                      '--derivs', '--threads', '2',
                      '--apphot',
                      '--expnum', '372547', '--ccdname', 'N26', '--out', 'forced2.fits'])
    assert(os.path.exists('forced2.fits'))
    _ = fits_table('forced2.fits')

    forced_main(args=['--survey-dir', surveydir,
                      '--no-ceres',
                      '--catalog-dir', 'out-testcase9b',
                      '--agn',
                      '--expnum', '257266', '--ccdname', 'S21', '--out', 'forced3.fits'])
    assert(os.path.exists('forced3.fits'))
    _ = fits_table('forced3.fits')

    if ceres:
        forced_main(args=['--survey-dir', surveydir,
                          '--catalog-dir', 'out-testcase9b',
                          '--derivs', '--threads', '2',
                          '--apphot',
                          '--expnum', '372546', '--ccdname', 'N26', '--out', 'forced4.fits'])
        assert(os.path.exists('forced4.fits'))
        _ = fits_table('forced4.fits')

    # Test cache_dir
    with tempfile.TemporaryDirectory() as cachedir, \
        tempfile.TemporaryDirectory() as tempsurveydir:
        files = []
        for dirpath, _, filenames in os.walk(surveydir):
            for fn in filenames:
                path = os.path.join(dirpath, fn)
                relpath = os.path.relpath(path, surveydir)
                files.append(relpath)
        # cache or no?
        files.sort()
        files_cache = files[::2]
        files_nocache = files[1::2]
        # Survey-ccds *must* be in nocache.
        fn = 'survey-ccds-1.kd.fits'
        if fn in files_cache:
            files_cache.remove(fn)
            files_nocache.append(fn)

        for fn in files_cache:
            src = os.path.join(surveydir, fn)
            dst = os.path.join(cachedir, fn)
            trymakedirs(dst, dir=True)
            print('Copy', src, dst)
            shutil.copy(src, dst)

        for fn in files_nocache:
            src = os.path.join(surveydir, fn)
            dst = os.path.join(tempsurveydir, fn)
            trymakedirs(dst, dir=True)
            print('Copy', src, dst)
            shutil.copy(src, dst)

        main(args=['--radec', '9.1228', '3.3975', '--width', '100',
                   '--height', '100', '--no-wise',
                   '--survey-dir', tempsurveydir,
                   '--cache-dir', cachedir,
                   '--outdir', 'out-testcase9cache', '--force-all'])

    reset_env()

    # if ceres:
    #     surveydir = os.path.join(os.path.dirname(__file__), 'testcase3')
    #     main(args=['--brick', '2447p120', '--zoom', '1020', '1070', '2775', '2815',
    #                '--no-wise', '--force-all', '--no-write', '--ceres',
    #                '--survey-dir', surveydir,
    #                '--outdir', 'out-testcase3-ceres',
    #                '--no-depth-cut'])

    # MzLS + BASS data
    # python legacypipe/runbrick.py --run north --brick 1773p595 --zoom 1300 1500 700 900 --survey-dir dr9-north -s coadds
    # fitscopy coadd/177/1773p595/legacysurvey-1773p595-ccds.fits"[#row<3 || #row==12]" cx.fits
    # python legacyanalysis/create_testcase.py cx.fits test/mzlsbass2 1773p595 --survey-dir dr9-north/ --fpack
    surveydir2 = os.path.join(os.path.dirname(__file__), 'mzlsbass2')
    set_ref_env(surveydir2)
    main(args=['--brick', '1773p595', '--zoom', '1300', '1500', '700', '900',
               '--no-wise', '--force-all', '--no-write',
               '--survey-dir', surveydir2,
               '--outdir', 'out-mzlsbass2'])

    T = fits_table('out-mzlsbass2/tractor/177/tractor-1773p595.fits')
    assert(np.sum(T.ref_cat == 'G2') == 3)
    assert(np.sum(T.ref_id > 0) == 3)

    # Test --max-blobsize, --checkpoint, --bail-out

    outdir = 'out-mzlsbass2b'
    chk = 'checkpoint-mzb2b.p'
    if os.path.exists(chk):
        os.unlink(chk)
    main(args=['--brick', '1773p595', '--zoom', '1300', '1500', '700', '900',
               '--no-wise', '--force-all', '--stage', 'fitblobs',
               '--write-stage', 'srcs',
               '--survey-dir', surveydir2,
               '--outdir', outdir,
               '--checkpoint', chk,
               '--nblobs', '3'])
    # err... --max-blobsize does not result in bailed-out blobs masked,
    # because it treats large blobs as *completed*...
    #'--max-blobsize', '3000',

    outdir = 'out-mzlsbass2c'
    main(args=['--brick', '1773p595', '--zoom', '1300', '1500', '700', '900',
               '--no-wise', '--force-all',
               '--survey-dir', surveydir2,
               '--outdir', outdir, '--bail-out', '--checkpoint', chk,
               '--no-write'])

    reset_env()

    M = fitsio.read(os.path.join(outdir, 'coadd', '177', '1773p595',
                                 'legacysurvey-1773p595-maskbits.fits.fz'))
    assert(np.sum((M & MASKBITS['BAILOUT'] ) > 0) >= 1000)

    # Test RexGalaxy

    surveydir = os.path.join(os.path.dirname(__file__), 'testcase6')
    outdir = 'out-testcase6-rex'
    the_args = ['--brick', '1102p240', '--zoom', '500', '600', '650', '750',
               '--force-all', '--no-write', '--no-wise',
                '--skip-calibs',
    #'--rex', #'--plots',
               '--survey-dir', surveydir,
               '--outdir', outdir]
    print('python legacypipe/runbrick.py', ' '.join(the_args))
    set_ref_env(surveydir)
    main(args=the_args)
    fn = os.path.join(outdir, 'tractor', '110', 'tractor-1102p240.fits')
    assert(os.path.exists(fn))
    T = fits_table(fn)
    assert(len(T) == 2)
    print('Types:', T.type)
    # Since there is a Tycho-2 star in the blob, forced to be PSF.
    assert(T.type[0].strip() == 'PSF')
    cmd = ('(cd %s && sha256sum -c %s)' %
           (outdir, os.path.join('tractor', '110', 'brick-1102p240.sha256sum')))
    print(cmd)
    rtn = os.system(cmd)
    assert(rtn == 0)

    # Test with a Tycho-2 star in the blob.

    surveydir = os.path.join(os.path.dirname(__file__), 'testcase6')
    outdir = 'out-testcase6'
    main(args=['--brick', '1102p240', '--zoom', '500', '600', '650', '750',
               '--force-all', '--no-write', '--no-wise',
               '--survey-dir', surveydir,
               '--outdir', outdir])
    fn = os.path.join(outdir, 'tractor', '110', 'tractor-1102p240.fits')
    assert(os.path.exists(fn))
    T = fits_table(fn)
    assert(len(T) == 2)
    print('Types:', T.type)
    # Since there is a Tycho-2 star in the blob, forced to be PSF.
    assert(T.type[0].strip() == 'PSF')
    reset_env()

    test_4(ceres, psfex)

    # Wrap-around, hybrid PSF
    surveydir = os.path.join(os.path.dirname(__file__), 'testcase8')
    outdir = 'out-testcase8'
    set_ref_env(surveydir)

    main(args=['--brick', '1209p050', '--zoom', '720', '1095', '3220', '3500',
               '--force-all', '--no-write', '--no-wise', #'--plots',
               '--survey-dir', surveydir,
               '--outdir', outdir])

    # Test with a Tycho-2 star + another saturated star in the blob.

    surveydir = os.path.join(os.path.dirname(__file__), 'testcase7')
    outdir = 'out-testcase7'
    set_ref_env(surveydir)
    main(args=['--brick', '1102p240', '--zoom', '250', '350', '1550', '1650',
               '--force-all', '--no-write', '--no-wise', #'--plots',
               '--survey-dir', surveydir,
               '--outdir', outdir])
    reset_env()
    fn = os.path.join(outdir, 'tractor', '110', 'tractor-1102p240.fits')
    assert(os.path.exists(fn))
    T = fits_table(fn)
    assert(len(T) == 4)

    # Check skipping blobs outside the brick's unique area.
    # (this now doesn't detect any sources at all, reasonably)
    # surveydir = os.path.join(os.path.dirname(__file__), 'testcase5')
    # outdir = 'out-testcase5'
    #
    # fn = os.path.join(outdir, 'tractor', '186', 'tractor-1867p255.fits')
    # if os.path.exists(fn):
    #     os.unlink(fn)
    #
    # main(args=['--brick', '1867p255', '--zoom', '0', '150', '0', '150',
    #            '--force-all', '--no-write', '--coadd-bw',
    #            '--survey-dir', surveydir,
    #            '--early-coadds',
    #            '--outdir', outdir] + extra_args)
    #
    # assert(os.path.exists(fn))
    # T = fits_table(fn)
    # assert(len(T) == 1)

    test_4b()
    
    surveydir = os.path.join(os.path.dirname(__file__), 'testcase3')
    outdir = 'out-testcase3'
    set_ref_env(surveydir)
    checkpoint_fn = os.path.join(outdir, 'checkpoint.pickle')
    if os.path.exists(checkpoint_fn):
        os.unlink(checkpoint_fn)
    main(args=['--brick', '2447p120', '--zoom', '1020', '1070', '2775', '2815',
               '--no-wise', '--force-all', '--no-write',
               '--survey-dir', surveydir,
               '--outdir', outdir,
               '--checkpoint', checkpoint_fn,
               '--checkpoint-period', '1',
               '--threads', '2'])

    # Read catalog into Tractor sources to test read_fits_catalog
    survey = LegacySurveyData(survey_dir=outdir)
    fn = survey.find_file('tractor', brick='2447p120')
    print('Checking', fn)
    T = fits_table(fn)
    cat = read_fits_catalog(T)
    print('Read catalog:', cat)

    assert(len(cat) == 2)
    src = cat[1]
    print('Source0', src)
    from tractor.sersic import SersicGalaxy
    assert(type(src) in [DevGalaxy, SersicGalaxy])
    assert(np.abs(src.pos.ra  - 244.77973) < 0.00001)
    assert(np.abs(src.pos.dec -  12.07234) < 0.00002)
    src = cat[0]
    print('Source1', src)
    assert(type(src) ==  PointSource)
    assert(np.abs(src.pos.ra  - 244.77828) < 0.00001)
    assert(np.abs(src.pos.dec -  12.07250) < 0.00001)

    # Check that we can run again, using that checkpoint file.
    main(args=['--brick', '2447p120', '--zoom', '1020', '1070', '2775', '2815',
               '--no-wise', '--force-all', '--no-write',
               '--survey-dir', surveydir,
               '--outdir', outdir,
               '--checkpoint', checkpoint_fn,
               '--checkpoint-period', '1',
               '--threads', '2'])
    # Assert...... something?

    # Test --checkpoint without --threads
    main(args=['--brick', '2447p120', '--zoom', '1020', '1070', '2775', '2815',
               '--no-wise', '--force-all', '--no-write',
               '--survey-dir', surveydir,
               '--outdir', outdir,
               '--checkpoint', checkpoint_fn,
               '--checkpoint-period', '1' ])

    # From Kaylan's Bootes pre-DR4 run
    # surveydir2 = os.path.join(os.path.dirname(__file__), 'mzlsbass3')
    # main(args=['--brick', '2173p350', '--zoom', '100', '200', '100', '200',
    #            '--no-wise', '--force-all', '--no-write',
    #            '--survey-dir', surveydir2,
    #            '--outdir', 'out-mzlsbass3'] + extra_args)

    # With plots!
    main(args=['--brick', '2447p120', '--zoom', '1020', '1070', '2775', '2815',
               '--no-wise', '--force-all', '--no-write',
               '--survey-dir', surveydir,
               '--outdir', 'out-testcase3', '--plots',
               '--nblobs', '1'])

    # Decals Image Simulations
    # Uncomment WHEN galsim build for Travis
    #os.environ["DECALS_SIM_DIR"]= os.path.join(os.path.dirname(__file__),'image_sims')
    #brick= '2447p120'
    #sim_main(args=['--brick', brick, '-n', '2', '-o', 'STAR', \
    #               '-ic', '1', '--rmag-range', '18', '26', '--threads', '1',\
    #               '--zoom', '1020', '1070', '2775', '2815'])
    # Check if correct files written out
    #rt_dir= os.path.join(os.getenv('DECALS_SIM_DIR'),brick,'star','001')
    #assert( os.path.exists(os.path.join(rt_dir,'../','metacat-'+brick+'-star.fits')) )
    #for fn in ['tractor-%s-star-01.fits' % brick,'simcat-%s-star-01.fits' % brick]:
    #    assert( os.path.exists(os.path.join(rt_dir,fn)) )
    #for fn in ['image','model','resid','simscoadd']:
    #    assert( os.path.exists(os.path.join(rt_dir,'qa-'+brick+'-star-'+fn+'-01.jpg')) )

    # if ceres:
    #     # With ceres
    #     main(args=['--brick', '2447p120', '--zoom', '1020', '1070', '2775', '2815',
    #                '--no-wise', '--force-all', '--no-write', '--ceres',
    #                '--survey-dir', surveydir,
    #                '--outdir', 'out-testcase3-ceres'] + extra_args)

def test_4(ceres, psfex):
    from legacypipe.survey import LegacySurveyData

    # Test that we can run splinesky calib if required...

    from legacypipe.decam import DecamImage
    DecamImage.splinesky_boxsize = 128

    surveydir = os.path.join(os.path.dirname(__file__), 'testcase4')
    set_ref_env(surveydir)
    survey = LegacySurveyData(surveydir)
    # get brick by id
    brickid = 473357
    brick = survey.get_brick(brickid)
    assert(brick.brickname == '1867p255')
    assert(brick.brickid == brickid)

    outdir = 'out-testcase4'

    fn = os.path.join(surveydir, 'calib', 'sky-single', 'decam', 'CP', 'V4.8.2',
                      'CP20170315', 'c4d_170316_062107_ooi_z_ls9',
                      'c4d_170316_062107_ooi_z_ls9-N2-splinesky.fits')
    if os.path.exists(fn):
        os.unlink(fn)

    main(args=['--brick', '1867p255', '--zoom', '2050', '2300', '1150', '1400',
               '--force-all', '--no-write', '--coadd-bw',
               '--unwise-dir', os.path.join(surveydir, 'images', 'unwise'),
               '--unwise-tr-dir', os.path.join(surveydir,'images','unwise-tr'),
               '--blob-image', '--no-hybrid-psf',
               '--survey-dir', surveydir,
               '--outdir', outdir, '-v', '--no-wise-ceres'])
    print('Checking for calib file', fn)
    assert(os.path.exists(fn))

    # Test with blob-masking when creating sky calib.
    os.unlink(fn)

    main(args=['--brick', '1867p255', '--zoom', '2050', '2300', '1150', '1400',
               '--force-all', '--no-write', '--coadd-bw',
               '--blob-mask-dir', surveydir,
               '--survey-dir', surveydir,
               '--stage', 'image_coadds',
               '--outdir', 'out-testcase4b', '--plots'])
    print('Checking for calib file', fn)
    assert(os.path.exists(fn))

    if ceres:
        main(args=['--brick', '1867p255', '--zoom', '2050', '2300', '1150', '1400',
                   '--force-all', '--no-write', '--coadd-bw',
                   '--unwise-dir', os.path.join(surveydir, 'images', 'unwise'),
                   '--unwise-tr-dir', os.path.join(surveydir,'images','unwise-tr'),
                   '--survey-dir', surveydir,
                   '--outdir', outdir])
    if psfex:
        sys.stdout.flush()
        print('Test 4: regenerating PsfEx file')
        sys.stdout.flush()
        # Check that we can regenerate PsfEx files if necessary.
        fn = os.path.join(surveydir, 'calib', 'psfex', 'decam', 'CP', 'V4.8.2',
                          'CP20170315', 'c4d_170316_062107_ooi_z_ls9-psfex.fits')
        if os.path.exists(fn):
            os.unlink(fn)

        main(args=['--brick', '1867p255', '--zoom', '2050', '2300', '1150', '1400',
                   '--force-all', '--no-write', '--coadd-bw',
                   '--stage', 'tims',
                   '--survey-dir', surveydir,
                   '--outdir', outdir, '-v'])

        # if will generate a psfex-single file
        sfn = os.path.join(surveydir, 'calib', 'psfex-single', 'decam', 'CP', 'V4.8.2',
                           'CP20170315', 'c4d_170316_062107_ooi_z_ls9',
                           'c4d_170316_062107_ooi_z_ls9-N2-psfex.fits')
        assert(os.path.exists(sfn))
        P = fits_table(sfn)
        print('PSF model:')
        P.about()
        print(P[0])

        # sys.stdout.flush()
        # print('After generating PsfEx calib:')
        # os.system('find %s' % (os.path.join(surveydir, 'calib')))
        # sys.stdout.flush()

def test_4b():
    print('Test 4b')
    
    # Custom RA,Dec; blob ra,dec.
    surveydir = os.path.join(os.path.dirname(__file__), 'testcase4')
    set_ref_env(surveydir)
    outdir = 'out-testcase4b'
    # Catalog written with one entry (--blobradec)
    fn = os.path.join(outdir, 'tractor', 'cus',
                      'tractor-custom-186743p25461.fits')
    if os.path.exists(fn):
        os.unlink(fn)
    main(args=['--radec', '186.743965', '25.461788',
               '--width', '250', '--height', '250',
               '--force-all', '--no-write', '--no-wise',
               '--blobradec', '186.740369', '25.453855',
               '--survey-dir', surveydir,
               '--outdir', outdir])

    assert(os.path.exists(fn))
    T = fits_table(fn)
    assert(len(T) == 1)
    print('Test 4b finished')

if __name__ == '__main__':
    rbmain()
