from __future__ import print_function
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import os
import sys
import numpy as np
from legacypipe.runbrick import main
#from legacyanalysis.decals_sim import main as sim_main
from astrometry.util.fits import fits_table

def rbmain():
    travis = 'travis' in sys.argv

    extra_args = ['--old-calibs-ok',
    #'--verbose',
        ]
    if travis:
        extra_args.extend(['--no-wise-ceres', '--no-gaia', '--no-large-galaxies'])
    
    if 'ceres' in sys.argv:
        surveydir = os.path.join(os.path.dirname(__file__), 'testcase3')
        main(args=['--brick', '2447p120', '--zoom', '1020', '1070', '2775', '2815',
                   '--no-wise', '--force-all', '--no-write', '--ceres',
                   '--survey-dir', surveydir,
                   '--outdir', 'out-testcase3-ceres',
                   '--no-depth-cut'])
        sys.exit(0)
    
    # demo RexGalaxy, with plots
    if False:
        from legacypipe.survey import RexGalaxy
        from tractor import NanoMaggies, PixPos
        from tractor import Image, GaussianMixturePSF, LinearPhotoCal
        from legacypipe.survey import LogRadius
        rex = RexGalaxy(
            PixPos(1., 2.),
            NanoMaggies(r=3.),
            LogRadius(0.))
        print('Rex:', rex)
        print('Rex params:', rex.getParams())
        print('Rex nparams:', rex.numberOfParams())
        H,W = 100,100
        tim = Image(data=np.zeros((H,W), np.float32),
                    inverr=np.ones((H,W), np.float32),
                    psf=GaussianMixturePSF(1., 0., 0., 4., 4., 0.),
                    photocal=LinearPhotoCal(1., band='r'))
        derivs = rex.getParamDerivatives(tim)
        print('Derivs:', len(derivs))
        print('Rex params:', rex.getParamNames())
    
        import pylab as plt
        from astrometry.util.plotutils import PlotSequence
        ps = PlotSequence('rex')
    
        for d,nm in zip(derivs, rex.getParamNames()):
            plt.clf()
            plt.imshow(d.patch, interpolation='nearest', origin='lower')
            plt.title('Derivative %s' % nm)
            ps.savefig()
        
        sys.exit(0)

    # MzLS + BASS data
    # python legacypipe/runbrick.py --run north --brick 1773p595 --zoom 1300 1500 700 900 --survey-dir dr9-north -s coadds
    # fitscopy coadd/177/1773p595/legacysurvey-1773p595-ccds.fits"[#row<3 || #row==12]" cx.fits
    # python legacyanalysis/create_testcase.py cx.fits test/mzlsbass2 1773p595 --survey-dir dr9-north/ --fpack
    surveydir2 = os.path.join(os.path.dirname(__file__), 'mzlsbass2')
    os.environ['GAIA_CAT_DIR'] = os.path.join(surveydir2, 'gaia')
    os.environ['GAIA_CAT_VER'] = '2'
    my_extra_args = [a for a in extra_args if a != '--no-gaia']
    main(args=['--brick', '1773p595', '--zoom', '1300', '1500', '700', '900',
               '--no-wise', '--force-all', '--no-write',
               '--survey-dir', surveydir2,
               '--outdir', 'out-mzlsbass2'] + my_extra_args)
    del os.environ['GAIA_CAT_DIR']
    del os.environ['GAIA_CAT_VER']

    # surveydir2 = os.path.join(os.path.dirname(__file__), 'mzlsbass')
    # main(args=['--brick', '3521p002', '--zoom', '2400', '2450', '1200', '1250',
    #            '--no-wise', '--force-all', '--no-write',
    #            '--survey-dir', surveydir2,
    #            '--outdir', 'out-mzlsbass'])
        
    # Test RexGalaxy

    surveydir = os.path.join(os.path.dirname(__file__), 'testcase6')
    outdir = 'out-testcase6-rex'
    the_args = ['--brick', '1102p240', '--zoom', '500', '600', '650', '750',
               '--force-all', '--no-write', '--no-wise',
                '--skip-calibs',
    #'--rex', #'--plots',
               '--survey-dir', surveydir,
               '--outdir', outdir] + extra_args
    print('python legacypipe/runbrick.py', ' '.join(the_args))
    os.environ['GAIA_CAT_DIR'] = os.path.join(surveydir, 'gaia')
    os.environ['GAIA_CAT_VER'] = '2'
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
               '--outdir', outdir] + extra_args)
    fn = os.path.join(outdir, 'tractor', '110', 'tractor-1102p240.fits')
    assert(os.path.exists(fn))
    T = fits_table(fn)
    assert(len(T) == 2)
    print('Types:', T.type)
    # Since there is a Tycho-2 star in the blob, forced to be PSF.
    assert(T.type[0].strip() == 'PSF')
    del os.environ['GAIA_CAT_DIR']
    del os.environ['GAIA_CAT_VER']
    
    # Test that we can run splinesky calib if required...
    
    from legacypipe.decam import DecamImage
    DecamImage.splinesky_boxsize = 128
    
    surveydir = os.path.join(os.path.dirname(__file__), 'testcase4')
    outdir = 'out-testcase4'

    fn = os.path.join(surveydir, 'calib', 'decam', 'CP', 'V4.8.2a',
                      'CP20150410', 'c4d_150411_035242_ooi_z_ls9',
                      'c4d_150411_035242_ooi_z_ls9-N3-splinesky.fits')
    if os.path.exists(fn):
        os.unlink(fn)

    main(args=['--brick', '1867p255', '--zoom', '2050', '2300', '1150', '1400',
               '--force-all', '--no-write', '--coadd-bw',
               '--unwise-dir', os.path.join(surveydir, 'images', 'unwise'),
               '--unwise-tr-dir', os.path.join(surveydir,'images','unwise-tr'),
               '--unwise-coadds',
               '--blob-image', '--no-hybrid-psf',
               '--survey-dir', surveydir,
               '--outdir', outdir] + extra_args + ['-v'])
    print('Checking for calib file', fn)
    assert(os.path.exists(fn))


    # Wrap-around, hybrid PSF
    surveydir = os.path.join(os.path.dirname(__file__), 'testcase8')
    outdir = 'out-testcase8'
    
    main(args=['--brick', '1209p050', '--zoom', '720', '1095', '3220', '3500',
               '--force-all', '--no-write', '--no-wise', #'--plots',
               '--survey-dir', surveydir,
               '--outdir', outdir] + extra_args)
    
    # Test with a Tycho-2 star + another saturated star in the blob.

    surveydir = os.path.join(os.path.dirname(__file__), 'testcase7')
    outdir = 'out-testcase7'
    # remove --no-gaia
    my_extra_args = [a for a in extra_args if a != '--no-gaia']
    os.environ['GAIA_CAT_DIR'] = os.path.join(surveydir, 'gaia')
    os.environ['GAIA_CAT_VER'] = '2'
    main(args=['--brick', '1102p240', '--zoom', '250', '350', '1550', '1650',
               '--force-all', '--no-write', '--no-wise', #'--plots',
               '--survey-dir', surveydir,
               '--outdir', outdir] + my_extra_args)
    del os.environ['GAIA_CAT_DIR']
    del os.environ['GAIA_CAT_VER']
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

    # Custom RA,Dec; blob ra,dec.
    surveydir = os.path.join(os.path.dirname(__file__), 'testcase4')
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
               '--outdir', outdir] + extra_args)

    assert(os.path.exists(fn))
    T = fits_table(fn)
    assert(len(T) == 1)

    surveydir = os.path.join(os.path.dirname(__file__), 'testcase3')
    outdir = 'out-testcase3'
    checkpoint_fn = os.path.join(outdir, 'checkpoint.pickle')
    if os.path.exists(checkpoint_fn):
        os.unlink(checkpoint_fn)
    main(args=['--brick', '2447p120', '--zoom', '1020', '1070', '2775', '2815',
               '--no-wise', '--force-all', '--no-write',
               '--survey-dir', surveydir,
               '--outdir', outdir,
               '--checkpoint', checkpoint_fn,
               '--checkpoint-period', '1',
               '--threads', '2'] + extra_args)

    # Read catalog into Tractor sources to test read_fits_catalog
    from legacypipe.catalog import read_fits_catalog
    from legacypipe.survey import LegacySurveyData, GaiaSource
    from tractor.galaxy import DevGalaxy
    from tractor import PointSource

    survey = LegacySurveyData(survey_dir=outdir)
    fn = survey.find_file('tractor', brick='2447p120')
    print('Checking', fn)
    T = fits_table(fn)
    cat = read_fits_catalog(T, fluxPrefix='')
    print('Read catalog:', cat)
    assert(len(cat) == 2)
    src = cat[0]
    assert(type(src) == DevGalaxy)
    assert(np.abs(src.pos.ra  - 244.77973) < 0.00001)
    assert(np.abs(src.pos.dec -  12.07233) < 0.00001)
    src = cat[1]
    print('Source', src)
    assert(type(src) in [PointSource, GaiaSource])
    assert(np.abs(src.pos.ra  - 244.77830) < 0.00001)
    assert(np.abs(src.pos.dec -  12.07250) < 0.00001)
    # DevGalaxy(pos=RaDecPos[244.77975494973529, 12.072348111713127], brightness=NanoMaggies: g=19.2, r=17.9, z=17.1, shape=re=2.09234, e1=-0.198453, e2=0.023652,
    # PointSource(RaDecPos[244.77833280764278, 12.072521274981987], NanoMaggies: g=25, r=23, z=21.7)

    # Check that we can run again, using that checkpoint file.
    main(args=['--brick', '2447p120', '--zoom', '1020', '1070', '2775', '2815',
               '--no-wise', '--force-all', '--no-write',
               '--survey-dir', surveydir,
               '--outdir', outdir,
               '--checkpoint', checkpoint_fn,
               '--checkpoint-period', '1',
               '--threads', '2'] + extra_args)
    # Assert...... something?

    # Test --checkpoint without --threads
    main(args=['--brick', '2447p120', '--zoom', '1020', '1070', '2775', '2815',
               '--no-wise', '--force-all', '--no-write',
               '--survey-dir', surveydir,
               '--outdir', outdir,
               '--checkpoint', checkpoint_fn,
               '--checkpoint-period', '1' ] + extra_args)
    
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
               '--nblobs', '1'] + extra_args)

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

    if not travis:
        # With ceres
        main(args=['--brick', '2447p120', '--zoom', '1020', '1070', '2775', '2815',
                   '--no-wise', '--force-all', '--no-write', '--ceres',
                   '--survey-dir', surveydir,
                   '--outdir', 'out-testcase3-ceres'] + extra_args)

if __name__ == '__main__':
    rbmain()

