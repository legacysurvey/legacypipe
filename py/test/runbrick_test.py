from __future__ import print_function
import os
import sys
import numpy as np
from legacypipe.runbrick import main
#from legacyanalysis.decals_sim import main as sim_main

if __name__ == '__main__':

    travis = 'travis' in sys.argv

    surveydir = os.path.join(os.path.dirname(__file__), 'testcase3')
    outdir = 'out-testcase3'
    main(args=['--brick', '2447p120', '--zoom', '1020', '1070', '2775', '2815',
               '--no-wise', '--force-all', '--no-write',
               '--survey-dir', surveydir,
               '--outdir', outdir])

    # Read catalog into Tractor sources to test read_fits_catalog
    from legacypipe.catalog import read_fits_catalog
    from legacypipe.survey import LegacySurveyData
    from astrometry.util.fits import fits_table
    from tractor.galaxy import DevGalaxy
    from tractor import PointSource
    
    survey = LegacySurveyData(survey_dir=outdir)
    fn = survey.find_file('tractor', brick='2447p120')
    T = fits_table(fn)
    T.shapeexp = np.vstack((T.shapeexp_r, T.shapeexp_e1, T.shapeexp_e2)).T
    T.shapedev = np.vstack((T.shapedev_r, T.shapedev_e1, T.shapedev_e2)).T
    cat = read_fits_catalog(T)
    print('Read catalog:', cat)
    assert(len(cat) == 2)
    src = cat[0]
    assert(type(src) == DevGalaxy)
    assert(np.abs(src.pos.ra  - 244.77975) < 0.00001)
    assert(np.abs(src.pos.dec -  12.07234) < 0.00001)
    src = cat[1]
    assert(type(src) == PointSource)
    assert(np.abs(src.pos.ra  - 244.77833) < 0.00001)
    assert(np.abs(src.pos.dec -  12.07252) < 0.00001)
    # DevGalaxy(pos=RaDecPos[244.77975494973529, 12.072348111713127], brightness=NanoMaggies: g=19.2, r=17.9, z=17.1, shape=re=2.09234, e1=-0.198453, e2=0.023652,
    # PointSource(RaDecPos[244.77833280764278, 12.072521274981987], NanoMaggies: g=25, r=23, z=21.7)

    # MzLS + BASS data
    surveydir2 = os.path.join(os.path.dirname(__file__), 'mzlsbass')
    main(args=['--brick', '3521p002', '--zoom', '2400', '2450', '1200', '1250',
               '--no-wise', '--force-all', '--no-write',
               '--survey-dir', surveydir2,
               '--outdir', 'out-mzlsbass'])

    # With plots!
    main(args=['--brick', '2447p120', '--zoom', '1020', '1070', '2775', '2815',
               '--no-wise', '--force-all', '--no-write',
               '--survey-dir', surveydir,
               '--outdir', 'out-testcase3', '--plots'])

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
                   '--outdir', 'out-testcase3-ceres'])
    

    
