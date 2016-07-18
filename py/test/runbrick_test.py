from __future__ import print_function
import os
import sys
from legacypipe.runbrick import main
from legacyanalysis.decals_sim import main as sim_main

if __name__ == '__main__':

    travis = 'travis' in sys.argv

    surveydir = os.path.join(os.path.dirname(__file__), 'testcase3')
    main(args=['--brick', '2447p120', '--zoom', '1020', '1070', '2775', '2815',
               '--no-wise', '--force-all', '--no-write',
               '--survey-dir', surveydir,
               '--outdir', 'out-testcase3'])

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
    os.environ["DECALS_SIM_DIR"]= os.path.join(os.path.dirname(__file__),'image_sims')
    brick= '2447p120'
    sim_main(args=['--brick', brick, '-n', '2', '-o', 'STAR', \
                   '-ic', '1', '--rmag-range', '18', '26', '--threads', '1',\
                   '--zoom', '1020', '1070', '2775', '2815'])
    # Check if correct files written out
    rt_dir= os.path.join(os.getenv('DECALS_SIM_DIR'),brick,'star','001')
    for fn in ['metacat-%s-star.fits' % brick,'tractor-%s-star-01.fits' % brick,\
               'simcat-%s-star-01.fits' % brick]: 
        assert( os.path.exists(os.path.join(rt_dir,fn)) )
    for fn in ['image','model','resid','simscoadd']: 
        assert( os.path.exists(os.path.join(rt_dir,'qa-'+brick+'-star-'+fn+'-01.jpg')) )
     
    if not travis:
        # With ceres
        main(args=['--brick', '2447p120', '--zoom', '1020', '1070', '2775', '2815',
                   '--no-wise', '--force-all', '--no-write', '--ceres',
                   '--survey-dir', surveydir,
                   '--outdir', 'out-testcase3-ceres'])
    

    
