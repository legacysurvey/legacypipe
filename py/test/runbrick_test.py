from __future__ import print_function
import os
import sys
from legacypipe.runbrick import main
from legacyanalysis.decals_sim import main as sim_main

def bash(cmd):
    ret= os.system('%s' % cmd)
    if ret:
        print('command failed: %s' % cmd)
        raise ValueError


if __name__ == '__main__':

    travis = 'travis' in sys.argv

    # surveydir = os.path.join(os.path.dirname(__file__), 'testcase3')
    # main(args=['--brick', '2447p120', '--zoom', '1020', '1070', '2775', '2815',
    #            '--no-wise', '--force-all', '--no-write',
    #            '--survey-dir', surveydir,
    #            '--outdir', 'out-testcase3'])

    # # MzLS + BASS data
    # surveydir2 = os.path.join(os.path.dirname(__file__), 'mzlsbass')
    # main(args=['--brick', '3521p002', '--zoom', '2400', '2450', '1200', '1250',
    #            '--no-wise', '--force-all', '--no-write',
    #            '--survey-dir', surveydir2,
    #            '--outdir', 'out-mzlsbass'])

    # # With plots!
    # main(args=['--brick', '2447p120', '--zoom', '1020', '1070', '2775', '2815',
    #            '--no-wise', '--force-all', '--no-write',
    #            '--survey-dir', surveydir,
    #            '--outdir', 'out-testcase3', '--plots'])

    # Decals Image Simulations
    os.environ["DECALS_SIM_DIR"]= os.path.join(os.path.dirname(__file__),'image_sims')
    #sim_main(args=['--brick', '2523p355', '-n', '5', '-o', 'STAR', \
    #               '-ic', '1', '--rmag-range', '18 26', '--threads', '1',\
    #               '--zoom', '1750 1800 1750 1800'])
    bash("python legacyanalysis/decals_sim.py --brick 2523p355 -n 2 -o STAR -ic 1 --rmag-range 18 26 --threads 1")
    # Check if correct files written out
    rt_dir= os.path.join(os.getenv('DECALS_SIM_DIR'),'2523p355','star','001')
    for fn in ['metacat-2523p355-star.fits','tractor-2523p355-star-01.fits',\
               'simcat-2523p355-star-01.fits']: 
        assert( os.path.exists(os.path.join(rt_dir,fn)) )
    for fn in ['image','model','resid','simscoadd']: 
        assert( os.path.exists(os.path.join(rt_dir,'qa-2523p355-star-'+fn+'-01.jpg')) )
     
    # if not travis:
    #     # With ceres
    #     main(args=['--brick', '2447p120', '--zoom', '1020', '1070', '2775', '2815',
    #                '--no-wise', '--force-all', '--no-write', '--ceres',
    #                '--survey-dir', surveydir,
    #                '--outdir', 'out-testcase3-ceres'])
    

    
