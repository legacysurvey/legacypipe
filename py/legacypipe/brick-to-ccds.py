"""This script does something.
"""
from __future__ import print_function
from common import * # This is not best practice.
from sys import exit
#
#
#
def main():
    """Main program.
    """
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('brickname',nargs='+',help="Name(s) of brick(s).",metavar='BRICK')
    opt = parser.parse_args()

    survey = LegacySurveyData()
    CCDs = survey.get_ccds()
    for brickname in opt.brickname:
        brick = survey.get_brick_by_name(brickname)
        print('# Brick', brickname, 'RA,Dec', brick.ra, brick.dec)
        wcs = wcs_for_brick(brick)
        I = ccds_touching_wcs(wcs, CCDs)
        for i in I:
            print(i, CCDs.filter[i], CCDs.expnum[i], CCDs.extname[i], CCDs.cpimage[i], CCDs.cpimage_hdu[i], CCDs.calname[i])
    return 0
#
#
#
if __name__ == '__main__':
    exit(main())
