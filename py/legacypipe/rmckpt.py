"""
Removes checkpoint for a given brick and output directory
"""

import os, errno
import argparse

def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred

def removeckpt(brick, rundir):
    subdir = brick[0:3]
    if rundir[-1] != '/':
        rundir+='/'
    filename =  rundir + "checkpoints/" + subdir + "/checkpoint-" + brick + ".pickle"
    silentremove(filename)
    filename = rundir + "pickles/" + subdir + "/runbrick-" + brick + "-coadds.pickle"
    silentremove(filename)
    filename = rundir + "pickles/" + subdir + "/runbrick-" + brick + "-fitblobs.pickle"
    silentremove(filename)
    filename = rundir + "pickles/" + subdir + "/runbrick-" + brick + "-mask_junk.pickle"
    silentremove(filename)
    filename = rundir + "pickles/" + subdir + "/runbrick-" + brick + "-srcs.pickle"
    silentremove(filename)
    filename = rundir + "pickles/" + subdir + "/runbrick-" + brick + "-tims.pickle"
    silentremove(filename)
    filename = rundir + "pickles/" + subdir + "/runbrick-" + brick + "-wise_forced.pickle"
    silentremove(filename)
    filename = rundir + "pickles/" + subdir + "/runbrick-" + brick + "-writecat.pickle"
    silentremove(filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Removes checkpoint and pickle files for given brick.')
    parser.add_argument('--brick', type=str, required=True, help='Brick name.')
    parser.add_argument('--outdir',type=str, required=True, help='Directory where checkpoints and pickles subdirectories are.')
    opt = parser.parse_args()
    removeckpt(opt.brick, opt.outdir)

