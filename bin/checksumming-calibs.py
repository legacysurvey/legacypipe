# Generates checksums for merged calibration files
# Martin Landriau, LBNL February 2018

from glob import glob
from hashlib import sha256
import os
import sys

drdir = "/global/project/projectdirs/cosmo/work/legacysurvey/dr8/calib/"
cameras = ("90prime", "mosaic", "decam")
calibtypes = ("psfex-merged", "splinesky-merged")

for camera in cameras:
    for caltype in calibtypes:
        os.chdir(drdir+camera+"/"+caltype)
        sdlist = glob("*")
        sdlist.sort()
        for sd in sdlist:
            os.chdir(drdir+camera+"/"+caltype+"/"+sd)
            flist = glob("*")
            flist.sort()
            f = open("legacysurvey_dr8_calib_"+camera+"_"+caltype+"_"+sd+".sha256sum", "w")
            for calfile in flist:
                g = open(calfile,'rb')
                m = sha256(g.read())
                f.write(m.hexdigest()+"  "+calfile+"\n")
                g.close()
            f.close()
    #sys.exit()

