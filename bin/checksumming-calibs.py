# Generates checksums for merged calibration files
# Martin Landriau, LBNL February 2018

from glob import glob
from hashlib import sha256
import os
import sys

drdir = "/global/projecta/projectdirs/cosmo/work/legacysurvey/dr6/calib/"
cameras = ("90prime", "mosaic")
calibtypes = ("psfex", "splinesky")

for camera in cameras:
    for caltype in calibtypes:
        os.chdir(drdir+camera+"/"+caltype)
        sdlist = glob("*")
        sdlist.sort()
        for sd in sdlist:
            os.chdir(drdir+camera+"/"+caltype+"/"+sd)
            flist = glob("*")
            f = open("legacysurvey_dr6_calib_"+camera+"_"+caltype+"_"+sd+".sha256sum", "w")
            for calfile in flist:
                g = open(calfile,'rb')
                m = sha256(g.read())
                f.write(m.hexdigest()+"  "+calfile+"\n")
                g.close()
            f.close()
            #sys.exit()

