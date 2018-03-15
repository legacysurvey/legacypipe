"""
Repackages checksums for tractor output in DR convention
February 2018 Martin Landriau LBNL
"""

import os
from glob import glob
import sys

outdir = "/global/projecta/projectdirs/cosmo/work/legacysurvey/dr6/"
dumpdir = "/global/projecta/projectdirs/cosmo/work/legacysurvey/dr6-dump/"

os.chdir(outdir+"tractor")
sdlist = glob("*")
sdlist.sort()
for subdir in sdlist:
    os.chdir(outdir+"tractor/"+subdir)
    if not os.path.exists(dumpdir+"/"+subdir):
        os.makedirs(dumpdir+"/"+subdir)
    metrics = open(outdir+"metrics/"+subdir+"/legacysurvey_dr6_metrics_"+subdir+".sha256sum", "w")
    tractor = open(outdir+"tractor/"+subdir+"/legacysurvey_dr6_tractor_"+subdir+".sha256sum", "w")
    bricklist = glob("brick*")
    bricklist.sort()
    for brickfile in bricklist:
        brick = brickfile[6:14]
        coadd = open(outdir+"coadd/"+subdir+"/"+brick+"/legacysurvey_dr6_coadd_"+subdir+"_"+brick+".sha256sum", "w")
        f = open("brick-"+brick+".sha256sum")
        filelist = f.readlines()
        for line in filelist:
            words = line.split()
            cs256 = words[0]
            longfilename = words[1][1:]
            dirs = longfilename.split("/")
            filename = dirs[-1]
            if dirs[0] == "coadd":
                coadd.write(cs256+"  "+filename+"\n")
            elif dirs[0] == "metrics":
                metrics.write(cs256+"  "+filename+"\n")
            elif dirs[0] == "tractor":
                tractor.write(cs256+"  "+filename+"\n")
        f.close()
        coadd.close()
        os.rename("brick-"+brick+".sha256sum", dumpdir+"/"+subdir+"/brick-"+brick+".sha256sum")
    metrics.close()
    tractor.close()

