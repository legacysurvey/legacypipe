"""
Repackages checksums for tractor output in DR convention
DR6: original version, Feb 2018
DR7: treating maskbit and tractor files differently due to bright-neighbour post-processing,
     added dr variable to make changes easier between DRs, Jul 2018 
DR8: only tractor files need to have their checksums re-computed.
Martin Landriau LBNL
"""

import os
from glob import glob
import sys
from hashlib import sha256

outdir = "/global/project/projectdirs/cosmo/work/legacysurvey/dr8/south/"
#dumpdir = "/global/project/projectdirs/cosmo/work/legacysurvey/dr8-north-attic/original_shasum"
csdir = "/global/project/projectdirs/cosmo/work/legacysurvey/dr8-garage/south/original-tractor/"
dr = "dr8_south"

os.chdir(outdir+"tractor")
sdlist = glob("*")
sdlist.sort()
for subdir in sdlist:
    #os.chdir(outdir+"tractor/"+subdir)
    os.chdir(csdir+subdir)
    #if not os.path.exists(dumpdir+"/"+subdir):
    #    os.makedirs(dumpdir+"/"+subdir)
    metrics = open(outdir+"metrics/"+subdir+"/legacysurvey_"+dr+"_metrics_"+subdir+".sha256sum", "w")
    tractor = open(outdir+"tractor/"+subdir+"/legacysurvey_"+dr+"_tractor_"+subdir+".sha256sum", "w")
    #bricklist = glob("tractor*")
    bricklist =  glob("brick*")
    bricklist.sort()
    for brickfile in bricklist:
        #brick = brickfile[8:16]
        brick = brickfile[6:14]
        coadd = open(outdir+"coadd/"+subdir+"/"+brick+"/legacysurvey_"+dr+"_coadd_"+subdir+"_"+brick+".sha256sum", "w")
        f = open(csdir+subdir+"/brick-"+brick+".sha256sum")
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
                #tractor.write(cs256+"  "+filename+"\n")
                g = open(outdir+"tractor/"+subdir+"/"+filename,'rb')
                m = sha256(g.read())
                tractor.write(m.hexdigest()+"  "+filename+"\n")
                g.close()
        f.close()
        coadd.close()
        #os.rename("brick-"+brick+".sha256sum", dumpdir+"/"+subdir+"/brick-"+brick+".sha256sum")
    metrics.close()
    tractor.close()

