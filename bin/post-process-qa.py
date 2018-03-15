"""
Checks output directory for extraneous files
and that there are no files missing.
August 2017 Martin Landriau LBNL
"""

import os
import qdo
import glob
from hashlib import sha256
import sys

nstart = int(sys.argv[1])
nend = int(sys.argv[2])

outdir = "/global/projecta/projectdirs/cosmo/work/legacysurvey/dr6/"
ncoaddall = 0
notractor = []
q = qdo.connect('dr6')
a = q.tasks(state=qdo.Task.SUCCEEDED)
n = len(a)
for i in range(nstart, nend):
    brick = a[i].task
    subdir = brick[0:3]
    fullpath = outdir+"tractor/"+subdir+"/"
    filename0 = "brick-"+brick+".sha256sum"
    ncoadd = 0
    if(not os.path.isfile(fullpath+filename0)):
        notractor.append(filename0)
    else:
        f = open(outdir+"tractor/"+subdir+"/brick-"+brick+".sha256sum")
        filelist = f.readlines()
        for line in filelist:
            words = line.split()
            cs256 = words[0]
            filename = words[1][1:]
            dirs = filename.split("/")
            if dirs[0] == "coadd":
                ncoadd += 1
                ncoaddall += 1
            if dirs[0] != "tractor-i":
                g = open(outdir+filename,'rb')
                m = sha256(g.read())
                if m.hexdigest() != cs256:
                    print(filename)
        dirname = outdir+"coadd/"+subdir+"/"+brick+"/*"
        coaddlist = glob.glob(dirname)
        nc = len(coaddlist)
        if nc != ncoadd:
            print("Inconsistent number of files in the coadd directory for brick "+brick)

print("\n")
print("Number of bricks without tractor catalogues: "+str(len(notractor)))
print("Total number of coadd files: "+str(ncoaddall))

