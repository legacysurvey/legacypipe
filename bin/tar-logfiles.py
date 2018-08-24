import glob
import os
import sys

os.chdir("/global/projecta/projectdirs/cosmo/work/legacysurvey/dr7-attic/logs/")
dirlist = glob.glob("./*")
for dirs in dirlist:
    dirname = dirs[2:]
    filename = "legacysurvey_dr7_logs_"+dirname+".tar.gz"
    cmd = "tar -czvf "+filename+" "+dirname
    #print(cmd)
    #sys.exit()
    os.system(cmd)

