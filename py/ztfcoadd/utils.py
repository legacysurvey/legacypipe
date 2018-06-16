#!/usr/bin/env python

import numpy as np
import os
import shutil
import subprocess
import psycopg2
import fnmatch
from astropy.io import fits
from importlib import reload

PROJECTPATH = os.environ['PROJECTPATH']
CODEPATH = os.environ['CODEPATH']

def is_number(s):
    """ Test if the string 's' is a number
  
    Parameters
    ----------
    s : `str`
        The string which is to be tested as a number
    """
    try:
        float(s)
        return True
    except ValueError:
        pass
    
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

def is_int(s):
    """ Test if the string 's' is an int
  
    Parameters
    ----------
    s : `str`
        The string which is to be tested as an int
    """

    if not is_number(s):
        return false

    try:
        assert int(float(s)) == float(s)
        return True
    except AssertionError:
        pass

    return False

def print_d(string,debug):
    if debug == True:
        print(string)
    else:
        pass

def load_auth_info(auth_file, auth_type):
    """Loads authentication information from file (see README.md)
    Parameters
    ----------
    auth_type : `str`
        `archive`, `depot`, or `db`
    """
    username, password = None, None

    with open(auth_file, 'r') as f:
        for line in f:
            key = line.split(':')[0].replace(' ','')
            value = line.split(':')[1].replace(' ','').replace('\n','')
            if key == 'AUTH_%s_USERNAME' % auth_type.upper():
                username = value
            if key == 'AUTH_%s_PASSWORD' % auth_type.upper():
                password = value

    if (username == None) or (password == None):
        print('Incorrectly formatted authentication file!')
        sys.exit()

    return (username, password)

def execute(cmd,shell=False):
    """Execute a shell command, log the stdout and stderr, and check
    the return code."""

    args = cmd.split()
    process = subprocess.Popen( args, stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell=shell)
    stdout, stderr = process.communicate()

    return stdout, stderr

def xy2sky(im, x, y):
    """Convert the x and y coordinates on an image to RA, DEC in degrees."""
    command = '%s/xy2sky -d %s %d %d' % (os.environ['CODEPATH'], im, x, y)
    stdout, stderr = execute(command)
    ra, dec, epoch, x, y = stdout.split()
    return float(ra), float(dec)

def parse_sexcat(cat, bin=False):
    """Read a sextractor catalog file (path: `cat`) and return a numpy
    record array containing the values."""

    if bin:
        data = fits.open(cat)[2].data
    else:
        data = np.genfromtxt(cat, dtype=None)
    return data

def get_folders(base_dir, folder_type='chip', file_min=4, folder_limit=0):

    if base_dir[-1] == '/':
        base_dir = base_dir[:-1]

    if folder_type == 'chip':
        folder_char = 'C'
        depth_level = 4
    elif folder_type == 'field':
        folder_char = 'f'
        depth_level = 3
        file_min = 0
    else:
        print("""folder_type must be 'field' or 'chip'""")
        return []

    folder_arr = []
    root_depth = base_dir.count(os.path.sep) - 1
    count = 0
    for root, dirs, files in os.walk(base_dir,topdown=True):

        depth = root.count(os.path.sep) - root_depth
        if root.split('/')[-1][0] == folder_char:
            scie_files = fnmatch.filter(files,"PTF*scie*c??.fits")
            if len(scie_files) >= file_min:
                count += 1
                folder_arr.append(root)
                if count == folder_limit:
                    return folder_arr

        if depth >= depth_level:
            del dirs[:]

    return folder_arr

def replace_list(list, list_name):
    if os.path.exists(list_name):
        os.remove(list_name)

    with open(list_name,'w') as f:
        for item in list:
            f.write('%s\n'%item)

def update_fits(image_fname, data, header):

    image_tmp = image_fname.replace("fits","fits.tmp")
    fits.writeto(image_tmp,data,header,overwrite=True)

    if os.path.exists(image_fname):
        os.remove(image_fname)
    shutil.move(image_tmp,image_fname)

def create_fits(image_fname, data, header=None):

    hdu = fits.PrimaryHDU(data)
    if header != None:
        hdu.header = header
    if os.path.exists(image_fname):
        os.remove(image_fname)
    hdu.writeto(image_fname)

def return_correct_file_size(im_list):

    file_size_arr = []
    for im in im_list:
        if os.path.exists(im):
            file_size_arr.append(os.path.getsize(im))

    if len(file_size_arr) > 0:
        return max(file_size_arr)
    else:
        return 0

sortsplit = lambda iterable, n: [[iterable[i] for i in range(j,len(iterable),n)] for j in range(n)]
split = lambda iterable, n: [iterable[:len(iterable)/n]] + \
    split(iterable[len(iterable)/n:], n - 1) if n != 0 else []
trim = lambda s: s.split('/')[-1]