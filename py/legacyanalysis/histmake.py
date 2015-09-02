#!/usr/bin/env python


import numpy as np
from astropy.table import Table
from astropy.io import ascii
    

def histmake( filein, fileout, bin_gr, bin_rz, grmin = -999., grmax = -999., rzmin = -999., rzmax = -999.):

    '''
    Written by Gregory Rudnick 27 August 2015

    PURPOSE: 

    reads in a list of colors from an ascii file and ouputs a 2D
    histogram in color
      
    CALLING SEQUENCE
    
    histmake(filein, fileout, grmin, grmax, rzmin, rzmax, bin_gr, bin_rz)

    INPUT PARAMETERS

    filein - the input ascii file

    fileout - the output 2D histogram as an ASCII file

    bin_gr, bin_rz - the bin sizes in colors

    OPTIONAL KEYWORKDS
      
    grmin, grmax, rzmin, rzmax - the min and max boundaries of the
    histogram.  If they are not specified as options they will be
    computed from the data and truncated to the nearest 0.1

    OUTPUTS

    an output file containin the 2D histogram normalized to the total
    number of objects in the histogram.

    EXAMPLES
 
    PROCEDURE

    REQUIRED PYTHON MODULES

    astropy
    numpy

    NOTES

    '''
    #read in catalog as ascii table
    data = ascii.read(filein)
    
    #set max and min values to be the max and min limits rounded to
    #the nearest 0.1 if they are not specified at the function call.
    if grmin < -990.:
        grmin = round(np.ndarray.min(data['gr']),1)
        
    if grmax < -990.:
        grmax = round(np.ndarray.max(data['gr']),1)

    if rzmin < -990.:
        rzmin = round(np.ndarray.min(data['rz']),1)
        
    if rzmax < -990.:
        rzmax = round(np.ndarray.max(data['rz']),1)
    

    print grmin, grmax, rzmin, rzmax

    #make gr grid and gz grid
    grcent = np.arange(grmin,grmax,bin_gr)
    rzcent = np.arange(rzmin,rzmax,bin_rz)

    #count total number of objects in bounds of array to normalize the total.
    ntot = (data['gr'] > grmin) & \
        (data['gr'] <= grmax) & \
        (data['rz'] > rzmin) & \
        (data['rz'] <= rzmax)
    ntot = np.sum(ntot)

    #open file for writing and write a header
    fo = open(fileout, "w")
    fo.write("# grcent rzcent ninbin prob\n")

    #loop through each bin center and count the number of objects in
    #that bin.
    ngr = len(grcent)
    nrz = len(rzcent)
    fo.write('# numer of g-r bins =  {}; number of r-z bins =  {}\n'.format(ngr, nrz))
    for gr in grcent:
        for rz in rzcent:
            inbin = (data['gr'] > gr - bin_gr/2.0) & \
                (data['gr'] <= gr + bin_gr/2.0) & \
                (data['rz'] > rz - bin_rz/2.0) & \
                (data['rz'] <= rz + bin_rz/2.0)
            ninbin = np.sum(inbin)

            # normalize the number in that bin by the total number
            # within the histogram limits.  Multiply normalization
            # factor by binsizes to properly normalize this to a
            # differential probability
            probbin = float(ninbin) / (float(ntot) * bin_gr * bin_rz)
            
            fo.write('{} {} {} {}\n'.format(gr, rz, ninbin, probbin))

# Close opend file
    fo.close()

    return;
            
