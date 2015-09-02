#!/usr/bin/env python


import numpy as np
from astropy.table import Table
from astropy.io import ascii
import matplotlib.pyplot as plt

def color_draw(filein, ndraw, drawplot = 0):
    '''Written by Gregory Rudnick 27 August 2015

    PURPOSE: 

      This function reads in a 2D histogram in color, collapses it
      into 1D and draws each of the colors from the collapsed
      distribution.  It can return many draws.

      The spacing of the histogram is read from the input file, and so
      requires constant linear spacing in color.
      
    CALLING SEQUENCE
    

    INPUT PARAMETERS
      filein - input ASCII file with 2D histogram
      ndraw - number of draws
      
    OPTIONAL INPUT PARAMETERS

      drawplot - if == 1, draw a plot with the two histograms of the
      draws overlaid as well as the collapsed probabilities of the
      input distribution

    OUTPUT PARAMETERS
      gr - list of g-r colors
      rz - list of r-z colors

    EXAMPLES
 
    PROCEDURE

    REQUIRED PYTHON MODULES
    numpy
    astropy
    matplotlib

    ADDITIONAL REQUIRED MODULES

    NOTES

    assumes constant spacing in color

    '''

    #get the column names
    data = ascii.read(filein)
    colnames = data.colnames

    #get binsize assuming that there is a constant increment.  Look
    #for the first instance in each color where the value changes and
    #use that as the incremement
    igr = 1
    while (abs(data['grcent'][igr] - data['grcent'][igr - 1]) < 0.01):
        igr = igr + 1
    bin_gr = abs(data['grcent'][igr] - data['grcent'][igr - 1])

    irz = 1
    while (abs(data['rzcent'][irz] - data['rzcent'][irz - 1]) < 0.01):
        irz = irz + 1
    bin_rz = abs(data['rzcent'][irz] - data['rzcent'][irz - 1])

    #find number of elements
    grmin = np.ndarray.min(data['grcent'])
    grmax = np.ndarray.max(data['grcent'])
    ngr = int((grmax - grmin) / bin_gr ) + 1

    rzmin = np.ndarray.min(data['rzcent'])
    rzmax = np.ndarray.max(data['rzcent'])
    nrz = int((rzmax - rzmin) / bin_rz) + 1

    print grmin,grmax,ngr
    print rzmin,rzmax,nrz
    
    #make the color axes vectors
    grcent = np.arange(grmin, grmax + bin_gr, bin_gr)
    rzcent = np.arange(rzmin, rzmax + bin_rz, bin_rz)
    print len(grcent), len(rzcent)

    #loop through every line to build the 2D array
    probarr = np.zeros((ngr+1,nrz+1))
    igr = 0
    for gr in grcent:
        irz = 0
        for rz in rzcent:
            ind = (abs(gr - data['grcent']) < 0.01) & \
                (abs(rz - data['rzcent']) < 0.01)
            probarr[igr][irz] = data['prob'][ind]
            #print igr, irz, grcent[igr],rzcent[irz],probarr[igr][irz]
            irz = irz + 1
        igr = igr + 1       

    nbinhist = 50
    grnorm = float(nbinhist) / (grmax - grmin)
    rznorm = float(nbinhist) / (rzmax - rzmin)
        
    rzprob=probarr.sum(axis=0)

    grprob=probarr.sum(axis=1)
    
    #make a cumulative histogram in each color
    grcum = np.cumsum(grprob, dtype=float)
    rzcum = np.cumsum(rzprob, dtype=float)

    #draw ndraw samples of each color
    x = np.random.uniform(0.0, 1.0, size=ndraw)

    grdraw = np.interp(x, grcum, grcent)
    rzdraw = np.interp(x, rzcum, rzcent)

    if drawplot == 1:
        plt.clf()
        plt.plot(grcent,grprob*ndraw/bin_rz/rznorm,'gs')
        plt.plot(rzcent,rzprob*ndraw/bin_gr/grnorm,'ro')
        plt.hist(grdraw, bins=nbinhist)
        plt.hist(rzdraw, bins=nbinhist)
    
    return grdraw,rzdraw;

