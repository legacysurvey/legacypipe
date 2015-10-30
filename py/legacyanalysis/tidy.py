if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import pylab as plt
import numpy as np
import sys
import os
import scipy.ndimage

from tractor.brightness import NanoMaggies
from tractor.ellipses import EllipseE

from astrometry.util.fits import fits_table, merge_tables
from astrometry.util.miscutils import *
from astrometry.util.plotutils import *

if __name__ == '__main__':
    #brick = '1498p017'
    #base = 'dr2j'

    base = 'dr2m'
    brick = '2400p050'

    ps = PlotSequence('tidy')
    
    plt.figure(figsize=(10,10))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
                        hspace=0, wspace=0)
    
    fn = os.path.join('%s/tractor/%s/tractor-%s.fits' %
                      (base, brick[:3], brick))
    print 'Reading', fn
    T = fits_table(fn)
    print len(T), 'sources'
    jpeg = os.path.join('%s/coadd/%s/%s/decals-%s-image.jpg' %
                        (base, brick[:3], brick, brick))
    img = plt.imread(jpeg)
    img = np.flipud(img)

    mags = NanoMaggies.nanomaggiesToMag(T.decam_flux)
    mags[np.logical_not(np.isfinite(mags))] = 99.
    
    T.g = mags[:,1]
    T.r = mags[:,2]
    T.z = mags[:,4]

    # Convert 
    I = np.flatnonzero((T.r < 15) * (T.type == 'SIMP'))
    print 'Converting', len(I), 'bright SIMP objects into PSFs'
    T.type[I] = 'PSF '
    

    if False:
        P = T[T.type == 'PSF ']
        print len(P), 'PSFs'
        
        P.cut((P.g < 26) * (P.r < 26))
    
        P.cut(P.r < 24)
        
        #color = P.g - P.r
        color = P.g - P.z
        deciles = np.percentile(color, np.linspace(0, 100, 11))
        print 'color range', color.min(), color.max(), 'deciles', deciles
    
        rows,cols = 10,10
    
        plt.clf()
        k = 1
        for i,(lo,hi) in enumerate(zip(deciles, deciles[1:])):
            PI = P[(color >= lo) * (color < hi)]
            print len(PI), 'with color between', lo, 'and', hi
            PI = PI[np.argsort(PI.r)]
            print 'r mags:', PI.r
            for j in range(cols):
                plt.subplot(rows, cols, k)
                k += 1
                p = PI[j]
                x,y = int(np.round(p.bx)), int(np.round(p.by))
                H,W,nil = img.shape
                S = 20
                subimg = img[max(y-S, 0) : min(y+S+1, H),
                             max(x-S, 0) : min(x+S+1, W)]
                dimshow(subimg, ticks=False)
        ps.savefig()

    # rdeciles = np.percentile(P.r, np.linspace(0, 100, 11))
    # plt.clf()
    # k = 1
    # for i,(rlo,rhi) in enumerate(zip(rdeciles, rdeciles[1:])):
    #     PI = P[(P.r >= rlo) * (P.r < rhi)]
    #     print len(PI), 'with r between', rlo, 'and', rhi
    #     PI = PI[np.argsort(PI.g - PI.r)]
    #     print 'g-r:', PI.g - PI.r
    #     for j in range(cols):
    #         plt.subplot(rows, cols, k)
    #         k += 1
    #         p = PI[j]
    #         x,y = int(np.round(p.bx)), int(np.round(p.by))
    #         H,W,nil = img.shape
    #         S = 20
    #         subimg = img[max(y-S, 0) : min(y+S+1, H),
    #                      max(x-S, 0) : min(x+S+1, W)]
    #         dimshow(subimg, ticks=False)
    # ps.savefig()
    


    if False:
        S = T[T.type == 'SIMP']
        print len(S), 'SIMPs'
        
        S.cut((S.g < 26) * (S.r < 26))
        S.cut(S.r < 24)
    
        color = S.g - S.z
        deciles = np.percentile(color, np.linspace(0, 100, 11))
        print 'color range', color.min(), color.max(), 'deciles', deciles
    
        rows,cols = 10,10
        plt.clf()
        k = 1
        for i,(lo,hi) in enumerate(zip(deciles, deciles[1:])):
            SI = S[(color >= lo) * (color < hi)]
            print len(SI), 'with color between', lo, 'and', hi
            SI = SI[np.argsort(SI.r)]
            print 'r mags:', SI.r
            for j in range(cols):
                plt.subplot(rows, cols, k)
                k += 1
                p = SI[j]
                x,y = int(np.round(p.bx)), int(np.round(p.by))
                H,W,nil = img.shape
                sz = 20
                subimg = img[max(y-sz, 0) : min(y+sz+1, H),
                             max(x-sz, 0) : min(x+sz+1, W)]
                dimshow(subimg, ticks=False)
        ps.savefig()




    E = T[T.type == 'EXP ']
    print len(E), 'EXPs'
    
    E.cut((E.g < 26) * (E.r < 26))
    E.cut(E.r < 24)

    color = E.g - E.z
    deciles = np.percentile(color, np.linspace(0, 100, 11))
    print 'color range', color.min(), color.max(), 'deciles', deciles

    if False:
        rows,cols = 10,10
        plt.clf()
        k = 1
        for i,(lo,hi) in enumerate(zip(deciles, deciles[1:])):
            EI = E[(color >= lo) * (color < hi)]
            print len(EI), 'with color between', lo, 'and', hi
            EI = EI[np.argsort(EI.r)]
            print 'r mags:', EI.r
            for j in range(cols):
                plt.subplot(rows, cols, k)
                k += 1
                p = EI[j]
                x,y = int(np.round(p.bx)), int(np.round(p.by))
                H,W,nil = img.shape
                sz = 20
    
                outy,iny = get_overlapping_region(y - sz, y + sz, 0, H-1)
                outx,inx = get_overlapping_region(x - sz, x + sz, 0, W-1)
    
                subimg = np.zeros((sz*2+1, sz*2+1), np.float32)
                subimg[outy,outx] = img[iny,inx]
                
                #subimg = img[max(y-sz, 0) : min(y+sz+1, H),
                #             max(x-sz, 0) : min(x+sz+1, W)]
                dimshow(subimg, ticks=False)
        ps.savefig()
    


    rows,cols = 10,10
    plt.clf()
    k = 1
    for i,(lo,hi) in enumerate(zip(deciles, deciles[1:])):
        EI = E[(color >= lo) * (color < hi)]
        print len(EI), 'with color between', lo, 'and', hi
        EI = EI[np.argsort(EI.r)]
        print 'r mags:', EI.r
        for j in range(cols):
            plt.subplot(rows, cols, k)
            k += 1
            p = EI[j]
            x,y = int(np.round(p.bx)), int(np.round(p.by))
            H,W,planes = img.shape
            sz = 29

            iny,outy = get_overlapping_region(y - sz, y + sz, 0, H-1)
            inx,outx = get_overlapping_region(x - sz, x + sz, 0, W-1)

            # print 'outy,outx', outy,outx
            # print 'iny,inx', iny,inx
            
            subimg = np.zeros((sz*2+1, sz*2+1,planes), img.dtype)
            subimg[outy,outx] = img[iny,inx]

            ell = EllipseE(p.shapeexp_r, p.shapeexp_e1, p.shapeexp_e2)
            angle = np.rad2deg(ell.theta)
            print 'Angle:', angle
            
            subimg = scipy.ndimage.rotate(subimg, angle + 90, reshape=False)
            subimg = subimg[9:-9, 9:-9, :]
            print 'image shape', subimg.shape

            dimshow(subimg, ticks=False)
    ps.savefig()
    
