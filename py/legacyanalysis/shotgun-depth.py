from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np
import sys

from astrometry.util.fits import *
from astrometry.util.plotutils import *
from astrometry.libkd.spherematch import match_radec

from tractor.sfd import SFDMap

from legacypipe.common import *

def main():
    ps = PlotSequence('shotgun')

    survey = LegacySurveyData()
    C = fits_table('survey-ccds-annotated.fits')
    print(len(C), 'CCDs')
    C.cut(C.photometric)
    C.cut(C.blacklist_ok)
    print(len(C), 'photometric and not blacklisted')

    # HACK
    print('FIXME not cutting on DECALS')
    #C.cut(C.tilepass > 0)
    #print(len(C), 'taken by DECaLS')

    targets = dict(g=24.0, r=23.4, z=22.5)

    def ivtomag(iv, nsigma=5.):
        return -2.5 * (np.log10(nsigma / np.sqrt(iv)) - 9)

    def band_index(band):
        allbands = 'ugrizY'
        return allbands.index(band)

    ccmap = dict(g='g', r='r', z='m')

    ceil_exptime = dict(g=125., r=125., z=250.)
    
    #plt.clf()

    bands = 'grz'
    for band in bands:
        tmag = targets[band]
        print()
        print(band, 'band, target depth', tmag)
        ccds = C[C.filter == band]
        ccdarea = (2046*4094*(0.262/3600.)**2)
        print(len(ccds), 'CCDs, total exptime', np.sum(ccds.exptime),
              '(mean %.1f)' % np.mean(ccds.exptime), 'total area',
              len(ccds)*ccdarea, 'sq.deg')
        detsig1 = ccds.sig1 / ccds.galnorm_mean
        totiv = np.sum(1. / detsig1**2)
        # depth we would have if we had all exposure time in one CCD
        # print('5-sigma galaxy depth if concentrated in one CCD:', ivtomag(totiv))
        # # mean depth
        # print('5-sigma galaxy depth if spread equally among', len(ccds), 'CCDs:', ivtomag(totiv / len(ccds)))
        # print('vs median depth', np.median(ccds.galdepth))
        # print('5-sigma galaxy depth if spread equally among %i/2' % (len(ccds)), 'CCDs:', ivtomag(totiv / (len(ccds)/2)))
        # print('5-sigma galaxy depth if spread equally among %i/3' % (len(ccds)), 'CCDs:', ivtomag(totiv / (len(ccds)/3)))
        # spread over 6000 sq deg
        sqdeg = 6000
        avgiv = totiv * ccdarea / sqdeg
        #print('5-sigma galaxy depth if spread over', sqdeg, 'sqdeg:', ivtomag(avgiv))

        tflux = 10.**(tmag / -2.5 + 9)
        tiv = 1. / (tflux / 5)**2
        #print('Fraction of', sqdeg, 'sqdeg survey complete:', avgiv / tiv)

        iband = band_index(band)
        ext = ccds.decam_extinction[:,iband]
        medext = np.median(ext)
        print('With extinction (median %.2f mag):' % medext)

        transmission = 10.**(-ext / 2.5)

        detsig1 = ccds.sig1 / ccds.galnorm_mean / transmission
        totiv = np.sum(1. / detsig1**2)
        # depth we would have if we had all exposure time in one CCD
        print('5-sigma galaxy depth if concentrated in one CCD: %.3f' % ivtomag(totiv))
        # mean depth
        print('5-sigma galaxy depth if spread equally among', len(ccds), 'CCDs: %.3f' % ivtomag(totiv / len(ccds)))
        print('vs median depth: %.3f' % np.median(ccds.galdepth - ext))
        print('5-sigma galaxy depth if spread equally among %i/2' % (len(ccds)), 'CCDs: %.3f' % ivtomag(totiv / (len(ccds)/2)))
        print('5-sigma galaxy depth if spread equally among %i/3' % (len(ccds)), 'CCDs: %.3f' % ivtomag(totiv / (len(ccds)/3)))
        # spread over 6000 sq deg
        sqdeg = 6000
        avgiv = totiv * ccdarea / sqdeg
        print('5-sigma galaxy depth if spread over', sqdeg, 'sqdeg: %.3f' % ivtomag(avgiv))
        print('Fraction of', sqdeg, 'sqdeg survey complete: %.3f' % (avgiv / tiv))

        # plt.hist(ccds.exptime, range=(0,250), bins=50, histtype='step', color=ccmap[band])

        # I = np.flatnonzero(ccds.exptime < (ceil_exptime[band] - 1.))
        # ccds.cut(I)
        # print('Cutting out exposures with ceil exposure time:', len(ccds))
        # 
        # plt.hist(ccds.exptime, bins=25, histtype='step', color=ccmap[band],
        #          linestyle='dotted', linewidth=3, alpha=0.3)
        # 
        # transmission = transmission[I]
        # ext = ext[I]
        # 
        # detsig1 = ccds.sig1 / ccds.galnorm_mean / transmission
        # totiv = np.sum(1. / detsig1**2)
        # # depth we would have if we had all exposure time in one CCD
        # print('5-sigma galaxy depth if concentrated in one CCD:', ivtomag(totiv))
        # # mean depth
        # print('5-sigma galaxy depth if spread equally among', len(ccds), 'CCDs:', ivtomag(totiv / len(ccds)))
        # print('vs median depth', np.median(ccds.galdepth - ext))
        # print('5-sigma galaxy depth if spread equally among %i/2' % (len(ccds)), 'CCDs:', ivtomag(totiv / (len(ccds)/2)))
        # print('5-sigma galaxy depth if spread equally among %i/3' % (len(ccds)), 'CCDs:', ivtomag(totiv / (len(ccds)/3)))
        # # spread over 6000 sq deg
        # sqdeg = 6000
        # avgiv = totiv * ccdarea / sqdeg
        # print('5-sigma galaxy depth if spread over', sqdeg, 'sqdeg:', ivtomag(avgiv))
        # print('Fraction of', sqdeg, 'sqdeg survey complete:', avgiv / tiv)

        
    # plt.xlabel('Exposure time (s)')
    # ps.savefig()
        
    print()

    dra  = 4094 / 2. / 3600 * 0.262
    ddec = 2046 / 2. / 3600 * 0.262

    ralo  = max(  0, min(C.ra  - dra / np.cos(np.deg2rad(C.dec))))
    rahi  = min(360, max(C.ra  + dra / np.cos(np.deg2rad(C.dec))))
    declo = max(-90, min(C.dec - ddec))
    dechi = min( 90, max(C.dec + ddec))

    # brick 0001m002
    #ralo,rahi = 0., 0.25
    #declo,dechi = -0.375, -0.125

    #ralo,  rahi  = 0, 1
    #declo, dechi = 0, 1

    ralo,  rahi  = 0, 0.5
    declo, dechi = 0, 0.5
    
    print('RA,Dec range', (ralo, rahi), (declo, dechi))

    N = 10000

    nbatch = 1000
    rr,dd = [],[]
    ntotal = 0
    while ntotal < N:
        ru = np.random.uniform(size=nbatch)
        d = np.random.uniform(low=declo, high=dechi, size=nbatch)
        # Taper the accepted width in RA based on Dec
        cosd = np.cos(np.deg2rad(d))
        I = np.flatnonzero(ru < cosd)
        if len(I) == 0:
            continue
        r = ralo + (rahi - ralo) * ru[I]/cosd[I]
        d = d[I]
        rr.append(r)
        dd.append(d)
        ntotal += len(r)
        print('Kept', len(r), 'of', nbatch)

    ra  = np.hstack(rr)
    dec = np.hstack(dd)
    del rr
    del dd
    ra  = ra [:N]
    dec = dec[:N]

    print('RA,Dec ranges of samples:', (ra.min(), ra.max()), (dec.min(), dec.max()))
    
    # plt.clf()
    # plt.plot(ra, dec, 'b.', alpha=0.1)
    # ps.savefig()

    # CCD size
    margin = 10 * 0.262 / 3600.
    # C.dra is in degrees on the sphere, not delta-RA
    # dra = C.dra / np.cos(np.deg2rad(C.dec))
    # ccds = C[(C.ra  +   dra  + margin >  ralo) *
    #          (C.ra  -   dra  - margin <  rahi) *
    #          (C.dec + C.ddec + margin > declo) *
    #          (C.dec - C.ddec - margin < dechi)]

    I,J,d = match_radec(C.ra, C.dec,
                        (ralo+rahi)/2., (declo+dechi)/2.,
                        degrees_between(ralo,declo, rahi,dechi)/2. + 1.,
                        nearest=True)
    ccds = C[I]

    print(len(ccds), 'nearby')
    assert(len(ccds))

    print('RA range', ccds.ra.min(), ccds.ra.max())

    radius = np.hypot(2046, 4096) / 2. * 0.262 / 3600 * 1.1
    II = match_radec(ccds.ra, ccds.dec, ra, dec,
                     radius, indexlist=True)
    #print('Matching:', II)

    depthrange = [20,25]
    depthbins  = 100
    depthbins2  = 500
    
    galdepths = dict([(b, np.zeros(len(ra))) for b in bands])

    iccds = []
    
    for iccd,I in enumerate(II):
        if I is None:
            continue
        ccd = ccds[iccd]
        #print('Matched to CCD %s: %i' % (ccd.expid, len(I)))
        # Actually inside CCD RA,Dec box?
        r = ra[I]
        d = dec[I]

        # print('degrees_between: ccd ra,dec', ccd.ra, ccd.dec)
        # for ri,di in zip(r, degrees_between(r, ccd.dec + np.zeros_like(r),
        #                                     ccd.ra, ccd.dec)):
        #     print('ri: di', ri, di)

        J = np.flatnonzero((degrees_between(r, ccd.dec+np.zeros_like(r),
                                            ccd.ra, ccd.dec) < dra) *
                                            (np.abs(d - ccd.dec) < ddec))
        #print('Actually inside CCD RA,Dec box:', len(J))
        if len(J) == 0:
            continue
        I = np.array(I)[J]

        print('%i samples in CCD %s' % (len(I), ccd.expid))
        
        # j = J[0]
        # print('deg between', degrees_between(r[j], ccd.dec,
        #                                      ccd.ra,  ccd.dec), 'vs', dra)
        # print('  dec', d[j], 'dist', np.abs(d[j] - ccd.dec), 'vs', ddec)
        
        iccds.append((iccd,I[0]))
        
        band = ccd.filter
        gd = galdepths[band]
        print('Gal depth in', band, ':', ccd.galdepth)
        if not np.isfinite(ccd.galdepth):
            print('  sig1', ccd.sig1)
            print('  galnorm', ccd.galnorm_mean)
            print('  psfnorm', ccd.psfnorm_mean)
            print('  seeing', ccd.seeing)
            print('  fwhm', ccd.fwhm)
            ccd.about()
            print('  ccd:', ccd)

            im = survey.get_image_object(ccd)
            tim = im.get_tractor_image(splinesky=True, pixPsf=True)
            print('  tim:', tim)
            
        # mag -> 5sig1 -> iv
        cgd = 10.**((ccd.galdepth - 22.5) / -2.5)
        cgd = 1. / cgd**2
        gd[I] += cgd

    plt.clf()

    # plt.plot(ra, dec, 'k.', alpha=0.1)
    
    # C1 = fits_table('decals-ccds-annotated.fits')
    # ii,jj,dd = match_radec(C1.ra, C1.dec, (ralo+rahi)/2.,(declo+dechi)/2,0.5)
    # C1.cut(ii)
    # C1.ra -= (C1.ra > 270)*360
    # for ccd in C1:
    #     r,dr = ccd.ra,  ccd.dra / np.cos(np.deg2rad(ccd.dec))
    #     d,dd = ccd.dec, ccd.ddec
    #     sty = dict(color='k', alpha=0.1)
    #     if not (ccd.photometric and ccd.blacklist_ok):
    #         sty = dict(color='c', lw=3)
    #     plt.plot([r-dr, r+dr, r+dr, r-dr, r-dr], [d-dd,d-dd,d+dd,d+dd,d-dd],
    #              '-', **sty)
    
    for iccd,i in iccds:
        ccd = ccds[iccd]
        r,dr = ccd.ra,  dra / np.cos(np.deg2rad(ccd.dec))
        # HACK
        r = r + (r > 270)*-360.

        d,dd = ccd.dec, ddec
        plt.plot([r-dr, r+dr, r+dr, r-dr, r-dr], [d-dd,d-dd,d+dd,d+dd,d-dd],
                 '-', color=ccmap[ccd.filter], alpha=0.5)

        plt.plot(ra[i], dec[i], 'k.')
        
    plt.plot([ralo,ralo,rahi,rahi,ralo],[declo,dechi,dechi,declo,declo],
             'k-', lw=2)
    #plt.axis([-0.1, 0.6, -0.1, 0.6])
    plt.axis([-0.5, 1., -0.5, 1.])
    plt.xlabel('RA')
    plt.ylabel('Dec')    
    ps.savefig()
        
    for band in bands:
        gd = galdepths[band]
        # iv -> 5sig1 -> mag
        gd = 1. / np.sqrt(gd)
        gd = -2.5 * (np.log10(gd) - 9.)

        plt.clf()
        plt.hist(gd, range=depthrange, bins=depthbins, histtype='step',
                 color=ccmap[band])
        plt.xlabel('5-sigma Galaxy depth (mag)')
        plt.title('%s band' % band)
        ps.savefig()
            


    print('Reading extinction values for sample points...')
    sfd = SFDMap()
    filts = ['%s %s' % ('DES', f) for f in bands]
    ebv,extinction = sfd.extinction(filts, ra, dec, get_ebv=True)
    print('Extinction:', extinction.shape)
    
    B = survey.get_bricks_readonly()
    I = survey.bricks_touching_radec_box(None, ralo, rahi, declo, dechi)
    B.cut(I)
    print(len(B), 'bricks touching RA,Dec box')

    depth_hists = {}
    depth_hists_2 = {}
    
    for brick in B:
        print('Brick', brick.brickname)
        I = np.flatnonzero((ra  > brick.ra1 ) * (ra  < brick.ra2) *
                           (dec > brick.dec1) * (dec < brick.dec2))
        print(len(I), 'samples in brick')
        if len(I) == 0:
            continue
        
        wcs = wcs_for_brick(brick)
        ok,x,y = wcs.radec2pixelxy(ra[I], dec[I])
        x = np.round(x - 1).astype(int)
        y = np.round(y - 1).astype(int)
        
        for iband,band in enumerate(bands):
            fn = survey.find_file('nexp', brick=brick.brickname, band=band)
            print('Reading', fn)
            if not os.path.exists(fn):
                print('Missing:', fn)
                continue
            nexp = fitsio.read(fn)
            nexp = nexp[y, x]

            ext = extinction[I, iband]
            
            fn = survey.find_file('galdepth', brick=brick.brickname, band=band)
            print('Reading', fn)
            galdepth = fitsio.read(fn)
            galdepth = galdepth[y, x]
            # iv -> mag
            galdepth = -2.5 * (np.log10(5. / np.sqrt(galdepth)) - 9)
            # extinction-corrected
            galdepth -= ext
            
            un = np.unique(nexp)
            print('Numbers of exposures:', un)

            for ne in un:
                if ne == 0:
                    continue
                J = np.flatnonzero(nexp == ne)
                gd = galdepth[J]
                key = (ne, band)
                H = depth_hists.get(key, 0)
                h,e = np.histogram(gd, range=depthrange, bins=depthbins)
                H = h + H
                depth_hists[key] = H

                H = depth_hists_2.get(key, 0)
                h,e = np.histogram(gd, range=depthrange, bins=depthbins2)
                H = h + H
                depth_hists_2[key] = H

    dlo,dhi = depthrange
    left = dlo + np.arange(depthbins) * (dhi-dlo) / float(depthbins)
    binwidth = left[1]-left[0]
    # for k,H in depth_hists.items():
    #     plt.clf()
    #     plt.bar(left, H, width=binwidth)
    #     plt.xlabel('Galaxy depth (mag)')
    #     (ne,band) = k
    #     plt.title('%s band, %i exposures' % (band, ne))
    #     plt.xlim(dlo, dhi)
    #     fn = 'depth-%s-%i.png' % (band, ne)
    #     plt.savefig(fn)
    #     print('Wrote', fn)

    rainbow = ['r', '#ffa000', 'y', 'g', 'b', 'm']
        
    for band in bands:
        plt.clf()
        for ne in range(1,10):
            key = (ne, band)
            if not key in depth_hists:
                continue
            H = depth_hists[key]
            print('hist length:', len(H))
            plt.plot(np.vstack((left, left+binwidth)).T.ravel(), np.repeat(H, 2), '-',
                     color=rainbow[ne-1 % len(rainbow)], label='%i exp' % ne)
        plt.title('%s band' % band)
        plt.xlabel('Galaxy depth (mag)')
        plt.legend(loc='upper left')
        ps.savefig()


    left2 = dlo + np.arange(depthbins2) * (dhi-dlo) / float(depthbins2)
    binwidth2 = left2[1]-left2[0]

    for band in bands:
        hsum = 0
        for ne in range(1,10):
            key = (ne, band)
            if not key in depth_hists_2:
                continue
            hsum += depth_hists_2[key]

        print('%s band:' % band)
            
        print('Total number of counts in histogram:', sum(hsum))
        # [-1::-1] = reversed
        hsum = np.cumsum(hsum[-1::-1])[-1::-1]
        hsum = hsum * 100. / float(N)

        plt.clf()
        #plt.plot(left2+binwidth/2, hsum, 'k-')
        plt.plot(left2, hsum, 'k-')
        plt.xlabel('Galaxy depth (mag)')
        plt.ylabel('Cumulative fraction (%)')
        plt.title('%s band' % band)
        # 90% to full depth
        # 95% to full depth - 0.3 mag
        # 98% to full depth - 0.6 mag
        for y,x,c in [
                (90, targets[band], 'r'),
                (95, targets[band] - 0.3, 'g'),
                (98, targets[band] - 0.6, 'b')]:
            xf = (x - dlo) / (dhi - dlo)
            yf = y / 100.
            plt.axvline(x, ymax=yf, color=c, lw=3, alpha=0.3)
            plt.axhline(y, xmax=xf, color=c, lw=3, alpha=0.3)

            #mid = left2 + binwidth2/2
            
            print('target  : %.1f %% deeper than %.2f' % (y, x))

            # ??? left or midpoint?
            ibin = np.flatnonzero(left2 > x)[0]
            pct = hsum[ibin]
            status = ' '*20
            if pct >= y:
                status += '-> pass'
            else:
                status += '-> FAIL'

                plt.axhline(pct, color=c, alpha=0.5) #xmax=xf, 
                
            print('achieved: %.1f %% %s' % (pct, status)) # deeper than %.2f' % (hsum[ibin], x))
            ii = np.flatnonzero(hsum > y)
            if len(ii):
                ibin = ii[-1]
                print('          %.1f %% deeper than %.2f' % (y, left2[ibin]))
            else:
                print('          Total coverage only %.1f %%' % max(hsum))
            print()
            
            
        plt.xlim(dlo, dhi)
        plt.ylim(0., 100.)
        ps.savefig()



        

        
if __name__ == '__main__':
    main()
