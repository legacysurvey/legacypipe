from __future__ import print_function
import matplotlib
matplotlib.use('Agg')

import os
from astrometry.util.fits import *
from astrometry.util.file import *
from astrometry.util.util import Tan
import numpy as np
import pylab as plt
from legacypipe.survey import LegacySurveyData
from collections import Counter

def make_pickle_file(pfn, derivs=False, agn=False):
    survey = LegacySurveyData()
    ccds = survey.get_ccds()

    brickname = '0364m042'
    
    bricks = survey.get_bricks()
    brick = bricks[bricks.brickname == brickname][0]
    print('Brick', brick)

    catfn = survey.find_file('tractor', brick=brickname)
    print('Reading catalog from', catfn)
    cat = fits_table(catfn)
    print(len(cat), 'catalog entries')
    cat.cut(cat.brick_primary)
    print(len(cat), 'brick primary')

    '''
    BRICKNAM     BRICKID BRICKQ    BRICKROW    BRICKCOL                      RA                     DEC
    0364m042      306043      3         343         145        36.4255910987483       -4.25000000000000

    RA1                     RA2                    DEC1                    DEC2
    36.3004172461752        36.5507649513213       -4.37500000000000       -4.12500000000000
    '''

    rlo,rhi = brick.ra1, brick.ra2
    dlo,dhi = brick.dec1, brick.dec2
    #rlo,rhi = 36.4, 36.5
    #dlo,dhi = -4.4, -4.3

    ra,dec = (rlo+rhi)/2., (dlo+dhi)/2.

    ## optional
    cat.cut((cat.ra > rlo) * (cat.ra < rhi) * (cat.dec > dlo) * (cat.dec < dhi))
    print('Cut to', len(cat), 'catalog objects in RA,Dec box')

    lightcurves = dict([((brickname, oid), []) for oid in cat.objid])

    # close enough to equator to ignore cos(dec)
    dra  = 4096 / 2. * 0.262 / 3600.
    ddec = 2048 / 2. * 0.262 / 3600.

    ccds.cut((np.abs(ccds.ra  - ra ) < (rhi-rlo)/2. + dra) *
             (np.abs(ccds.dec - dec) < (dhi-dlo)/2. + ddec))
    print('Cut to', len(ccds), 'CCDs overlapping brick')

    ### HACK
    #ccds = ccds[:50]

    for i,(expnum,ccdname) in enumerate(zip(ccds.expnum, ccds.ccdname)):
        ee = '%08i' % expnum

        flavor = 'vanilla'
        cols = ['brickname','objid','camera','expnum','ccdname','mjd','filter','flux','flux_ivar']
        if derivs:
            flavor = 'derivs'
            cols.extend(['flux_dra','flux_ddec','flux_dra_ivar','flux_ddec_ivar'])
        if agn:
            flavor = 'agn'
            cols.extend(['flux_agn', 'flux_agn_ivar'])

        fn = 'forced/%s/%s/%s/forced-decam-%s-%s.fits' % (flavor, ee[:5], ee, ee, ccdname)
        if not os.path.exists(fn):
            print('WARNING: missing:', fn)
            continue
        T = fits_table(fn)
        print(i+1, 'of', len(ccds), ':', len(T), 'in', fn)
        T.cut(T.brickname == brickname)
        print(len(T), 'in brick', brickname)
        found = 0
        for t in T: #oid,expnum,ccdname,mjd,filter,flux,fluxiv in zip(T.objid, T.expnum, T.ccdname, T.mjd, T.filter, T.flux, T.flux_ivar):
            lc = lightcurves.get((t.brickname, t.objid), None)
            if lc is None:
                continue
            found += 1
            #lc.append((expnum, ccdname, mjd, filter, flux, fluxiv))
            lc.append([t.get(c) for c in cols])
        print('Matched', found, 'sources to light curves')

    #pickle_to_file(lightcurves, pfn)

    ll = {}
    for k,v in lightcurves.items():
        if len(v) == 0:
            continue
        T = fits_table()
        # T.expnum = np.array([vv[0] for vv in v])
        # T.ccdname= np.array([vv[1] for vv in v])
        # T.mjd    = np.array([vv[2] for vv in v])
        # T.filter = np.array([vv[3] for vv in v])
        # T.flux   = np.array([vv[4] for vv in v])
        # T.fluxiv = np.array([vv[5] for vv in v])
        for i,c in enumerate(cols):
            T.set(c, np.array([vv[i] for vv in v]))
        ll[k] = T
    pickle_to_file(ll, pfn)


def plot_light_curves(pfn, ucal=False):
    lightcurves = unpickle_from_file(pfn)

    if ucal:
        tag = 'ucal-'
    else:
        tag = ''

    survey = LegacySurveyData()
    brickname = '0364m042'
    catfn = survey.find_file('tractor', brick=brickname)
    print('Reading catalog from', catfn)
    cat = fits_table(catfn)
    print(len(cat), 'catalog entries')
    cat.cut(cat.brick_primary)
    print(len(cat), 'brick primary')

    I = []
    for i,oid in enumerate(cat.objid):
        if (brickname,oid) in lightcurves:
            I.append(i)
    I = np.array(I)
    cat.cut(I)
    print('Cut to', len(cat), 'with light curves')

    S = fits_table('specObj-dr12-trim-2.fits')

    from astrometry.libkd.spherematch import match_radec
    I,J,d = match_radec(S.ra, S.dec, cat.ra, cat.dec, 2./3600.)
    print('Matched', len(I), 'to spectra')


    plt.subplots_adjust(hspace=0)

    movie_jpegs = []
    movie_wcs = None

    for i in range(28):
        fn = os.path.join('des-sn-movie', 'epoch%i' % i, 'coadd', brickname[:3],
                          brickname, 'legacysurvey-%s-image.jpg' % brickname)
        print(fn)
        if not os.path.exists(fn):
            continue

        img = plt.imread(fn)
        img = np.flipud(img)
        h,w,d = img.shape

        fn = os.path.join('des-sn-movie', 'epoch%i' % i, 'coadd', brickname[:3],
                          brickname, 'legacysurvey-%s-image-r.fits' % brickname)
        if not os.path.exists(fn):
            continue
        wcs = Tan(fn)

        movie_jpegs.append(img)
        movie_wcs = wcs


    plt.figure(figsize=(8,6), dpi=100)
    n = 0

    fluxtags = [('flux','flux_ivar', '', 'a')]
    if ucal:
        fluxtags.append(('uflux', 'uflux_ivar', ': ucal', 'b'))

    for oid,ii in zip(cat.objid[J], I):
        print('Objid', oid)
        spec = S[ii]
        k = (brickname, oid)
        v = lightcurves[k]

        # Cut bad CCDs
        v.cut(np.array([e not in [230151, 230152, 230153] for e in v.expnum]))

        plt.clf()
        print('obj', k, 'has', len(v), 'measurements')
        T = v

        for fluxtag,fluxivtag,fluxname,plottag in fluxtags:
            plt.clf()
    
            filts = np.unique(T.filter)
            for i,f in enumerate(filts):
                from tractor.brightness import NanoMaggies
                
                plt.subplot(len(filts),1,i+1)

                fluxes = np.hstack([T.get(ft[0])[T.filter == f] for ft in fluxtags])
                fluxes = fluxes[np.isfinite(fluxes
)]
                mn,mx = np.percentile(fluxes, [5,95])
                print('Flux percentiles for filter', f, ':', mn,mx)
                # note swap
                mn,mx = NanoMaggies.nanomaggiesToMag(mx),NanoMaggies.nanomaggiesToMag(mn)
                print('-> mags', mn,mx)
    
                cut = (T.filter == f) * (T.flux_ivar > 0)
                if ucal:
                    cut *= np.isfinite(T.uflux)
                I = np.flatnonzero(cut)
                                  
                print('  ', len(I), 'in', f, 'band')
                I = I[np.argsort(T.mjd[I])]
                mediv = np.median(T.flux_ivar[I])
                # cut really noisy ones
                I = I[T.flux_ivar[I] > 0.25 * mediv]
    
                #plt.plot(T.mjd[I], T.flux[I], '.-', color=dict(g='g',r='r',z='m')[f])
                # plt.errorbar(T.mjd[I], T.flux[I], yerr=1/np.sqrt(T.fluxiv[I]),
                #              fmt='.-', color=dict(g='g',r='r',z='m')[f])
                #plt.errorbar(T.mjd[I], T.flux[I], yerr=1/np.sqrt(T.fluxiv[I]),
                #             fmt='.', color=dict(g='g',r='r',z='m')[f])
    
                # if ucal:
                #     mag,dmag = NanoMaggies.fluxErrorsToMagErrors(T.flux[I], T.flux_ivar[I])
                # else:
                #     mag,dmag = NanoMaggies.fluxErrorsToMagErrors(T.uflux[I], T.uflux_ivar[I])
                mag,dmag = NanoMaggies.fluxErrorsToMagErrors(T.get(fluxtag)[I],
                                                             T.get(fluxivtag)[I])
    
                plt.errorbar(T.mjd[I], mag, yerr=dmag,
                             fmt='.', color=dict(g='g',r='r',z='m')[f])
                #yl,yh = plt.ylim()
                #plt.ylim(yh,yl)
                plt.ylim(mx, mn)

                plt.ylabel(f)
    
                if i+1 < len(filts):
                    plt.xticks([])
                #plt.yscale('symlog')
    
    
            outfn = 'cutout_%.4f_%.4f.jpg' % (spec.ra, spec.dec)
            if not os.path.exists(outfn):
                url = 'http://legacysurvey.org/viewer/jpeg-cutout/?ra=%.4f&dec=%.4f&zoom=14&layer=sdssco&size=128' % (spec.ra, spec.dec)
                cmd = 'wget -O %s "%s"' % (outfn, url)
                print(cmd)
                os.system(cmd)
            pix = plt.imread(outfn)
            h,w,d = pix.shape
            fig = plt.gcf()
    
            #print('fig bbox:', fig.bbox)
            #print('xmax, ymax', fig.bbox.xmax, fig.bbox.ymax)
            #plt.figimage(pix, 0, fig.bbox.ymax - h, zorder=10)
            #plt.figimage(pix, 0, fig.bbox.ymax, zorder=10)
            #plt.figimage(pix, fig.bbox.xmax - w, fig.bbox.ymax, zorder=10)
            plt.figimage(pix, fig.bbox.xmax - (w+2), fig.bbox.ymax - (h+2), zorder=10)
    
            plt.suptitle('SDSS spectro object: %s at (%.4f, %.4f)%s' % (spec.label.strip(), spec.ra, spec.dec, fluxname))
            plt.savefig('forced-%s%i-%s.png' % (tag, n, plottag))

        ok,x,y = movie_wcs.radec2pixelxy(spec.ra, spec.dec)
        x = int(np.round(x-1))
        y = int(np.round(y-1))
        sz = 32

        plt.clf()
        plt.subplots_adjust(hspace=0, wspace=0)
        k = 1
        for i,img in enumerate(movie_jpegs):
            stamp = img[y-sz:y+sz+1, x-sz:x+sz+1]            

            plt.subplot(5, 6, k)
            plt.imshow(stamp, interpolation='nearest', origin='lower')
            plt.xticks([]); plt.yticks([])
            k += 1
        plt.suptitle('SDSS spectro object: %s at (%.4f, %.4f): DES images' % (spec.label.strip(), spec.ra, spec.dec))
        plt.savefig('forced-%s%i-c.png' % (tag, n))

        n += 1

def ubercal(pfn):
    lightcurves = unpickle_from_file(pfn)
    print(len(lightcurves), 'light curves')
    k = lightcurves.keys()[0]
    print('key:', k)
    print('value:', lightcurves[k])
    lightcurves[k].about()
    keys = lightcurves.keys()

    # medfluxes = {}
    # nfluxes = []
    # ccds = set()
    # for k in keys:
    #     v = lightcurves[k]
    #     medfluxes[k] = np.median(v.flux)
    #     nfluxes.append(len(v))
    #     ccds = ccds.union(set(zip(v.camera, v.expnum, v.ccdname)))
    #print('N fluxes: deciles', np.percentile(nfluxes, np.arange(0, 101, 10)))
    #print('N ccds:', len(ccds))

    #dfluxes = dict([(ccd,[]) for ccd in ccds])
    dfluxes = {}
    for j,k in enumerate(keys):
        if j % 1000 == 1:
            print('source', j-1)
        v = lightcurves[k]
        #dflux = v.flux / medfluxes[k]
        #dflux = v.flux / np.median(v.flux)
        meds = dict([(f, np.median(v.flux[v.filter == f])) for f in np.unique(v.filter)])
        print('medians for', k, ':', meds)
        for i,ccd in enumerate(zip(v.camera, v.expnum, v.ccdname)):
            if not ccd in dfluxes:
                dfluxes[ccd] = []
            dfluxes[ccd].append(v.flux[i] / meds[v.filter[i]])

    # for i,k in enumerate(dfluxes.keys()[:10]):
    #     plt.clf()
    #     df = dfluxes[k]
    #     df = np.sort(df)
    #     plt.hist(df, 20, range=np.percentile(df, [2,98]))
    #     plt.xlabel('flux ratio vs median')
    #     plt.title('CCD ' + str(k))
    #     plt.savefig('dflux-%i.png' % i)

    # ubercal flux factor corrections;
    # factor by which the flux in CCD k exceeds the median flux
    dflux = dict([(k, np.median(d)) for k,d in dfluxes.items()])

    print('dflux:', dflux)

    df = np.array(dflux.values())
    df = df[np.isfinite(df)]

    plt.clf()
    plt.hist(df, 20)
    plt.xlabel('Flux factors per CCD')
    plt.savefig('forced-fluxfactor.png')

    plt.clf()
    df = dfluxes.values()
    plt.plot([len(d) for d in df], [np.median(d) for d in df], 'b.')
    plt.xlabel('Number of sources in CCD')
    plt.ylabel('Flux factor (per CCD)')
    plt.savefig('forced-fluxfactor2.png')

    for j,k in enumerate(keys):
        if j % 1000 == 1:
            print('source', j-1)
        v = lightcurves[k]
        # For each object, look up the CCD and apply the flux factor correction
        df = []
        for i,ccd in enumerate(zip(v.camera, v.expnum, v.ccdname)):
            df.append(dflux[ccd])
        v.dflux = np.array(df)
        # Correct by 
        v.uflux = v.flux / v.dflux
        v.uflux_ivar = v.flux_ivar * v.dflux**2

    outfn = pfn.replace('.pickle', '-ucal.pickle')
    pickle_to_file(lightcurves, outfn)
    print('Wrote', outfn)


if __name__ == '__main__':
    import sys

    # pfn = 'pickles/lightcurves-vanilla.pickle'
    # ubercal(pfn)

    pfn = 'pickles/lightcurves-vanilla-ucal.pickle'
    plot_light_curves(pfn, 'ucal')

    sys.exit(0)




    # pfn = 'pickles/lightcurves.pickle'
    # plot_light_curves(pfn)

    # pfn = 'pickles/lightcurves-vanilla.pickle'
    # if not os.path.exists(pfn):
    #     make_pickle_file(pfn)

    # pfn = 'pickles/lightcurves-derivs.pickle'
    # if not os.path.exists(pfn):
    #     make_pickle_file(pfn, derivs=True)
    
    # pfn = 'pickles/lightcurves-agn.pickle'
    # if not os.path.exists(pfn):
    #     make_pickle_file(pfn, agn=True)

