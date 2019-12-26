from tractor.galaxy import *
from tractor import *
from legacypipe.survey import *
from legacypipe.coadds import quick_coadds
from tractor.devagn import *
from tractor.sersic import SersicGalaxy, SersicIndex
from tractor.seragn import SersicAgnGalaxy
from astrometry.util.file import pickle_to_file, unpickle_from_file

import pylab as plt
import numpy as np

import matplotlib.gridspec as gridspec
import os

ima = dict(interpolation='nearest', origin='lower')

class Duck(object):
    pass

def flux_string(br):
    s = []
    for band in 'grz':
        flux = br.getFlux(band)
        if flux <= 0:
            s.append('%s=(%.2f nmy)' % (band, flux))
        else:
            s.append('%s=%.2f' % (band, NanoMaggies.nanomaggiesToMag(flux)))
    s = ', '.join(s)
    return s

def showmods(tims,mods):
    #plt.figure(figsize=(10,6))
    cols = len(tims) // 3
    panels = [((tim.getImage()-mod)*tim.getInvError()) for tim,mod in list(zip(tims, mods))]
    h = min([p.shape[0] for p in panels])
    w = min([p.shape[1] for p in panels])
    panels = [p[:h,:w] for p in panels]
    stack = []
    while len(panels):
        stack.append(np.hstack(panels[:cols]))
        panels = panels[cols:]
    stack = np.vstack((stack))
    plt.imshow(stack, vmin=-2, vmax=2, **ima)
    plt.xticks(w * (0.5 + np.arange(cols)), np.arange(cols))
    plt.yticks(h * (0.5 + np.arange(3)), ['g','r','z'])

def showresid(tims,mods,wcs,bands='grz'):
    co,_ = quick_coadds(tims, bands, wcs)
    comod,_ = quick_coadds(tims, bands, wcs, images=mods)    
    plt.imshow(np.flipud(get_rgb([i-m for i,m in zip(co,comod)], bands)))
    plt.xticks([]); plt.yticks([])
    return co,comod
    
def showboth(tims,mods,wcs,bands):
    fig = plt.figure(num=1, figsize=(14,6), constrained_layout=True)
    fig.clf()
    gs = fig.add_gridspec(3, 4)
    ax = fig.add_subplot(gs[:, :3])

    cols = len(tims) // 3
    rows = 3
    panels = [((tim.getImage()-mod)*tim.getInvError()) for tim,mod in list(zip(tims, mods))]
    h = min([p.shape[0] for p in panels])
    w = min([p.shape[1] for p in panels])
    panels = [p[:h,:w] for p in panels]
    stack = []
    #hzero = np.zeros((h,1))
    while len(panels):
        # hs = [hzero]
        # for p in panels[:cols]:
        #     hs.extend([p, hzero])
        # hs = np.hstack(hs)
        # hh,hw = hs.shape
        # if len(stack) == 0:
        #     wzero = np.zeros((1,hw))
        #     stack.append(wzero)
        # stack.append(hs)
        # stack.append(wzero)
        stack.append(np.hstack(panels[:cols]))
        panels = panels[cols:]
    stack = np.vstack((stack))
    ax.imshow(stack, vmin=-2, vmax=2, **ima)
    xl,yl = ax.get_xlim(), ax.get_ylim()
    #a = ax.get_axis()
    for c in range(cols+1):
        plt.axvline(c * w, color='k')
    for r in range(rows+1):
        plt.axhline(r * h, color='k')
    #ax.set_axis(a)
    ax.set_xlim(xl)
    ax.set_ylim(yl)
    #xl,yl = ax.get_xlim(), ax.get_ylim()
    ax.set_xticks(w * (0.5 + np.arange(cols)))
    ax.set_xticklabels(np.arange(cols))
    ax.set_yticks(h * (0.5 + np.arange(3)))
    ax.set_yticklabels(['g','r','z'])

    ax = fig.add_subplot(gs[2, 3])
    co,comod = showresid(tims, mods, wcs)
    ax = fig.add_subplot(gs[0, 3])
    plt.imshow(np.flipud(get_rgb(co, bands)))
    plt.xticks([]); plt.yticks([])
    ax = fig.add_subplot(gs[1, 3])
    plt.imshow(np.flipud(get_rgb(comod, bands)))
    plt.xticks([]); plt.yticks([])

def main():

    if os.path.exists('results.pickle'):
        results = unpickle_from_file('results.pickle')
    else:
        results = []


    
    for isrc,(ra,dec,brickname) in enumerate([
        (0.2266, 3.9822, '0001p040'),
        (7.8324, 1.2544, '0078p012'),
        (1.1020, 3.9040, '0011p040'),
        (7.3252, 4.6847, '0073p047'),
        (3.1874, 3.9724, '0031p040'),
        (9.5112, 4.6934, '0094p047'),
        (4.4941, 1.1058, '0043p010'),
        (3.8900, 0.6041, '0038p005'),
        (8.1934, 4.0124, '0081p040'),
        (6.8125, 0.5463, '0068p005'),
        ]):

        #if isrc < 7:
        #    continue
        if isrc not in [4,6,7,8,9]:
            continue
        
        outdir = 'out_%.4f_%.4f' % (ra,dec)
        datadir = outdir.replace('out_', 'data_')
        #cmd = 'ssh cori "cd legacypipe2/py && python legacypipe/runbrick.py --radec %.4f %.4f --width 100 --height 100 --survey-dir fakedr9 --outdir %s --stage image_coadds --skip-calibs && python legacyanalysis/create_testcase.py --survey-dir fakedr9 %s/coadd/*/*/*-ccds.fits %s %s"' % (ra, dec, outdir, outdir, datadir, brickname)
        #cmd = 'ssh cori "cd legacypipe2/py && python legacypipe/runbrick.py --radec %.4f %.4f --width 100 --height 100 --survey-dir fakedr9 --outdir %s --stage image_coadds --skip-calibs && python legacyanalysis/create_testcase.py --survey-dir fakedr9 --outlier-dir %s %s/coadd/*/*/*-ccds.fits %s %s"' % (ra, dec, outdir, outdir, outdir, datadir, brickname)

        outbrick = ('custom-%06i%s%05i' %
                    (int(1000*ra), 'm' if dec < 0 else 'p',
                     int(1000*np.abs(dec))))

        cmd = 'ssh cori "cd legacypipe2/py && python legacyanalysis/create_testcase.py --survey-dir fakedr9 --outlier-dir %s --outlier-brick %s %s/coadd/*/*/*-ccds.fits %s %s"' % (outdir, outbrick, outdir, datadir, brickname)
        #os.system(cmd)
    
        cmd = 'rsync -arv cori:legacypipe2/py/%s .' % datadir
        #os.system(cmd)
        #continue

        survey = LegacySurveyData(datadir)
    
        b = Duck()
        b.ra = ra
        b.dec = dec
    
        W,H = 80,80
        wcs = wcs_for_brick(b, W=W, H=H)
        targetrd = np.array([wcs.pixelxy2radec(x,y) for x,y in
                             [(1,1),(W,1),(W,H),(1,H),(1,1)]])
        ccds = survey.ccds_touching_wcs(wcs)
        print(len(ccds), 'CCDs')
    
        ims = [survey.get_image_object(ccd) for ccd in ccds]
        keepims = []
        for im in ims:
            h,w = im.shape
            if h >= H and w >= W:
                keepims.append(im)
        ims = keepims
        gims = [im for im in ims if im.band == 'g']
        rims = [im for im in ims if im.band == 'r']
        zims = [im for im in ims if im.band == 'z']
        nk = min([len(gims), len(rims), len(zims), 5])
        ims = gims[:nk] + rims[:nk] + zims[:nk]
        print('Keeping', len(ims), 'images')
        
        tims = [im.get_tractor_image(pixPsf=True, hybridPsf=True, normalizePsf=True, splinesky=True, radecpoly=targetrd) for im in ims]
    
        bands = 'grz'
    
        devsrc = DevGalaxy(RaDecPos(ra,dec), NanoMaggies(**dict([(b,10.) for b in bands])), EllipseESoft(0., 0., 0.))
        tr = Tractor(tims, [devsrc])
        tr.freezeParam('images')
        tr.optimize_loop()
        print('Fit DeV source:', devsrc)
        devmods = list(tr.getModelImages())
        showboth(tims, devmods, wcs, bands);
        s = flux_string(devsrc.brightness)
        plt.suptitle('DeV model: ' + s + '\ndchisq 0.')
        plt.savefig('src%02i-dev.png' % isrc)
        devchi = 2. * tr.getLogLikelihood()
        
        dasrc = DevAgnGalaxy(devsrc.pos.copy(), devsrc.brightness.copy(), devsrc.shape.copy(), NanoMaggies(**dict([(b,1.) for b in bands])))
        #dasrc = DevAgnGalaxy(RaDecPos(ra,dec), NanoMaggies(**dict([(b,10.) for b in bands])), EllipseESoft(0., 0., 0.), NanoMaggies(**dict([(b,1.) for b in bands])))
        tr = Tractor(tims, [dasrc])
        tr.freezeParam('images')
        tr.optimize_loop()
        print('Fit Dev+PSF source:', dasrc)
        damods = list(tr.getModelImages())
        showboth(tims, damods, wcs, bands)
        s1 = flux_string(dasrc.brightnessDev)
        s2 = flux_string(dasrc.brightnessPsf)
        pcts = [100. * dasrc.brightnessPsf.getFlux(b) / dasrc.brightnessDev.getFlux(b) for b in bands]
        s3 = ', '.join(['%.2f' % p for p in pcts])
        dachi = 2. * tr.getLogLikelihood()
        plt.suptitle('DeV + Point Source model: DeV %s, PSF %s' % (s1, s2) + ' (%s %%)' % s3 +
                     '\ndchisq %.1f' % (dachi - devchi))
        plt.savefig('src%02i-devcore.png' % isrc)
    
        #sersrc = SersicGalaxy(RaDecPos(ra, dec), NanoMaggies(**dict([(b,10.) for b in bands])), EllipseESoft(0.5, 0., 0.), SersicIndex(4.0))
        sersrc = SersicGalaxy(devsrc.pos.copy(), devsrc.brightness.copy(), devsrc.shape.copy(), SersicIndex(4.0))
        tr = Tractor(tims, [sersrc])
        tr.freezeParam('images')
        r = tr.optimize_loop()
        print('Opt:', r)
        print('Fit Ser source:', sersrc)
        if sersrc.sersicindex.getValue() >= 6.0:
            sersrc.freezeParam('sersicindex')
            r = tr.optimize_loop()
            print('Re-fit Ser source:', sersrc)
        sermods = list(tr.getModelImages())
        showboth(tims, sermods, wcs, bands)
        s = flux_string(sersrc.brightness)
        serchi = 2. * tr.getLogLikelihood()
        plt.suptitle('Sersic model: %s, index %.2f' % (s, sersrc.sersicindex.getValue()) +
                    '\ndchisq %.1f' % (serchi - devchi))
        plt.savefig('src%02i-ser.png' % isrc)
    
        #sasrc = SersicAgnGalaxy(RaDecPos(ra, dec), NanoMaggies(**dict([(b,10.) for b in bands])), EllipseESoft(0.5, 0., 0.), SersicIndex(4.0), NanoMaggies(**dict([(b,1.) for b in bands])))

        si = sersrc.sersicindex.getValue()
        if si > 6.0:
            si = 4.0
        si = SersicIndex(si)
        
        sasrc = SersicAgnGalaxy(sersrc.pos.copy(), sersrc.brightness.copy(), sersrc.shape.copy(), si, NanoMaggies(**dict([(b,1.) for b in bands])))
        tr = Tractor(tims, [sasrc])
        tr.freezeParam('images')
        r = tr.optimize_loop()
        print('Fit Ser+PSF source:', sasrc)
        if sasrc.sersicindex.getValue() >= 6.0:
            sasrc.freezeParam('sersicindex')
            r = tr.optimize_loop()
            print('Re-fit Ser+PSF source:', sasrc)
        samods = list(tr.getModelImages())
        showboth(tims, samods, wcs, bands)
        s1 = flux_string(sasrc.brightness)
        s2 = flux_string(sasrc.brightnessPsf)
        pcts = [100. * sasrc.brightnessPsf.getFlux(b) / sasrc.brightness.getFlux(b) for b in bands]
        s3 = ', '.join(['%.2f' % p for p in pcts])
        sachi = 2. * tr.getLogLikelihood()
        plt.suptitle('Sersic + Point Source model: Ser %s, index %.2f, PSF %s' % (s1, sasrc.sersicindex.getValue(), s2) +
                     ' (%s %%)' % s3 +
                    '\ndchisq %.1f' % (sachi - devchi))
        plt.savefig('src%02i-sercore.png' % isrc)

        ri = (ra, dec, brickname, devsrc, devchi, dasrc, dachi,
              sersrc, serchi, sasrc, sachi)
        if len(results) > isrc:
            results[isrc] = ri
        else:
            results.append(ri)

    pickle_to_file(results, 'results.pickle')


if __name__ == '__main__':
    main()
