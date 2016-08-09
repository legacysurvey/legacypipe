from __future__ import print_function
import sys
import os
import numpy as np
import pylab as plt
import fitsio
from astrometry.util.fits import fits_table, merge_tables
from astrometry.util.plotutils import PlotSequence, plothist, loghist
from astrometry.util.ttime import Time
from legacypipe.common import LegacySurveyData, imsave_jpeg, get_rgb
from scipy.ndimage.filters import gaussian_filter

'''
A script to examine the pixel distributions vs the CP weight maps.
'''


def plot_correlations(img, ie, sig1, medians=True):
    corrs = []
    corrs_x = []
    corrs_y = []

    mads = []
    mads_x = []
    mads_y = []

    rcorrs = []
    rcorrs_x = []
    rcorrs_y = []
    dists = np.arange(1, 51)
    for dist in dists:
        offset = dist
        slice1 = (slice(0,-offset,1),slice(0,-offset,1))
        slice2 = (slice(offset,None,1),slice(offset,None,1))

        slicex = (slice1[0], slice2[1])
        slicey = (slice2[0], slice1[1])

        corr = img[slice1] * img[slice2]
        corr = corr[(ie[slice1] > 0) * (ie[slice2] > 0)]

        diff = img[slice1] - img[slice2]
        diff = diff[(ie[slice1] > 0) * (ie[slice2] > 0)]
        sig1 = 1.4826 / np.sqrt(2.) * np.median(np.abs(diff).ravel())
        mads.append(sig1)

        #print('Dist', dist, '; number of corr pixels', len(corr))
        #t0 = Time()
        if medians:
            rcorr = np.median(corr) / sig1**2
            rcorrs.append(rcorr)
        #t1 = Time()
        corr = np.mean(corr) / sig1**2
        #t2 = Time()
        #print('median:', t1-t0)
        #print('mean  :', t2-t1)
        corrs.append(corr)

        corr = img[slice1] * img[slicex]
        corr = corr[(ie[slice1] > 0) * (ie[slicex] > 0)]

        diff = img[slice1] - img[slicex]
        diff = diff[(ie[slice1] > 0) * (ie[slicex] > 0)]
        sig1 = 1.4826 / np.sqrt(2.) * np.median(np.abs(diff).ravel())
        mads_x.append(sig1)

        if medians:
            rcorr = np.median(corr) / sig1**2
            rcorrs_x.append(rcorr)
        corr = np.mean(corr) / sig1**2
        corrs_x.append(corr)

        corr = img[slice1] * img[slicey]
        corr = corr[(ie[slice1] > 0) * (ie[slicey] > 0)]

        diff = img[slice1] - img[slicey]
        diff = diff[(ie[slice1] > 0) * (ie[slicey] > 0)]
        #Nmad = len(diff)
        sig1 = 1.4826 / np.sqrt(2.) * np.median(np.abs(diff).ravel())
        mads_y.append(sig1)

        if medians:
            rcorr = np.median(corr) / sig1**2
            rcorrs_y.append(rcorr)
        corr = np.mean(corr) / sig1**2
        corrs_y.append(corr)

        #print('Dist', dist, '-> corr', corr, 'X,Y', corrs_x[-1], corrs_y[-1],
        #      'robust', rcorrs[-1], 'X,Y', rcorrs_x[-1], rcorrs_y[-1])

    pix = img[ie > 0].ravel()
    Nmad = len(pix) / 2
    P = np.random.permutation(len(pix))[:(Nmad*2)]
    diff = pix[P[:Nmad]] - pix[P[Nmad:]]
    mad_random = 1.4826 / np.sqrt(2.) * np.median(np.abs(diff))

    plt.clf()
    p1 = plt.plot(dists, corrs, 'b.-')
    p2 = plt.plot(dists, corrs_x, 'r.-')
    p3 = plt.plot(dists, corrs_y, 'g.-')
    if medians:
        p4 = plt.plot(dists, rcorrs, 'b.--')
        p5 = plt.plot(dists, rcorrs_x, 'r.--')
        p6 = plt.plot(dists, rcorrs_y, 'g.--')
    plt.xlabel('Pixel offset')
    plt.ylabel('Correlation')
    plt.axhline(0, color='k', alpha=0.3)
    plt.legend([p1[0],p2[0],p3[0]], ['Diagonal', 'X', 'Y'], loc='upper right')

    return (dists, corrs, corrs_x, corrs_y, rcorrs, rcorrs_x, rcorrs_y,
            mads, mads_x, mads_y, mad_random)

ps = PlotSequence('noise')

# fitsgetext -i /global/homes/d/dstn/cosmo/staging/decam/DECam_Raw/20160107/c4d_160107_213046_fri.fits.fz -e 0 -e 1 -o flat.fits

# img = fitsio.read('flat.fits').astype(np.float32)
# # trim the edges
# img = img[100:-100, 100:-100]
# img -= np.median(img)

# Pixel correlations in a flat image:
if False:
    
    from obsbot.measure_raw import DECamMeasurer
    from obsbot.decam import DecamNominalCalibration
    
    nom = DecamNominalCalibration()
    fn = 'flat.fits'
    ext = 1
    meas = DECamMeasurer(fn, ext, nom)
    F = fitsio.FITS(fn)
    img,hdr = meas.read_raw(F, ext)
    
    rawimg = F[ext].read()
    plo,phi = np.percentile(rawimg.ravel(), [16,84])
    plt.clf()
    plt.imshow(rawimg, interpolation='nearest', origin='lower', vmin=plo, vmax=phi)
    plt.title('Flat image (raw)')
    plt.colorbar()
    ps.savefig()
    
    plo,phi = np.percentile(img.ravel(), [16,84])
    plt.clf()
    plt.imshow(img, interpolation='nearest', origin='lower', vmin=plo, vmax=phi)
    plt.title('Flat image')
    plt.colorbar()
    ps.savefig()
    
    img,trim_x0,trim_y0 = meas.trim_edges(img)
    sky,sig1 = meas.get_sky_and_sigma(img)
    img -= sky
    meas.remove_sky_gradients(img)
    
    plt.clf()
    plt.imshow(img, interpolation='nearest', origin='lower', vmin=-2.*sig1, vmax=2.*sig1)
    plt.title('Flat image (gradient removed)')
    plt.colorbar()
    ps.savefig()
    
    plt.clf()
    plt.imshow(img, interpolation='nearest', origin='lower', vmin=-4.*sig1, vmax=4.*sig1)
    plt.title('Flat image (gradient removed)')
    plt.colorbar()
    ps.savefig()
    
    plt.clf()
    n,b,p = plt.hist(img.ravel() / sig1, range=(-5,5), bins=100, histtype='step', color='b')
    binc = (b[1:] + b[:-1])/2.
    yy = np.exp(-0.5 * binc**2)
    plt.plot(binc, yy / yy.sum() * np.sum(n), 'k-', alpha=0.5, lw=2)
    plt.xlim(-5,5)
    plt.xlabel('Pixel values / sigma')
    plt.title('Flat image (gradient removed)')
    ps.savefig()
    
    # #binned = (img[0::4, :] + img[1::4, :] + img[2::4, :] + img[3::4, :]) / 4.
    # plt.clf()
    # #plt.plot(np.median(binned, axis=1), 'b-')
    # plt.plot([np.median(img[i*10:(i+1)*10, :]) for i in range(400)], 'b.')
    # plt.ylabel('Row median')
    # ps.savefig()
    
    fitsio.write('flat-grad.fits', img, clobber=True)
    
    X = plot_correlations(img, np.ones_like(img), sig1, medians=False)
    plt.title('Flat image')
    ps.savefig()
    (dists, corrs, corrs_x, corrs_y, rcorrs, rcorrs_x, rcorrs_y,
     mads, mads_x, mads_y, mad_random) = X
    
    # plt.clf()
    # p4 = plt.plot(dists, rcorrs, 'b.--')
    # p5 = plt.plot(dists, rcorrs_x, 'r.--')
    # p6 = plt.plot(dists, rcorrs_y, 'g.--')
    # plt.xlabel('Pixel offset')
    # plt.ylabel('Correlation')
    # plt.axhline(0, color='k', alpha=0.3)
    # plt.legend([p4[0],p5[0],p6[0]], ['Diagonal (med)', 'X (med)', 'Y (med)'],
    #            loc='upper right')
    # plt.title('Flat image')
    # ps.savefig()
    
    plt.clf()
    p4 = plt.plot(dists, mads, 'b.-')
    p5 = plt.plot(dists, mads_x, 'r.-')
    p6 = plt.plot(dists, mads_y, 'g.-')
    plt.xlabel('Pixel offset')
    plt.ylabel('MAD error estimate')
    #plt.axhline(0, color='k', alpha=0.3)
    p7 = plt.axhline(mad_random, color='k', alpha=0.3)
    plt.legend([p4[0],p5[0],p6[0], p7], ['Diagonal', 'X', 'Y', 'Random'],
               loc='lower right')
    plt.title('Flat image')
    ps.savefig()
    
    sys.exit(0)




def measure_mads(img, ie, offsets, step=1):
    sig = []
    for offset in offsets:
        slice1 = (slice(0,-offset,step),slice(0,-offset,step))
        slice2 = (slice(offset,None,step),slice(offset,None,step))
        slicex = (slice1[0], slice2[1])
        slicey = (slice2[0], slice1[1])

        diff = img[slice1] - img[slice2]
        diff = diff[(ie[slice1] > 0) * (ie[slice2] > 0)]
        mad = np.median(np.abs(diff).ravel())
        bsig1 = 1.4826 * mad / np.sqrt(2.)
        sig.append(bsig1)

        diff = img[slice1] - img[slicex]
        diff = diff[(ie[slice1] > 0) * (ie[slicex] > 0)]
        mad = np.median(np.abs(diff).ravel())
        sig.append(1.4826 * mad / np.sqrt(2.))

        diff = img[slice1] - img[slicey]
        diff = diff[(ie[slice1] > 0) * (ie[slicey] > 0)]
        mad = np.median(np.abs(diff).ravel())
        sig.append(1.4826 * mad / np.sqrt(2.))

    Ndiff = len(diff)
    # Draw random pixels and see what that MAD distribution looks like
    pix = img[ie>0].ravel()
    Nrand = min(Ndiff, len(pix)/2)
    P = np.random.permutation(len(pix))[:(Nrand*2)]
    diff2 = pix[P[:Nrand]] - pix[P[Nrand:]]
    mad2 = np.median(np.abs(diff2).ravel())
    bsig2 = 1.4826 * mad2 / np.sqrt(2.)
    sig.append(bsig2)
    return sig

def plot_mads(allsigs, names, loc='upper left'):
    allsigs = np.array(allsigs)
    allsigs = np.atleast_2d(allsigs)
    #print('Allsigs shape', allsigs.shape)
    rsigs = allsigs[:,np.array([-1])]
    allsigs = allsigs[:,:-1]
    dsigs = allsigs[:,::3]
    xsigs = allsigs[:,1::3]
    ysigs = allsigs[:,2::3]
    dsigs = np.hstack((dsigs, rsigs))
    nr,nc = xsigs.shape

    plt.clf()
    p1 = plt.plot(dsigs.T, 'bo-', label='Diagonal')
    p2 = plt.plot(np.repeat(np.arange(nc)[:,np.newaxis], 2, axis=1),
                  xsigs.T, 'go-', label='X')
    p3 = plt.plot(np.repeat(np.arange(nc)[:,np.newaxis], 2, axis=1),
                  ysigs.T, 'ro-', label='Y')
    plt.xticks(np.arange(nc+1), ['%i pix'%o for o in offsets] + ['Random'])
    plt.axhline(1., color='k', alpha=0.3)
    plt.ylabel('Error estimate / sig1')
    plt.legend([p1[0],p2[0],p3[0]], ['Diagonal', 'X', 'Y'], loc=loc)

    names = np.array(names)
    yy = dsigs[:,-1]
    ii = np.argsort(yy)
    yy = np.linspace(min(yy), max(yy), len(ii))
    for i,y in zip(ii, yy):
        plt.text(nc+0.1, y, names[i], ha='left', va='center', color='b',
                 fontsize=10)
    plt.xlim(-0.5, nc+3)

    
# Simulate the effect of sources in a MAD estimate.
if True:
    np.random.seed(44)

    H,W = 4000,2000
    #Nsrcs = 4000
    #psf_sigma = 2.
    sig1 = 1.
    #nsigma = 20.

    offsets = [1,2,3,5,10,20,40]

    #### Simulate effect of pixel-to-pixel correlations

    allsigs1 = []
    names = []
    title = 'Simulated MAD'

    for corr in [ 0.01, 0.02, 0.03 ]:
        img = sig1 * np.random.normal(size=(H,W)).astype(np.float32)
        ie = np.ones_like(img) / sig1
        cimg = np.zeros_like(img)
        cimg[1:-1, 1:-1] = ((1. - 4.*corr) * img[1:-1, 1:-1] +
                            corr * img[:-2, 1:-1] +
                            corr * img[2: , 1:-1] +
                            corr * img[1:-1, :-2] +
                            corr * img[1:-1, 2: ])
        img = cimg
        names.append('Correlation %g' % corr)
        sig = measure_mads(img, ie, offsets)
        allsigs1.append(sig)
    plot_mads(allsigs1, names, loc='lower right')
    plt.title(title)
    ps.savefig()

    allsigs1 = []
    names = []
    for s in [0.1, 0.2, 0.3]:
        img = sig1 * np.random.normal(size=(H,W)).astype(np.float32)
        ie = np.ones_like(img) / sig1
        img = gaussian_filter(img, s)
        names.append('Gaussian %g' % s)
        sig = measure_mads(img, ie, offsets)
        allsigs1.append(sig)
    plot_mads(allsigs1, names, loc='center left')
    plt.title(title)
    ps.savefig()




    #### Effects of sources

    allsigs1 = []
    names = []
    title = 'Simulated MAD'

    for Nsrcs, psf_sigma in [
            (2000, 2.),
            (4000, 2.),
            (4000, 4.),
            ]:

        ph,pw = 25,25
        psz = pw/2
        xx,yy = np.meshgrid(np.arange(pw), np.arange(ph))
        psfimg = np.exp(-0.5 * ((xx - pw/2)**2 + (yy - ph/2)**2) /
                        (2.*psf_sigma**2))
        psfimg /= np.sum(psfimg)
        psfnorm = np.sqrt(np.sum(psfimg**2))
        #flux = nsigma * sig1 / psfnorm
        flux = 100.

        print('N sigma:', flux * psfnorm / sig1)
        
        img = np.zeros((H,W), np.float32)
        ie = np.ones_like(img) / sig1
        sx = np.random.randint(low=psz, high=W-psz, size=Nsrcs)
        sy = np.random.randint(low=psz, high=H-psz, size=Nsrcs)
        for x,y in zip(sx,sy):
            img[y-psz: y-psz+ph, x-psz: x-psz+pw] += psfimg * flux
        img += sig1 * np.random.normal(size=img.shape)

        #plt.clf()
        #plt.imshow(img, interpolation='nearest', origin='lower',
        #           vmin=-2.*sig1, vmax=5.*sig1)
        #ps.savefig()

        names.append('%i srcs, seeing %.2f' % (Nsrcs, psf_sigma * 2.35 * 0.262))


        ####
        sig = measure_mads(img, ie, offsets)
        allsigs1.append(sig)
    

    plot_mads(allsigs1, names)
    plt.title(title)
    ps.savefig()




    


    
    
        

    sys.exit(0)



    
survey = LegacySurveyData()
ccds = survey.get_ccds_readonly()
#ccds = ccds[np.abs(ccds.mjd_obs - 57444) < 7.]
#print(len(ccds), 'CCDs near mjd')
ccds.cut(ccds.ccdname == 'N4')
print(len(ccds), 'exposures')
print('bands:', np.unique(ccds.filter))

## HACK
np.random.seed(44)

# Alternate 'oki' and 'ooi' images...
oki = np.array(['oki' in ccd.image_filename for ccd in ccds])
I1 = np.flatnonzero(oki)
I2 = np.flatnonzero(oki == False)
print(len(I1), 'oki images')
print(len(I2), 'non-oki images')
ccds.cut(np.hstack(zip(I1[np.random.permutation(len(I1))],
                       I2[np.random.permutation(len(I2))])))
#ccds.cut(np.random.permutation(len(ccds)))







allsigs1 = []
allsigs2 = []
names = []

offsets = [1,2,3,5,10,20,40]

for ccd in ccds[:16]:
    im = survey.get_image_object(ccd)

    tim1 = im.get_tractor_image(gaussPsf=True, splinesky=True, dq=False)

    tim2 = im.get_tractor_image(gaussPsf=True, splinesky=True, dq=False,
                                 nanomaggies=False, subsky=False)
    tim = tim2

    name = tim.name
    if 'oki' in im.imgfn:
        name += ' oki'
    elif 'ooi' in im.imgfn:
        name += ' ooi'
    names.append(name)

    for tim,sigs in [(tim1,allsigs1),(tim2,allsigs2)]:
        img = tim.getImage()
        ie = tim.getInvError()

        # # Estimate per-pixel noise via Blanton's 5-pixel MAD
        #plt.clf()

        sig = []
    
        #bsigs = {}
        for offset in offsets:
            step = 1

            slice1 = (slice(0,-offset,step),slice(0,-offset,step))
            slice2 = (slice(offset,None,step),slice(offset,None,step))

            slicex = (slice1[0], slice2[1])
            slicey = (slice2[0], slice1[1])

            diff = img[slice1] - img[slice2]
            #print('Blanton number of pixels:', len(diff.ravel()))
            diff = diff[(ie[slice1] > 0) * (ie[slice2] > 0)]
            #print('Blanton number of pixels kept:', len(diff.ravel()))
            mad = np.median(np.abs(diff).ravel())
            bsig1 = 1.4826 * mad / np.sqrt(2.)
            #bsigs[offset] = bsig1
            sig.append(bsig1)

            diff = img[slice1] - img[slicex]
            diff = diff[(ie[slice1] > 0) * (ie[slicex] > 0)]
            mad = np.median(np.abs(diff).ravel())
            #bsigs[1000 + offset] = 1.4826 * mad / np.sqrt(2.)
            sig.append(1.4826 * mad / np.sqrt(2.))

            diff = img[slice1] - img[slicey]
            diff = diff[(ie[slice1] > 0) * (ie[slicey] > 0)]
            mad = np.median(np.abs(diff).ravel())
            #bsigs[2000 + offset] = 1.4826 * mad / np.sqrt(2.)
            sig.append(1.4826 * mad / np.sqrt(2.))

        Ndiff = len(diff)
        # Draw random pixels and see what that MAD distribution looks like
        pix = img[ie>0].ravel()
        Nrand = min(Ndiff, len(pix)/2)
        P = np.random.permutation(len(pix))[:(Nrand*2)]
        diff2 = pix[P[:Nrand]] - pix[P[Nrand:]]
        mad2 = np.median(np.abs(diff2).ravel())
        bsig2 = 1.4826 * mad2 / np.sqrt(2.)
        #bsigs[0] = bsig2
        sig.append(bsig2)
        #plt.hist(np.abs(diff2), range=(0, tim.sig1*5), bins=50, histtype='step',
        #         label='Blanton(Random)', normed=True)
    
        # Draw Gaussian random pixels
        print('                                           sig1: %.2f' % tim.sig1)
        diff3 = tim.sig1 * (np.random.normal(size=Ndiff) - np.random.normal(size=Ndiff))
        mad3 = np.median(np.abs(diff3).ravel())
        bsig3 = 1.4826 * mad3 / np.sqrt(2.)
        print('Blanton sig1 estimate for Gaussian pixels      : %.2f' % bsig3)
        #bsigs[-1] = bsig3
    
        # for offset in [1,3,5]:
        #     print('Blanton sig1 estimate for offset of %i pixels   : %.2f' % (offset, bsigs[offset]))
        #print('Blanton sig1 estimate for randomly drawn pixels: %.2f' % bsig2)
    
        # sigs.append([bsigs[o]/tim.sig1 for o in [-1, 1, 3, 5, 0,
        #                                           1001,1003,1005,2001,2003,2005]])
        sigs.append([s/tim.sig1 for s in sig])


allsigs = np.array(allsigs1)
for alls,title in [(allsigs1,'Error estimates vs pixel difference distributions (w/splinesky)'),
                   (allsigs2, 'Error estimates vs pixel difference distributions')]:
    allsigs = np.array(alls)
    print('Allsigs shape', allsigs.shape)

    rsigs = allsigs[:,np.array([-1])]
    allsigs = allsigs[:,:-1]

    dsigs = allsigs[:,::3]
    xsigs = allsigs[:,1::3]
    ysigs = allsigs[:,2::3]

    print('rsigs:', rsigs.shape)
    print('dsigs:', dsigs.shape)

    dsigs = np.hstack((dsigs, rsigs))

    #xsigs = allsigs[:,5:8]
    #ysigs = allsigs[:,8:]
    #allsigs = allsigs[:,:5]

    nr,nc = xsigs.shape

    plt.clf()
    p1 = plt.plot(dsigs.T, 'bo-', label='Diagonal')
    p2 = plt.plot(np.repeat(np.arange(nc)[:,np.newaxis], 2, axis=1),
                  xsigs.T, 'go-', label='X')
    p3 = plt.plot(np.repeat(np.arange(nc)[:,np.newaxis], 2, axis=1),
                  ysigs.T, 'ro-', label='Y')
    plt.xticks(np.arange(nc+1), ['%i pix'%o for o in offsets] + ['Random'])
    plt.axhline(1., color='k', alpha=0.3)
    plt.ylabel('Error estimate / pipeline sig1')
    plt.title(title)
    #plt.legend(loc='upper left')
    plt.legend([p1[0],p2[0],p3[0]], ['Diagonal', 'X', 'Y'], loc='upper left')

    names = np.array(names)
    yy = dsigs[:,-1]
    ii = np.argsort(yy)
    yy = np.linspace(min(yy), max(yy), len(ii))
    for i,y in zip(ii, yy):
        plt.text(nc+0.1, y, names[i], ha='left', va='center', color='b')
    plt.xlim(-0.5, nc+3)
    
    ps.savefig()

sys.exit(0)












for ccd in ccds[:1]:
    im = survey.get_image_object(ccd)

    #tim2 = im.get_tractor_image(gaussPsf=True, splinesky=True, dq=False,
    #                             nanomaggies=False, subsky=False)
    #tim = tim2
    tim = im.get_tractor_image(gaussPsf=True, splinesky=True, dq=False)
    img = tim.getImage()
    ie = tim.getInvError()

    plt.clf()
    plt.subplot(2,1,1)
    n,b,p1 = plt.hist(img[ie>0].ravel(), range=(-5. * tim.sig1, 5. * tim.sig1),
                     bins=100, histtype='step', color='b')
    bc = (b[:-1] + b[1:])/2.
    y = np.exp(-0.5 * bc**2 / tim.sig1**2)
    N = np.sum(n)
    y *= N / np.sum(y)
    p2 = plt.plot(bc, y, 'r--')
    plt.xlim(-5.*tim.sig1, 5.*tim.sig1)
    plt.yscale('symlog')
    plt.legend((p1[0], p2[0]), ('Image pixels', 'Gaussian sig1'), loc='lower center')

    plt.subplot(2,1,2)
    n,b,p1 = plt.hist((img * ie)[ie>0].ravel(), range=(-5., 5.), bins=100,
                      histtype='step', color='b')
    bc = (b[:-1] + b[1:])/2.
    y = np.exp(-0.5 * bc**2)
    N = np.sum(n)
    y *= N / np.sum(y)
    p2 = plt.plot(bc, y, 'r--')
    plt.xlim(-5., 5.)
    plt.yscale('symlog')
    plt.legend((p1[0], p2[0]), ('Image pixels * sqrt(Weight map)', 'Unit gaussian'),
               loc='lower center')

    plt.suptitle(tim.name)
    ps.savefig()


for ccd in ccds[:2]:
    im = survey.get_image_object(ccd)

    tim2 = im.get_tractor_image(gaussPsf=True, splinesky=True, dq=False,
                                 nanomaggies=False, subsky=False)
    tim = tim2
    #tim = im.get_tractor_image(gaussPsf=True, splinesky=True, dq=False)
    img = tim.getImage()
    ie = tim.getInvError()

    skymod = np.zeros_like(img)
    tim.getSky().addTo(skymod)
    midsky = np.median(skymod)
    print('Median spline sky model level:', midsky)
    #img -= midsky

    medsky = np.median(img[ie > 0])
    print('Median image level:', medsky)
    img -= medsky

    # Select pixels in range [-2 sigma, +2 sigma]

    ie[np.logical_or(img < -2.*tim.sig1, img > 2.*tim.sig1)] = 0.

    medsky = np.median(img[ie > 0])
    print('Median of unmasked image pixels:', medsky)

    plt.clf()
    plt.imshow(img * (ie > 0), interpolation='nearest', origin='lower',
               vmin=-2.*tim.sig1, vmax=2.*tim.sig1)
    plt.title(tim.name)
    ps.savefig()

    plot_correlations(img, ie)
    plt.title(tim.name + ' ' + '/'.join(im.imgfn.split('/')[-2:]))
    #plt.ylim(-0.005, 0.02)
    plt.ylim(-0.001, 0.011)
    ps.savefig()



sys.exit(0)






for ccd in ccds[:1]:
    im = survey.get_image_object(ccd)

    plt.clf()
    plt.subplot(3,1,1)
    n,b,p1 = plt.hist(img[ie>0].ravel(), range=(-5. * tim.sig1, 5. * tim.sig1),
                     bins=100, histtype='step', color='b')
    bc = (b[:-1] + b[1:])/2.
    y = np.exp(-0.5 * bc**2 / tim.sig1**2)
    N = np.sum(n)
    y *= N / np.sum(y)
    p2 = plt.plot(bc, y, 'r--')
    plt.xlim(-5.*tim.sig1, 5.*tim.sig1)
    plt.yscale('symlog')
    plt.legend((p1[0], p2[0]), ('Image pixels', 'Gaussian sig1'), loc='lower center')

    plt.subplot(3,1,2)
    n,b,p1 = plt.hist((img * ie)[ie>0].ravel(), range=(-5., 5.), bins=100,
                      histtype='step', color='b')
    bc = (b[:-1] + b[1:])/2.
    y = np.exp(-0.5 * bc**2)
    N = np.sum(n)
    y *= N / np.sum(y)
    p2 = plt.plot(bc, y, 'r--')
    plt.xlim(-5., 5.)
    plt.yscale('symlog')
    plt.legend((p1[0], p2[0]), ('Image pixels / sqrt(Weight map)', 'Unit gaussian'),
               loc='lower center')


    bsig1 = bsigs[5]

    plt.subplot(3,1,3)
    lo,hi = min(tim.sig1 * 0.8, bsig1*0.9), max(tim.sig1 * 1.2, bsig1*1.1)
    n,b,p1 = plt.hist((1. / ie[ie>0]).ravel(), range=(lo,hi), bins=100,
                     histtype='step', color='b')
    p2 = plt.axvline(tim.sig1, color='b')
    p3 = plt.axvline(bsig1, color='r')

    pb3 = plt.axvline(bsigs[3], color='g')
    pb1 = plt.axvline(bsigs[1], color='m')
    pb0 = plt.axvline(bsigs[0], color='k')

    plt.xlim(lo,hi)
    #plt.legend((p1[0],p2,p3), ('Weight map', 'sig1', 'Blanton sig1'))
    plt.legend((p1[0],p2,p3,pb3,pb1,pb0), ('Weight map', 'sig1', 'Blanton(5) sig1',
                                           'Blanton(3)', 'Blanton(1)', 'Blanton(random)'))

    # This shows the results without any zeropoint scaling or sky subtraction; looks the same.
    # tim2 = im.get_tractor_image(gaussPsf=True, splinesky=True, dq=False,
    #                             nanomaggies=False, subsky=False)
    # img = tim2.getImage()
    # ie = tim2.getInvError()
    # diff = img[slice1] - img[slice2]
    # diff = diff[(ie[slice1] > 0) * (ie[slice2] > 0)]
    # mad = np.median(np.abs(diff).ravel())
    # bsig2 = 1.4826 * mad / np.sqrt(2.)
    # print('Blanton sig1 estimate #2:', bsig2)
    # print('sig1 #2:', tim2.sig1)
    # plt.subplot(2,2,4)
    # lo,hi = min(tim2.sig1 * 0.8, bsig2*0.9), max(tim2.sig1 * 1.2, bsig2*1.1)
    # n,b,p1 = plt.hist((1. / ie[ie > 0]).ravel(), range=(lo,hi), bins=100,
    #                  histtype='step', color='b')
    # p2 = plt.axvline(tim2.sig1, color='b')
    # p3 = plt.axvline(bsig2, color='r')
    # plt.xlim(lo,hi)
    # plt.legend((p1[0],p2,p3), ('Weight map', 'sig1', 'Blanton sig1'))

    plt.suptitle(tim.name)
    ps.savefig()


    


