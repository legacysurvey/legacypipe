import os
import numpy as np
from legacypipe.utils import find_unique_pixels

import logging
logger = logging.getLogger('legacypipe.depthcut')
def info(*args):
    from legacypipe.utils import log_info
    log_info(logger, args)
def debug(*args):
    from legacypipe.utils import log_debug
    log_debug(logger, args)


def make_depth_cut(survey, ccds, bands, targetrd, brick, W, H, pixscale,
                   plots, ps, splinesky, gaussPsf, pixPsf, normalizePsf, do_calibs,
                   gitver, targetwcs, old_calibs_ok, get_depth_maps=False, margin=0.5,
                   use_approx_wcs=False, decals_first=False,
                   max_gb_per_band=None,
                   keep_propids=None,
                   first_propids=None):
    '''
    keep_propids: iterable of PROPID strings to definitely keep
    first_propids: iterable of PROPID strings to try first
    '''
    if plots:
        import pylab as plt

    # For pixel-to-gb accounting: roughly, 4 bytes for image, 4 for ivar, 2 for DQ
    pix_to_gb = 10 / 1e9

    # Add some margin to our DESI depth requirements
    target_depth_map = dict(g=24.0 + margin, r=23.4 + margin, z=22.5 + margin,
                            # And make up some requirements for other bands!
                            i=23.0 + margin, y=22.0 + margin)

    # List extra (redundant) target percentiles so that increasing the depth at
    # any of these percentiles causes the image to be kept.
    target_percentiles = np.array(list(range(2, 10)) +
                                  list(range(10, 30, 5)) +
                                  list(range(30, 101, 10)))
    target_ddepths = np.zeros(len(target_percentiles), np.float32)
    target_ddepths[target_percentiles < 10] = -0.3
    target_ddepths[target_percentiles <  5] = -0.6
    #print('Target percentiles:', target_percentiles)
    #print('Target ddepths:', target_ddepths)

    target_nexp = 3
    target_nexp_pct = 95

    if keep_propids is None:
        keep_propids = []
    else:
        keep_propids = list(keep_propids)

    if first_propids is None:
        first_propids = []
    else:
        first_propids = list(first_propids)

    # this is old timey
    if decals_first:
        # Try DECaLS data first!
        first_propids.append('2014B-0404')

    # as an implementation detail, keep_propids will get added to first_propids
    first_propids = list(set(first_propids).union(keep_propids))
    
    cH,cW = H//10, W//10
    coarsewcs = targetwcs.scale(0.1)
    coarsewcs.imagew = cW
    coarsewcs.imageh = cH

    # Unique pixels in this brick (U: cH x cW boolean)
    U = find_unique_pixels(coarsewcs, cW, cH, None,
                           brick.ra1, brick.ra2, brick.dec1, brick.dec2)
    pixscale = 3600. * np.sqrt(np.abs(ccds.cd1_1*ccds.cd2_2 - ccds.cd1_2*ccds.cd2_1))
    seeing = ccds.fwhm * pixscale

    target_nexp_npix = int(np.sum(U) * float(target_nexp_pct) / 100.)

    # Compute the rectangle in *coarsewcs* covered by each CCD
    slices = []
    overlapping_ccds = np.zeros(len(ccds), bool)
    for i,ccd in enumerate(ccds):
        wcs = survey.get_approx_wcs(ccd)
        hh,ww = wcs.shape
        rr,dd = wcs.pixelxy2radec([1,ww,ww,1], [1,1,hh,hh])
        _,xx,yy = coarsewcs.radec2pixelxy(rr, dd)
        y0 = int(np.round(np.clip(yy.min(), 0, cH-1)))
        y1 = int(np.round(np.clip(yy.max(), 0, cH-1)))
        x0 = int(np.round(np.clip(xx.min(), 0, cW-1)))
        x1 = int(np.round(np.clip(xx.max(), 0, cW-1)))
        if y0 == y1 or x0 == x1:
            slices.append(None)
            continue
        # Check whether this CCD overlaps the unique area of this brick...
        if not np.any(U[y0:y1+1, x0:x1+1]):
            info('No overlap with unique area for CCD', ccd.expnum, ccd.ccdname)
            slices.append(None)
            continue
        overlapping_ccds[i] = True
        slices.append((slice(y0, y1+1), slice(x0, x1+1)))

    overlapping_pixels = np.zeros(len(ccds), np.int32)
    keep_ccds = np.zeros(len(ccds), bool)
    depthmaps = []

    npix_band = {}
    # status strings, by band
    depth_status = {}
    nexp_status = {}

    for band in bands:
        # scalar
        target_depth = target_depth_map[band]
        # vector
        target_depths = target_depth + target_ddepths

        nexp = np.zeros((cH,cW), np.int16)
        last_nexp = np.zeros_like(nexp)

        depthiv = np.zeros((cH,cW), np.float32)
        depthmap = np.zeros_like(depthiv)
        depthvalue = np.zeros_like(depthiv)
        last_pcts = np.zeros_like(target_depths)
        # indices of CCDs we still want to look at in the current band
        b_inds = np.where(ccds.filter == band)[0]
        info(len(b_inds), 'CCDs in', band, 'band')
        if len(b_inds) == 0:
            continue
        b_inds = np.array([i for i in b_inds if slices[i] is not None])
        info(len(b_inds), 'CCDs in', band, 'band overlap target')
        if len(b_inds) == 0:
            continue
        # CCDs that we will try before searching for good ones -- CCDs
        # from the same exposure number as CCDs we have chosen to
        # take.
        try_ccds = set()

        if len(first_propids):
            match = np.isin(ccds.propid[b_inds], first_propids)
            if np.any(match):
                info('Trying', np.sum(match), 'CCDs first based on PROPID')
                try_ccds.update(b_inds[match])

        plot_vals = []

        if plots:
            plt.clf()
            for i in b_inds:
                sy,sx = slices[i]
                x0,x1 = sx.start, sx.stop
                y0,y1 = sy.start, sy.stop
                plt.plot([x0,x0,x1,x1,x0], [y0,y1,y1,y0,y0], 'b-', alpha=0.5)
            plt.title('CCDs overlapping brick: %i in %s band' % (len(b_inds), band))
            ps.savefig()

            nccds = np.zeros((cH,cW), np.int16)
            plt.clf()
            for i in b_inds:
                nccds[slices[i]] += 1
            plt.imshow(nccds, interpolation='nearest', origin='lower', vmin=0)
            plt.colorbar()
            plt.title('CCDs overlapping brick: %i in %s band (%i / %i / %i)' %
                      (len(b_inds), band, nccds.min(), np.median(nccds), nccds.max()))

            ps.savefig()
            #continue

        # b_inds: indices of CCDs in this band, still to check
        while len(b_inds):
            if len(try_ccds) == 0:
                # Choose the next CCD to look at in this band.

                # A rough point-source depth proxy would be:
                # metric = np.sqrt(ccds.extime[b_inds]) / seeing[b_inds]
                # If we want to put more weight on choosing good-seeing images, we could do:
                #metric = np.sqrt(ccds.exptime[b_inds]) / seeing[b_inds]**2

                # depth would be ~ 1 / (sig1 * seeing); we privilege good seeing here.
                metric = 1. / (ccds.sig1[b_inds] * seeing[b_inds]**2)

                # This metric is *BIG* for *GOOD* ccds!

                # Here, we try explicitly to include CCDs that cover
                # pixels that are still shallow by the largest amount
                # for the largest number of percentiles of interest;
                # note that pixels with no coverage get depth 0, so
                # score high in this metric.
                #
                # The value is the depth still required to hit the
                # target, summed over percentiles of interest
                # (for pixels unique to this brick)
                depthvalue[:,:] = 0.
                active = (last_pcts < target_depths)
                for d in target_depths[active]:
                    depthvalue += U * np.maximum(0, d - depthmap)
                ccdvalue = np.zeros(len(b_inds), np.float32)
                for j,i in enumerate(b_inds):
                    #ccdvalue[j] = np.sum(depthvalue[slices[i]])
                    # mean -- we want the most bang for the buck per pixel?
                    ccdvalue[j] = np.mean(depthvalue[slices[i]])
                metric *= ccdvalue

                # *ibest* is an index into b_inds
                ibest = np.argmax(metric)
                # *iccd* is an index into ccds.
                iccd = b_inds[ibest]
                ccd = ccds[iccd]
                debug('Chose best CCD: seeing', seeing[iccd], 'exptime', ccds.exptime[iccd], 'with value', ccdvalue[ibest])

            else:
                iccd = try_ccds.pop()
                ccd = ccds[iccd]
                debug('Popping CCD from try_ccds list')

            # remove *iccd* from b_inds
            b_inds = b_inds[b_inds != iccd]

            im = survey.get_image_object(ccd)
            debug('Band', im.band, 'expnum', im.expnum, 'exptime', im.exptime, 'seeing', im.fwhm*im.pixscale, 'arcsec, propid', im.propid)

            im.check_for_cached_files(survey)
            debug(im)

            if do_calibs:
                kwa = dict(git_version=gitver, old_calibs_ok=old_calibs_ok)
                if gaussPsf:
                    kwa.update(psfex=False)
                if splinesky:
                    kwa.update(splinesky=True)
                im.run_calibs(**kwa)

            if use_approx_wcs:
                debug('Using approximate (TAN) WCS')
                wcs = survey.get_approx_wcs(ccd)
            else:
                debug('Reading WCS from', im.imgfn, 'HDU', im.hdu)
                wcs = im.get_wcs()

            x0,x1,y0,y1,_ = im.get_image_extent(wcs=wcs, radecpoly=targetrd)
            if x0==x1 or y0==y1:
                debug('No actual overlap')
                continue
            wcs = wcs.get_subimage(int(x0), int(y0), int(x1-x0), int(y1-y0))
            npix_ccd = (y1-y0)*(x1-x0)

            if 'galnorm_mean' in ccds.get_columns():
                galnorm = ccd.galnorm_mean
                debug('Using galnorm_mean from CCDs table:', galnorm)
            else:
                psf = im.read_psf_model(x0, y0, gaussPsf=gaussPsf, pixPsf=pixPsf,
                                        normalizePsf=normalizePsf)
                psf = psf.constantPsfAt((x1-x0)//2, (y1-y0)//2)
                # create a fake tim to compute galnorm
                from tractor import PixPos, Flux, ModelMask, Image, NullWCS
                from legacypipe.survey import SimpleGalaxy

                h,w = 50,50
                gal = SimpleGalaxy(PixPos(w//2,h//2), Flux(1.))
                tim = Image(data=np.zeros((h,w), np.float32),
                            psf=psf, wcs=NullWCS(pixscale=im.pixscale))
                mm = ModelMask(0, 0, w, h)
                galmod = gal.getModelPatch(tim, modelMask=mm).patch
                galmod = np.maximum(0, galmod)
                galmod /= galmod.sum()
                galnorm = np.sqrt(np.sum(galmod**2))
            detiv = 1. / (im.sig1 / galnorm)**2
            galdepth = -2.5 * (np.log10(5. * im.sig1 / galnorm) - 9.)
            debug('Galnorm:', galnorm, 'sig1:', im.sig1, 'galdepth', galdepth)

            # Add this image the the depth map...
            from astrometry.util.resample import resample_with_wcs, OverlapError
            try:
                Yo,Xo,_,_,_ = resample_with_wcs(coarsewcs, wcs)
                debug(len(Yo), 'of', (cW*cH), 'pixels covered by this image')
            except OverlapError:
                debug('No overlap')
                continue
            depthiv[Yo,Xo] += detiv

            nexp[Yo,Xo] += 1

            # compute the new depth map & percentiles (including the proposed new CCD)
            depthmap[:,:] = 0.
            depthmap[depthiv > 0] = 22.5 - 2.5*np.log10(5./np.sqrt(depthiv[depthiv > 0]))
            depthpcts = np.percentile(depthmap[U], target_percentiles)

            pmap = dict([(p,d) for p,d in zip(target_percentiles, depthpcts)])
            tmap = dict([(p,t) for p,t in zip(target_percentiles, target_depths)])
            s_depth = ('Depths in %s band: 2nd pct: %.2f, 5th pct: %.2f, 10th pct: %.2f, 50th pct: %.2f, max: %.2f mag' %
                 (band, pmap[2], pmap[5], pmap[10], pmap[50], pmap[100]))
            info(s_depth)
            info('      vs targets:          %.2f           %.2f            %.2f' %
                 (tmap[2], tmap[5], tmap[10]))

            for i,(p,d,t) in enumerate(zip(target_percentiles, depthpcts, target_depths)):
                debug('  pct % 3i, prev %5.2f -> %5.2f vs target %5.2f %s' % (p, last_pcts[i], d, t, ('ok' if d >= t else '')))

            debug('%i of %i required N_exp coarse pixels satisfied' % (np.sum(nexp[U] >= target_nexp), target_nexp_npix))
            from collections import Counter
            cn = Counter(nexp[U])

            n0 = cn.get(0,0)
            n1 = cn.get(1,0)
            n2 = cn.get(2,0)
            nT = np.sum(U)
            s_nexp = ('Percent of image (%s band) with 0 exposures: %.1f %%, <2 exp: %.1f %%, <3 exp: %.1f %%' %
                 (band, 100.*n0/nT, 100.*(n0+n1)/nT, 100.*(n0+n1+n2)/nT))
            info(s_nexp)

            debug('Nexp histogram:')
            for i in range(20):
                n = cn.get(i, 0)
                debug(' % 3i: % 9i' % (i, n))

            keep = False
            # Did we increase the depth of any target percentile that did not already exceed its target depth?
            if np.any((depthpcts > last_pcts) * (last_pcts < target_depths)):
                info('Keeping CCD to satisfy depth')
                keep = True

            if not keep:
                if np.any((nexp[U] > last_nexp[U]) * (last_nexp[U] < target_nexp)):
                    info('Keeping CCD to satisfy N_exp')
                    keep = True

            # Is it in the definitely-keep list?
            #if not keep and ccd.propid in keep_propids:
            if ccd.propid in keep_propids:
                info('Keeping CCD due to PROPID')
                keep = True
            else:
                info('  CCD propid:', ccd.propid, 'not in keep-list')#, keep_propids)

            if keep:
                depth_status[band] = s_depth
                nexp_status[band] = s_nexp

            # Add any other CCDs from this same expnum to the try_ccds list.
            # (before making the plot)
            I = np.where(ccd.expnum == ccds.expnum[b_inds])[0]
            try_ccds.update(b_inds[I])
            debug('Adding', len(I), 'CCDs with the same expnum to try_ccds list')

            if plots:
                cc = '1' if keep else '0'
                xx = [Xo.min(), Xo.min(), Xo.max(), Xo.max(), Xo.min()]
                yy = [Yo.min(), Yo.max(), Yo.max(), Yo.min(), Yo.min()]
                plot_vals.append(((xx,yy,cc),(last_pcts,depthpcts,keep),im.ccdname))

            if plots and (
                (len(try_ccds) == 0) or np.all(depthpcts >= target_depths)):
                plt.clf()

                plt.subplot2grid((2,2),(0,0))
                plt.imshow(depthvalue, interpolation='nearest', origin='lower',
                           vmin=0)
                plt.xticks([]); plt.yticks([])
                plt.colorbar()
                plt.title('heuristic value')

                plt.subplot2grid((2,2),(0,1))
                plt.imshow(depthmap, interpolation='nearest', origin='lower',
                           vmin=target_depth - 2, vmax=target_depth + 0.5)
                ax = plt.axis()
                for (xx,yy,cc) in [p[0] for p in plot_vals]:
                    plt.plot(xx,yy, '-', color=cc, lw=3)
                plt.axis(ax)
                plt.xticks([]); plt.yticks([])
                plt.colorbar()
                plt.title('depth map')

                plt.subplot2grid((2,2),(1,0), colspan=2)
                ax = plt.gca()
                plt.plot(target_percentiles, target_depths, 'ro', label='Target')
                plt.plot(target_percentiles, target_depths, 'r-')
                for (lp,dp,k) in [p[1] for p in plot_vals]:
                    plt.plot(target_percentiles, lp, 'k-',
                             label='Previous percentiles')
                for (lp,dp,k) in [p[1] for p in plot_vals]:
                    cc = 'b' if k else 'r'
                    plt.plot(target_percentiles, dp, '-', color=cc,
                             label='Depth percentiles')
                ccdnames = ','.join([p[2] for p in plot_vals])
                plot_vals = []

                plt.ylim(target_depth - 2, target_depth + 0.5)
                plt.xscale('log')
                plt.xlabel('Percentile')
                plt.ylabel('Depth')
                plt.title('depth percentiles')
                plt.suptitle('%s %i-%s, exptime %.0f, seeing %.2f, band %s' %
                             (im.camera, im.expnum, ccdnames, im.exptime,
                              im.pixscale * im.fwhm, band))
                ps.savefig()

            overlapping_pixels[iccd] = npix_ccd
            if keep:
                info('Keeping this CCD')
                if max_gb_per_band is not None:
                    if (npix_band.get(band, 0) + npix_ccd) * pix_to_gb > max_gb_per_band:
                        info('This CCD would exceed max_gb_per_band.')
                        break
                npix_band[band] = npix_band.get(band, 0) + npix_ccd
            else:
                info('Not keeping this CCD')
                depthiv[Yo,Xo] -= detiv
                nexp[Yo,Xo] -= 1
                continue

            keep_ccds[iccd] = True
            last_pcts = depthpcts
            last_nexp[:,:] = nexp

            if np.all(depthpcts >= target_depths):
                info('Reached all target depth percentiles for band', band)
                if np.sum(nexp[U] >= target_nexp) >= target_nexp_npix:
                    info('Reached all target n_exp for band', band)
                    break
                else:
                    info('Have not reached all target n_exp for band', band)

        if get_depth_maps:
            if np.any(depthiv > 0):
                depthmap[:,:] = 0.
                depthmap[depthiv > 0] = 22.5 -2.5*np.log10(5./np.sqrt(depthiv[depthiv > 0]))
                depthmap[np.logical_not(U)] = np.nan
                depthmaps.append((band, depthmap.copy()))

        if plots:
            I = np.where(ccds.filter == band)[0]
            plt.clf()
            plt.plot(seeing[I], ccds.exptime[I], 'k.')
            # which CCDs from this band are we keeping?
            kept, = np.nonzero(keep_ccds)
            if len(kept):
                kept = kept[ccds.filter[kept] == band]
                plt.plot(seeing[kept], ccds.exptime[kept], 'ro')
            plt.xlabel('Seeing (arcsec)')
            plt.ylabel('Exptime (sec)')
            plt.title('CCDs kept for band %s' % band)
            plt.ylim(0, np.max(ccds.exptime[I]) * 1.1)
            ps.savefig()

    for band in bands:
        info('Keeping', np.sum(ccds.filter[keep_ccds] == band), 'of', np.sum(ccds.filter == band), 'CCDs in band', band, 'estimated mem: %.1f GB' % (npix_band.get(band, 0) * pix_to_gb))
        info('  ', depth_status.get(band,''))
        info('  ', nexp_status.get(band,''))
    if get_depth_maps:
        return (keep_ccds, overlapping_ccds, overlapping_pixels, depthmaps)
    return keep_ccds, overlapping_ccds, overlapping_pixels






def run_one_brick(X):
    from legacypipe.survey import get_git_version, wcs_for_brick
    from astrometry.util.file import trymakedirs
    survey, brick, plots, kwargs = X
    outdir = kwargs.pop('outdir')
    dirnm = os.path.join(outdir, brick.brickname[:3])
    outfn = os.path.join(dirnm, 'ccds-%s.fits' % brick.brickname)
    if os.path.exists(outfn):
        print('Exists:', outfn)
        return 0

    H,W = 3600,3600
    pixscale = 0.262
    bands = ['g','r','i','z']

    # Get WCS object describing brick
    targetwcs = wcs_for_brick(brick, W=W, H=H, pixscale=pixscale)
    targetrd = np.array([targetwcs.pixelxy2radec(x,y) for x,y in
                         [(1,1),(W,1),(W,H),(1,H),(1,1)]])
    gitver = get_git_version()

    ccds = survey.ccds_touching_wcs(targetwcs)
    if ccds is None:
        print('No CCDs actually touching brick')
        return 0
    print(len(ccds), 'CCDs actually touching brick')
    ccds.cut(np.in1d(ccds.filter, bands))
    print('Cut on filter:', len(ccds), 'CCDs remain.')
    if 'ccd_cuts' in ccds.get_columns():
        norig = len(ccds)
        ccds.cut(ccds.ccd_cuts == 0)
        print(len(ccds), 'of', norig, 'CCDs pass cuts')
    else:
        print('No CCD cuts')
    if len(ccds) == 0:
        print('No CCDs left')
        return 0

    # DEBUG
    #ccds.writeto('all-ccds-%s.fits' % brick.brickname)

    ps = None
    if plots:
        from astrometry.util.plotutils import PlotSequence
        ps = PlotSequence('depth-%s' % brick.brickname)

    splinesky = True
    gaussPsf = False
    pixPsf = True
    do_calibs = False
    normalizePsf = True
    old_calibs_ok = False
    get_depth_maps = kwargs.pop('get_depth_maps', False)
    get_depth_percentiles = kwargs.pop('get_depth_percentiles', False)
    req_depth_maps = get_depth_maps or get_depth_percentiles
    try:
        D = make_depth_cut(
            survey, ccds, bands, targetrd, brick, W, H, pixscale,
            plots, ps, splinesky, gaussPsf, pixPsf, normalizePsf, do_calibs,
            gitver, targetwcs, old_calibs_ok, get_depth_maps=req_depth_maps, **kwargs)
        if req_depth_maps:
            keep,overlapping,n_pix,depthmaps = D
        else:
            keep,overlapping,n_pix = D
    except:
        print('Failed to make_depth_cut():')
        import traceback
        traceback.print_exc()
        return -1

    print(np.sum(overlapping), 'CCDs overlap the brick')
    print(np.sum(keep), 'CCDs passed depth cut')
    ccds.overlapping = overlapping
    ccds.passed_depth_cut = keep
    ccds.pixels_overlapping = n_pix

    trymakedirs(dirnm)
    if get_depth_maps:
        for band,depthmap in depthmaps:
            doutfn = os.path.join(dirnm, 'depth-%s-%s.fits' % (brick.brickname, band))
            hdr = fitsio.FITSHDR()
            # Plug the WCS header cards into these images
            targetwcs.add_to_header(hdr)
            hdr.delete('IMAGEW')
            hdr.delete('IMAGEH')
            hdr.add_record(dict(name='EQUINOX', value=2000.))
            hdr.add_record(dict(name='FILTER', value=band))
            fitsio.write(doutfn, depthmap, header=hdr)
            print('Wrote', doutfn)
    if get_depth_percentiles:
        from astrometry.util.fits import fits_table
        D = fits_table()
        pcts = np.arange(101)
        D.percentile = pcts[np.newaxis,:]
        for band,depthmap in depthmaps:
            # outside the unique brick area, the depthmaps are set to NaN -- so cut to unique
            p = np.percentile(depthmap[np.isfinite(depthmap)], pcts)
            print('Depth percentiles for band', band, ':')
            print(p)
            D.set('depth_' + band, p[np.newaxis,:])
            D.set('npix_all_' + band, np.array([
                np.sum(ccds.pixels_overlapping[ccds.filter == band])]))
            D.set('npix_keep_' + band, np.array([
                np.sum(ccds.pixels_overlapping[(ccds.filter == band) * ccds.passed_depth_cut])]))
        doutfn = os.path.join(dirnm, 'depths-%s.fits' % brick.brickname)
        D.writeto(doutfn)
        print('Wrote', doutfn)

    tmpfn = os.path.join(os.path.dirname(outfn), 'tmp-' + os.path.basename(outfn))
    ccds.writeto(tmpfn)
    os.rename(tmpfn, outfn)
    print('Wrote', outfn)
    return 0

def main():
    from legacypipe.survey import LegacySurveyData

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--margin', type=float, default=None,
                        help='Set margin, in mags, above the DESI depth requirements.')
    parser.add_argument('--depth-maps', action='store_true', default=False,
                        help='Write sub-scale depth map images?')
    parser.add_argument('--depth-percentiles', action='store_true', default=False,
                        help='Write summary depth map tables?')
    parser.add_argument('--plots', action='store_true', default=False)
    parser.add_argument('--outdir', default='depthcuts', help='Output directory')
    parser.add_argument('--max-gb-per-band', default=None, type=float,
                        help='Do not keep more CCDs than would take this amount of memory per band')
    parser.add_argument('--threads', type=int, help='"qdo" mode: number of threads')
    parser.add_argument('-v', '--verbose', dest='verbose', action='count',
                        default=0, help='Make more verbose')
    parser.add_argument('--dr10-propids', default=False, action='store_true',
                        help='Use DR10 keep-list of PROPIDs')
    parser.add_argument('bricks', nargs='*')
    args = parser.parse_args()
    plots = args.plots
    bricks = args.bricks

    if args.verbose:
        lvl = logging.DEBUG
    else:
        lvl = logging.INFO
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)
    # silence "findfont: score(<Font 'DejaVu Sans Mono' ...)" messages
    logging.getLogger('matplotlib.font_manager').disabled = True
    # route warnings through the logging system
    logging.captureWarnings(True)

    kwargs = dict(get_depth_maps=args.depth_maps,
                  get_depth_percentiles=args.depth_percentiles,
                  outdir=args.outdir)
    if args.margin is not None:
        kwargs.update(margin=args.margin)
    if args.max_gb_per_band is not None:
        kwargs.update(max_gb_per_band=args.max_gb_per_band)

    if args.dr10_propids:
        # From Eddie; https://github.com/legacysurvey/legacypipe/blob/main/py/legacyzpts/update_ccd_cuts.py#L27-L37
        kwargs.update(keep_propids = [
            '2013A-0741', '2013B-0440',
            '2014A-0035', '2014A-0412', '2014A-0624', '2016A-0618',
            '2015A-0397', '2015B-0314',
            '2016A-0366', '2016B-0301', '2016B-0905', '2016B-0909',
            '2017A-0388', '2017A-0916', '2017B-0110', '2017B-0906',
            '2018A-0242', '2018A-0273', '2018A-0913', '2018A-0914',
            '2018A-0386', '2019A-0272', '2019A-0305', '2019A-0910',
            '2019B-0323', '2020A-0399', '2020A-0909', '2020B-0241',
            '2019B-0371', '2019B-1014', '2020A-0908',
            '2021A-0149', '2021A-0922',
            '2022A-597406'
        ])
    allgood = 0
    bargs = []
    survey = LegacySurveyData()
    for brickname in bricks:
        #print('Checking for existing out file')
        # shortcut
        dirnm = os.path.join(args.outdir, brickname[:3])
        outfn = os.path.join(dirnm, 'ccds-%s.fits' % brickname)
        if os.path.exists(outfn):
            print('Exists:', outfn)
            continue
        #print('Getting brick', brickname)
        brick = survey.get_brick_by_name(brickname)
        bargs.append((survey, brick, plots, kwargs))

    if args.threads is not None:
        mp = multiproc(args.threads)
        rtns = mp.map(run_one_brick, bargs)
        for rtn in rtns:
            if rtn != 0:
                allgood = rtn
    else:
        for arg in bargs:
            rtn = run_one_brick(arg)
            if rtn != 0:
                allgood = rtn
            #print('Done, result', rtn)
    return allgood

if __name__ == '__main__':
    import sys
    sys.exit(main())
