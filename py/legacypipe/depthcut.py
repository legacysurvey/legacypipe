import numpy as np
from legacypipe.utils import find_unique_pixels

def make_depth_cut(survey, ccds, bands, targetrd, brick, W, H, pixscale,
                   plots, ps, splinesky, gaussPsf, pixPsf, normalizePsf, do_calibs,
                   gitver, targetwcs, old_calibs_ok, get_depth_maps=False, margin=0.5,
                   use_approx_wcs=False):
    if plots:
        import pylab as plt

    # Add some margin to our DESI depth requirements
    target_depth_map = dict(g=24.0 + margin, r=23.4 + margin, z=22.5 + margin)

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

    cH,cW = H//10, W//10
    coarsewcs = targetwcs.scale(0.1)
    coarsewcs.imagew = cW
    coarsewcs.imageh = cH

    # Unique pixels in this brick (U: cH x cW boolean)
    U = find_unique_pixels(coarsewcs, cW, cH, None,
                           brick.ra1, brick.ra2, brick.dec1, brick.dec2)
    pixscale = 3600. * np.sqrt(np.abs(ccds.cd1_1*ccds.cd2_2 - ccds.cd1_2*ccds.cd2_1))
    seeing = ccds.fwhm * pixscale

    # Compute the rectangle in *coarsewcs* covered by each CCD
    slices = []
    overlapping_ccds = np.zeros(len(ccds), bool)
    for i,ccd in enumerate(ccds):
        wcs = survey.get_approx_wcs(ccd)
        hh,ww = wcs.shape
        rr,dd = wcs.pixelxy2radec([1,ww,ww,1], [1,1,hh,hh])
        ok,xx,yy = coarsewcs.radec2pixelxy(rr, dd)
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

    keep_ccds = np.zeros(len(ccds), bool)
    depthmaps = []

    for band in bands:
        # scalar
        target_depth = target_depth_map[band]
        # vector
        target_depths = target_depth + target_ddepths

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

        # Try DECaLS data first!
        Idecals = np.where(ccds.propid[b_inds] == '2014B-0404')[0]
        if len(Idecals):
            try_ccds.update(b_inds[Idecals])
        debug('Added', len(try_ccds), 'DECaLS CCDs to try-list')

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
                debug('Popping CCD from use_ccds list')

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

            x0,x1,y0,y1,slc = im.get_image_extent(wcs=wcs, radecpoly=targetrd)
            if x0==x1 or y0==y1:
                debug('No actual overlap')
                continue
            wcs = wcs.get_subimage(int(x0), int(y0), int(x1-x0), int(y1-y0))

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

            # compute the new depth map & percentiles (including the proposed new CCD)
            depthmap[:,:] = 0.
            depthmap[depthiv > 0] = 22.5 - 2.5*np.log10(5./np.sqrt(depthiv[depthiv > 0]))
            depthpcts = np.percentile(depthmap[U], target_percentiles)

            for i,(p,d,t) in enumerate(zip(target_percentiles, depthpcts, target_depths)):
                info('  pct % 3i, prev %5.2f -> %5.2f vs target %5.2f %s' % (p, last_pcts[i], d, t, ('ok' if d >= t else '')))

            keep = False
            # Did we increase the depth of any target percentile that did not already exceed its target depth?
            if np.any((depthpcts > last_pcts) * (last_pcts < target_depths)):
                keep = True

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

            if keep:
                info('Keeping this exposure')
            else:
                info('Not keeping this exposure')
                depthiv[Yo,Xo] -= detiv
                continue

            keep_ccds[iccd] = True
            last_pcts = depthpcts

            if np.all(depthpcts >= target_depths):
                info('Reached all target depth percentiles for band', band)
                break

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

    if get_depth_maps:
        return (keep_ccds, overlapping_ccds, depthmaps)
    return keep_ccds, overlapping_ccds

