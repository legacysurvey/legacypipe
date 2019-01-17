import numpy as np


def patch_from_coadd(coimgs, targetwcs, bands, tims, mp=None):
    from astrometry.util.resample import resample_with_wcs, OverlapError

    H,W = targetwcs.shape
    ibands = dict([(b,i) for i,b in enumerate(bands)])
    for tim in tims:
        ie = tim.getInvvar()
        img = tim.getImage()
        if np.any(ie == 0):
            # Patch from the coadd
            co = coimgs[ibands[tim.band]]
            # resample from coadd to img -- nearest-neighbour
            iy,ix = np.nonzero(ie == 0)
            if len(iy) == 0:
                continue
            ra,dec = tim.subwcs.pixelxy2radec(ix+1, iy+1)[-2:]
            xx,yy = targetwcs.radec2pixelxy(ra, dec)
            xx = np.round(xx-1).astype(np.int16)
            yy = np.round(yy-1).astype(np.int16)
            keep = (xx >= 0) * (xx < W) * (yy >= 0) * (yy < H)
            if not np.any(keep):
                continue
            img[iy[keep],ix[keep]] = coimgs[ibands[tim.band]][yy[keep],xx[keep]]

            # try:
            #     yo,xo,yi,xi,nil = resample_with_wcs(tim.subwcs, targetwcs, [])
            #     I, = np.nonzero(ie[yo,xo] == 0)
            #     if len(I):
            #         img[yo[I],xo[I]] = coimgs[ibands[tim.band]][yi[I],xi[I]]
            # except OverlapError:
            #     print('No overlap')
            del co
