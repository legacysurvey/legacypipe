#!/usr/bin/env python

""" Takes a list of images and sextractor catalogs and appends\
seeing (C3SEE), zeropoints (C3ZP), and limiting magnitudes (C3lmtmag)\
to the header of each image."""

__author__ = 'Michael Medford <MichaelMedford@berkeley.edu'

from ztfcoadd import utils
import sys, os, shutil
import psycopg2
from astropy.io import fits
import numpy as np
import copy

def calc_lmt_mag(var, seeing, zp):

    return -2.5 * np.log10(5.0*np.sqrt(3.14159*seeing*seeing/1.01/1.01)*var*1.48) + zp

def calc_p9_zp(im):

    with fits.open(im) as f:
        header = f[0].header

    if 'APPAR01' in header:
        ZeroPoint = header['APPAR01']
        AirMassTerm = header['APPAR03']
        TimeTerm = header['APPAR05']
        Time2Term = header['APPAR06']
        ObsJD = header['OBSJD']
        MedianJD = header['APMEDJD']
        Airmass = header['AIRMASS']
    
        A = ZeroPoint
        B = AirMassTerm * Airmass
        C = TimeTerm * (ObsJD - MedianJD)
        D = Time2Term * (ObsJD - MedianJD)**2.
        E = 2.5*np.log10(60)
    
        return A + B + C + D + E
    else:
        return 0

def query_desi_stars(query_dict, desi_con):

    desi_matchquery = "SELECT id, ra, dec, {flt} " \
    "from dr1.ps1 where q3c_poly_query(ra, dec, "\
    "'{{{ra_ll}, {dec_ll}, {ra_lr}, {dec_lr}, {ra_ur}, {dec_ur}, {ra_ul}, {dec_ul}}}') "\
    " and {flt} >= 16.0 and {flt} <= 19.0;"

    cursor = desi_con.cursor()
    cursor.execute(desi_matchquery.format(**query_dict))

    # print desi_matchquery.format(**query_dict)

    result = np.array(cursor.fetchall(), dtype=[('id','<i8'),
                                                    ('ra','<f8'),
                                                    ('dec','<f8'),
                                                    ('mag','<f8')])
    cursor.close()
    return result

def query_iptf_stars(query_dict, iptf_con):

    # iptf_matchquery = "SELECT id, ra, dec, {flt} " \
    # "from calibration where q3c_poly_query(ra, dec, "\
    # "'{{{ra_ll}, {dec_ll}, {ra_lr}, {dec_lr}, {ra_ur}, {dec_ur}, {ra_ul}, {dec_ul}}}') "\
    # " and {flt} >= 16.0 and {flt} <= 19.0;"
    # print iptf_matchquery.format(**query_dict)

    query_dict['flt'] = 'sdss_' + query_dict['flt'][0].lower()

    iptf_matchquery = "SELECT id, ra, dec, mag " \
    "from {flt} where mag >= 16.0 and mag <= 19.0 and q3c_poly_query(ra, dec, "\
    "'{{{ra_ll}, {dec_ll}, {ra_lr}, {dec_lr}, {ra_ur}, {dec_ur}, {ra_ul}, {dec_ul}}}');"

    cursor = iptf_con.cursor()
    cursor.execute(iptf_matchquery.format(**query_dict))
    # print iptf_matchquery.format(**query_dict)

    result = np.array(cursor.fetchall(), dtype=[('id','<i8'),
                                                    ('ra','<f8'),
                                                    ('dec','<f8'),
                                                    ('mag','<f8')])
    cursor.close()
    return result

def return_star_info(catalog, real_stars):

    these_fluxes = []
    these_seeings = []
    these_zps = []

    for row in catalog:
        mag_c = row['MAG_AUTO']
        ra_c = row['X_WORLD']
        dec_c = row['Y_WORLD']
        fwhm = row['FWHM_IMAGE']
        flux = row['FLUX_AUTO']

        these_fluxes.append(flux)

        sep = 3600. * ( (np.cos(dec_c*np.pi/180.) * (ra_c - real_stars['ra']))**2 + (dec_c - real_stars['dec'])**2)**0.5

        match = real_stars[sep <= 2.]

        seeing = fwhm * 1.01 # arcsec (1.01 arcsec/pixel = PTF plate scale)
        these_seeings.append(seeing)

        for mps1 in match['mag']:
            zp = mps1 - mag_c
            these_zps.append(zp)

    return these_fluxes, these_seeings, these_zps

def zpsee_info(scie_list, pre_cat_list, debug, coaddFlag=False):

    zps = []
    seeings = []
    lmt_mags = []
    skys = []
    skysigs = []

    for i,im in enumerate(scie_list):
        utils.print_d("%i/%i "%(i+1,len(scie_list))+im.split('/')[-1],debug)

        with fits.open(im) as f:
            image_data = f[0].data
            image_header = f[0].header

        weight = im.replace("sciimg","weight")

        with fits.open(weight) as f:
            weight_data = f[0].data

        mask = im.replace("sciimg","mskimg")
        with fits.open(mask) as f:
            mask_data = f[0].data

        cat = utils.parse_sexcat(pre_cat_list[i], bin=True)
        cat = cat[cat['FLAGS'] == 0]
        cat = cat[cat['CLASS_STAR'] <= 0.1]

        real_stars_fname = im.replace(".fits",".real_stars.npz")
        real_stars = np.load(real_stars_fname)
        desi_stars = real_stars['desi_stars']
        iptf_stars = real_stars['iptf_stars']

        these_fluxes, these_seeings, these_zps = return_star_info(cat, desi_stars)
        result = copy.deepcopy(desi_stars)

        if len(these_zps) == 0:
            these_fluxes, these_seeings, these_zps = return_star_info(cat, iptf_stars)
            result = copy.deepcopy(iptf_stars)

        np.savez(real_stars_fname,desi_stars=desi_stars,iptf_stars=iptf_stars,real_stars=result)

        seeing = np.median(these_seeings)
        zp = np.median(these_zps)

        if not coaddFlag:
            cond0 = mask_data & 6141 == 0
            cond1 = mask_data & 2050 == 0
            image_data_flat = image_data[cond0 & cond1]
        else:
            image_data_flat = image_data[image_data != 0]

        sky = np.median(image_data_flat)
        med_image = np.abs(image_data_flat - sky)
        var = np.median(med_image)

        lmt_mag = calc_lmt_mag(var,seeing,zp)

        utils.print_d('Stars in Cat %i | Stars in Results %i'%(len(cat),len(result)),debug)
        utils.print_d('Number of Matches = %i'%len(these_zps),debug)
        utils.print_d('Median ZP = %.4f'%zp,debug)
        utils.print_d('Median Seeing = %.4f'%seeing,debug)
        utils.print_d('Limiting Magnitude = %.4f'%lmt_mag,debug)
        utils.print_d('SKY = %.4f'%sky,debug)
        utils.print_d('SKYSIG = %.4f'%var,debug)
        seeings.append(seeing)
        zps.append(zp)
        lmt_mags.append(lmt_mag)
        skys.append(sky)
        skysigs.append(var)
        utils.print_d('',debug)

    return np.array(zps), np.array(seeings), np.array(lmt_mags), np.array(skys), np.array(skysigs)

def update_image_headers(zp, see, lmt_mag, skys, skysigs, scie_list, debug):

    for zeropoint, seeing, mag, sky, skysig, image in zip(zp, see, lmt_mag, skys, skysigs, scie_list):

        with fits.open(image) as f:
            data = f[0].data
            header = f[0].header

        header["C3ZP"] = zeropoint
        header["C3SEE"] = seeing
        header["C3LMTMAG"] = mag
        header["C3SKY"] = sky
        header["C3SKYSIG"] = skysig

        fits.writeto(image,data,header,overwrite=True)
