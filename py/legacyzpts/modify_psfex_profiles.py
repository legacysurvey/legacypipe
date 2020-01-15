from __future__ import division, print_function
import sys, os, glob, time, warnings, gc
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table, vstack, hstack
import fitsio
from astropy.io import fits

from scipy.optimize import curve_fit

def get_frac_moffat(r, alpha, beta):
    """
    Calculate the fraction of light within radius r of a Moffat profile.
    """    
    frac = 1 - alpha**(2*(beta-1))*(alpha**2 + r**2)**(1-beta)
    return(frac)

def get_sb_moffat(r, alpha, beta):
    """
    Calculate the surface brightness of light at radius r of a Moffat profile.
    The integral (i.e., total flux) is unity by definition.
    """
    i = (beta-1)/(np.pi * alpha**2)*(1 + (r/alpha)**2)**(-beta)
    return i

def get_sb_moffat_plus_power_law(r, alpha1, beta1, plexp2, weight2):
    """
    Calculate the surface brightness of light at radius r of the sum of two Moffat profiles.
    The integral (i.e., total flux) is NOT unity.
    """
    i = (beta1-1)/(np.pi * alpha1**2)*(1 + (r/alpha1)**2)**(-beta1) \
        + weight2 *r**(plexp2)
    return i

def get_sb_double_moffat(r, alpha1, beta1, alpha2, beta2, weight2):
    """
    Calculate the surface brightness of light at radius r of the sum of two Moffat profiles.
    The integral (i.e., total flux) is NOT unity.
    """
    i = (beta1-1)/(np.pi * alpha1**2)*(1 + (r/alpha1)**2)**(-beta1) \
        + weight2 * (beta2-1)/(np.pi * alpha2**2)*(1 + (r/alpha2)**2)**(-beta2)
    return i


test_q = True  # only process a small number of exposures

output_dir = '/global/project/projectdirs/desi/users/rongpu/dr9/new-psfex-v2'
surveyccd_path = '/global/project/projectdirs/cosmo/work/legacysurvey/dr9/survey-ccds-decam-dr9-cut.fits.gz'

radius_lim1, radius_lim2 = 5.0, 6.0
radius_lim3, radius_lim4 = 7., 8.

params = {
'g_weight2': 0.00045, 'g_plexp2': -2.,
'r_weight2': 0.00033, 'r_plexp2': -2.,
'z_alpha2': 17.650, 'z_beta2': 1.7, 'z_weight2': 0.0145,
}

outlier_ccd_list = ['N20', 'S8', 'S10', 'S18', 'S21', 'S27']
params_outlier = {'z_alpha2': 16, 'z_beta2': 2.3, 'z_weight2': 0.0095}
pixscale = 0.262

ccd = fitsio.read(surveyccd_path)
ccd = Table(ccd)

mask = ccd['ccd_cuts']==0
print(np.sum(mask)/len(mask))
ccd = ccd[mask]

unique_expnum = np.unique(ccd['expnum'])
print(len(unique_expnum))

# randomly select exposures for testing
if test_q:
    np.random.seed(123)
    exp_index_list = np.random.choice(len(unique_expnum), size=50, replace=False)
else:
    exp_index_list = np.arange(len(unique_expnum))

# loop over the unique exposures
# for exp_index in [0]:
for exp_index in exp_index_list:

    mask = ccd['expnum']==unique_expnum[exp_index]
    band = ccd['filter'][mask][0]
    print('band = {}'.format(band))

    image_filename = ccd['image_filename'][mask][0]
    psfex_filename = image_filename[:image_filename.find('.fits.fz')]+'-psfex.fits'
    psfex_filename_new = image_filename[:image_filename.find('.fits.fz')]+'-psfex-patched.fits'
    psfex_path = os.path.join('/project/projectdirs/cosmo/work/legacysurvey/dr9/calib', psfex_filename)

    output_path = os.path.join(output_dir, psfex_filename_new)
    if os.path.isfile(output_path):
        raise ValueError
        # continue

    hdu = fits.open(psfex_path)
    data = Table(hdu[1].data)
    print(len(data))

    data['psf_patch_ver'] = 'psf_patch_ver'
    data['moffat_alpha'] = 0.
    data['moffat_beta'] = 0.
    # sum of the difference between the original and new PSF model (first eigen-image)
    data['sum_diff'] = 0.
    # the factor from fitting the new PSF with the original one; should be close to unity
    data['fit_original'] = 0.

    # loop over the CCDs in each exposure (note: not all CCDs meet ccd_cuts)
    for ccd_index in range(len(data)):

        # expnum_str = data['expnum'][ccd_index]
        ccdname = data['ccdname'][ccd_index]

        ########## Outer PSF parameters ###########
        if band=='z' and (ccdname in outlier_ccd_list):
            params_to_use = params_outlier
        else:
            params_to_use = params
            
        if band!='z':
            plexp2, weight2 = params_to_use[band+'_plexp2'], params_to_use[band+'_weight2']
        else:
            alpha2, beta2, weight2 = params_to_use[band+'_alpha2'], params_to_use[band+'_beta2'],  params_to_use[band+'_weight2']

        ############################# Modify the first eigen-image ####################################
        psf_index = 0

        psfi_original = data['psf_mask'][ccd_index, psf_index].copy()
        # normalize to a 22.5 magnitude star
        norm_factor = np.sum(psfi_original)
        psfi = psfi_original/norm_factor

        # vrange = 0.0002
        # ax = plot_cutout(psfi, pixscale, vmin=-vrange, vmax=vrange, axis_label=['DEC direction (arcsec)', 'RA direction (arcsec)'])
        # plt.show()

        grid = pixscale * np.linspace(-0.5*(psfi.shape[0]-1), 0.5*(psfi.shape[0]-1), psfi.shape[0])
        xx, yy = np.meshgrid(grid, grid)
        radius_grid = np.sqrt(xx**2 + yy**2)
        radius = radius_grid.flatten()

        ################# Moffat fit ##############

        radius_min, radius_max = 1.8, 5.0

        psfi_flat = psfi.flatten()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # popt, pcov =  curve_fit(get_sb_moffat, radius, psfi_flat/(pixscale**2))
            mask = (radius>radius_min) & (radius<radius_max)
            popt, pcov = curve_fit(get_sb_moffat, radius[mask], psfi_flat[mask]/(pixscale**2), bounds=((0, 1.8), np.inf))

        alpha, beta = popt
        print('{} {} alpha, beta = {:.3f}, {:.3f}'.format(ccdname, band, alpha, beta))

        # save the Moffat parameters
        data['moffat_alpha'][ccd_index] = alpha
        data['moffat_beta'][ccd_index] = beta

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if band!='z':
                psf_predict = pixscale**2 * get_sb_moffat_plus_power_law(radius_grid, alpha, beta, plexp2, weight2)
            else:
                psf_predict = pixscale**2 * get_sb_double_moffat(radius_grid, alpha, beta, alpha2, beta2, weight2)

        ################# Create hybrid PSF model ##############

        psfi_combine = psfi.copy()

        r1, r2 = radius_lim1, radius_lim2
        mask = (radius_grid>r1) & (radius_grid<r2)
        psfi_combine[mask] = psfi[mask] * (r2-radius_grid[mask])/(r2-r1) \
                       + psf_predict[mask] * (radius_grid[mask]-r1)/(r2-r1)

        r1, r2 = radius_lim2, radius_lim3
        mask = (radius_grid>=r1) & (radius_grid<r2)
        psfi_combine[mask] = psf_predict[mask]

        r1, r2 = radius_lim3, radius_lim4
        mask = (radius_grid>=r1) & (radius_grid<r2)
        psfi_combine[mask] = psf_predict[mask] * (r2-radius_grid[mask])/(r2-r1) \
                       + 0 * (radius_grid[mask]-r1)/(r2-r1)

        mask = (radius_grid>radius_lim4)
        psfi_combine[mask] = 0

        # restore to the original normalization
        psfi_combine = psfi_combine * norm_factor

        data['psf_mask'][ccd_index, psf_index] = psfi_combine

        #########################################################

        data['sum_diff'][ccd_index] = np.sum(psfi_original-psfi_combine)

        from scipy.optimize import minimize_scalar
        func = lambda a: np.sum((a * psfi_combine - psfi_original)**2)
        data['fit_original'][ccd_index] = minimize_scalar(func).x

        ############################# Modify the other PSFEx eigen-images ####################################

        for psf_index in range(1, 6):

            psfi = data['psf_mask'][ccd_index, psf_index]

            grid = pixscale * np.linspace(-0.5*(psfi.shape[0]-1), 0.5*(psfi.shape[0]-1), psfi.shape[0])
            xx, yy = np.meshgrid(grid, grid)
            radius_grid = np.sqrt(xx**2 + yy**2)
            radius = radius_grid.flatten()

            psfi_combine = psfi.copy()

            r1, r2 = radius_lim1, radius_lim2
            mask = (radius_grid>=r1) & (radius_grid<r2)
            psfi_combine[mask] = psfi[mask] * (r2-radius_grid[mask])/(r2-r1) \
                           + 0 * (radius_grid[mask]-r1)/(r2-r1)

            mask = (radius_grid>radius_lim2)
            psfi_combine[mask] = 0

            data['psf_mask'][ccd_index, psf_index] = psfi_combine

    ############################# Save new PSFEx file ####################################

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    data.write(output_path)
