# Example:
# # srun -N 1 -C haswell -t 04:00:00 -q interactive python modify_psfex_profiles.py --ccd_path /dvs_ro/cfs/cdirs/cosmo/work/legacysurvey/dr11/survey-ccds-decam-dr11-merged.fits --base_dir /dvs_ro/cfs/cdirs/cosmo/work/legacysurvey/dr11/ --output_dir /pscratch/sd/r/rongpu/dr11/psfex-patched > modify_psfex_profiles.log
# Or simply get the patched PSFEx model for an (unpatched) PSFEx file:
# # from legacyzpts import modify_psfex_profiles
# # computed_patched_psfex('/dvs_ro/cfs/cdirs/cosmo/work/legacysurvey/dr11/calib/psfex/decam/CP/V5.5/CP20230515/c4d_230516_085330_ooi_z_v1-psfex.fits', 'z')

import os, warnings
import numpy as np
from astropy.table import Table
import fitsio
from astropy.io import fits
from multiprocessing import Pool
from scipy.optimize import curve_fit
import argparse


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


n_processes = 128

radius_lim1, radius_lim2 = 5.0, 6.0
radius_lim3, radius_lim4 = 7., 8.

params = {
'g_weight2': 0.00045, 'g_plexp2': -2.,
'r_weight2': 0.00033, 'r_plexp2': -2.,
'i_weight2': 0.00033, 'i_plexp2': -2.,
'z_alpha2': 17.650, 'z_beta2': 1.7, 'z_weight2': 0.0145,
}

outlier_ccd_list = ['N20', 'S8', 'S10', 'S18', 'S21', 'S27']
params_outlier = {'z_alpha2': 16, 'z_beta2': 2.3, 'z_weight2': 0.0095}
pixscale = 0.262


def computed_patched_psfex(psfex_path, band, recompute=False):

    data = Table(fitsio.read(psfex_path))
    #print('Reading', psfex_path)
    hdu = fits.open(psfex_path)
    data = Table(hdu[1].data)
    #print(len(data))

    if 'psf_patch_ver' in data.colnames:
        print('Input PsfEx file is already patched:', psfex_path)
        if not recompute:
            return None

    from legacypipe.survey import get_git_version
    version = get_git_version()

    data['psf_patch_ver'] = version
    data['moffat_alpha'] = 0.
    data['moffat_beta'] = 0.
    # sum of the difference between the original and new PSF model (first eigen-image)
    data['sum_diff'] = 0.
    # the factor from fitting the new PSF with the original one; should be close to unity
    data['fit_original'] = 0.
    data['failure'] = False

    # loop over the CCDs in each exposure (note: not all CCDs meet ccd_cuts)
    for ccd_index in range(len(data)):

        # expnum_str = data['expnum'][ccd_index]
        ccdname = data['ccdname'][ccd_index].strip()

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

        # catch problematic psfex images
        if norm_factor==0 or np.isnan(norm_factor) or np.isinf(norm_factor):
            norm_factor = 1.

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
            try:
                popt, pcov = curve_fit(get_sb_moffat, radius[mask], psfi_flat[mask]/(pixscale**2), bounds=((0, 1.8), np.inf))
                alpha, beta = popt
            except RuntimeError:
                print("Error: "+psfex_path+", "+ccdname+".")
                print("Error: fit failed to converge.")
                alpha, beta = 0.8, 2.2  # using default values
                data['failure'][ccd_index] = True
                import traceback
                traceback.print_exc()

        #print('{} {} alpha, beta = {:.3f}, {:.3f}'.format(ccdname, band, alpha, beta))

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

    return data


def _wrapper(image_filename):

    image_path = os.path.join(_base_dir, 'images', image_filename)
    band = fitsio.read_header(image_path)['BAND']
    #print('band = {}'.format(band))

    psfex_filename = image_filename.replace('.fits.fz', '-psfex.fits')
    psfex_path = os.path.join(_base_dir, 'calib/psfex', psfex_filename)
    output_path = os.path.join(_output_dir, psfex_filename)

    if os.path.isfile(output_path):
        print('Output exists:', output_path)
        return None

    if not os.path.exists(psfex_path):
        print('Input PsfEx file does not exist:', psfex_path)
        return None

    data = computed_patched_psfex(psfex_path, band)
    if data is None:
        return None

    # Save new PSFEx file
    if not os.path.exists(os.path.dirname(output_path)):
        try:
            os.makedirs(os.path.dirname(output_path))
        except FileExistsError:  # in case other processes created the directory
            pass
    data.write(output_path)
    print('Wrote', output_path)

    return None


def main():

    global _base_dir, _output_dir
    # _base_dir = '/dvs_ro/cfs/cdirs/cosmo/work/legacysurvey/dr11/'
    # _output_dir = '/pscratch/sd/r/rongpu/dr11/psfex'

    parser = argparse.ArgumentParser(description='Create patched PSFEx models')
    parser.add_argument('--ccd_path', type=str, required=True, help='path of the survey-ccds file')
    parser.add_argument('--base_dir', type=str, required=True, help='base directory (e.g., /dvs_ro/cfs/cdirs/cosmo/work/legacysurvey/dr11)')
    parser.add_argument('--output_dir', type=str, required=True, help='output directory')
    args = parser.parse_args()
    surveyccd_path, _base_dir, _output_dir = args.ccd_path, args.base_dir, args.output_dir

    ccd = Table(fitsio.read(surveyccd_path, columns=['expnum', 'image_filename']))

    # Only keep unique exposures
    _, idx = np.unique(ccd['expnum'], return_index=True)
    ccd = ccd[idx]

    with Pool(processes=n_processes) as pool:
        res = pool.map(_wrapper, ccd['image_filename'])


if __name__=="__main__":
    main()
