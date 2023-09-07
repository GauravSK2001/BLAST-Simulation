import numpy as np
import os
import astropy.units as u
import astropy.io.fits as fits
import astropy.coordinates as coord
import astropy.convolution as conv
import astropy.wcs as wcs
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve as fft
import time

# Function to rebin and convolve a super-sampled array
def rebin_and_convolve(super_sampled_array, scale_factor):
    # Rebin the super-sampled array by summing pixels
    rebinned_array = super_sampled_array.reshape(
        (super_sampled_array.shape[0] // scale_factor, scale_factor,
         super_sampled_array.shape[1] // scale_factor, scale_factor)
    ).sum(-1).sum(1)

    # Define a Gaussian PSF kernel
    psf = conv.Gaussian2DKernel(x_stddev=1, y_stddev=1, theta=None)

    # Convolve the rebinned array with the PSF using FFT
    convolved_image = fft(rebinned_array, psf, mode="same")
    
    # Ensure non-negative values in the convolved image
    convolved_image[convolved_image < 0] = 0

    return convolved_image, rebinned_array

print("----------- Getting Data From GAIA Catalogue -----------")
start_time = time.time()
print(os.getcwd())
# Load data from a local file
data = fits.open(os.path.join(os.getcwd(), "Polaris_40x40_le16.fits.gz"), memmap=True)

# Define the central coordinate
Polaris_coord = coord.SkyCoord('02 31 49.09', '+89 15 50.8', frame="icrs", unit=(u.hourangle, u.deg))

retrieval_time = time.time() - start_time
print(f"Data retrieved from GAIA Catalogue. It took {retrieval_time} seconds.")

# Detector parameters
size = (6400, 9600) * u.pix
exposure_time = 5 * u.s
gain = 56 * u.electron * u.adu**(-1)
qe = 0.9

print("----------- Creating Super Sampling Array -----------")

# Super sampled array parameters
scale_factor = 3
ss_size = scale_factor * size
Y, X = int(ss_size.value[0]), int(ss_size.value[1])

# Generate a blank image
image = np.zeros((Y, X), dtype=np.float64)

print("----------- Constructing Header -----------")

# Construct WCS header for the supersampled image
header = wcs.WCS(data[1].header)
header.wcs.ctype = ["RA---TAN", "DEC--TAN"]
header.wcs.crpix = [X // 2, Y // 2]
pixel_scale = 40.0  # degrees
cdelt1 = pixel_scale / X
cdelt2 = pixel_scale / Y
header.wcs.cdelt = [cdelt1, cdelt1]
header.wcs.crval = coord.Angle("02h31m49.09s").degree, coord.Angle("+89d15m50.8s").degree

print("----------- Adding the Stars Onto the Pixels -----------")
total_stars = 0
total_added_stars = 0

for star in data[1].data:
    pix_coords = header.wcs_world2pix(np.array([[star['ra'], star['dec']]]), 1, ra_dec_order=True)[0]
    total_stars += 1
    x, y = int(pix_coords[0]), int(pix_coords[1])
    
    if 0 <= y < Y and 0 <= x < X and star["phot_g_mean_mag"] <= 16:
        n_adu = (star['phot_g_mean_flux'] * (u.electron / u.second) * exposure_time) / gain
        image[y, x] += int(n_adu.value)
        total_added_stars += 1

print(f"The total number of stars from the GAIA Catalogue is {total_stars}")
print(f"The total number of added sstars from the GAIA Catalogue is {total_added_stars}")

print("----------- Rebinning and Convolving -----------")
final_rebinned_image, rebinned_array = rebin_and_convolve(image, scale_factor)
print("----------- Noise Addition -----------")
'''
Masterbias = "master_bias.fits"  # replace with master bias path
# Open and load the FITS files
with fits.open(os.path.join(os.getcwd(), Masterbias), memmap=True) as hdul3:
    Masterbias = hdul3[0].data[2300:3500,2300:3500]
'''
#bias level measured from master bias file
Bias_level= 432.693
RON=1.38309

readout_noise = RON #i think it is in ADU
print(RON,"is the read noise")

def calculate_sky_flux(magnitude, zero_point):
    flux = 10 ** (-0.4 * (magnitude - zero_point)) 
    return flux #should be e/s
Sky=18 #probably in cph 
zeropoint=22  # i guessed a value


Poisson=np.random.poisson(int(calculate_sky_flux(Sky, zeropoint)/gain.value),(size.value).astype(int))
Read=np.random.normal(Bias_level,readout_noise,(size.value).astype(int))

Total_noise=Poisson+Read
print(np.shape(Total_noise))

print("------------ Rebinned Header ---------")

# constructing WCS header for the rebinned image
header.wcs.crpix = [X // (2 * scale_factor), Y // (2 * scale_factor)]
cdelt1 = pixel_scale / (X / scale_factor)
cdelt2 = pixel_scale / (Y / scale_factor)
header.wcs.cdelt = [cdelt1, cdelt1]

print("----------- Creating Rebinned Image -----------")

# Save the rebinned and convolved image to FITS file
output_path = os.path.join(os.getcwd(), f'Rebinned_Simulated_Image_of_scale_{scale_factor}.fits')
hdu = fits.PrimaryHDU(np.round(final_rebinned_image+Total_noise).astype(int), header=header.to_header())
hdu.writeto(output_path, overwrite=True)

print(f'Image saved at {output_path}')

# Calculate and print execution time
end_time = time.time()
elapsed_time = end_time - start_time
print(f'Execution time: {elapsed_time} seconds')
