########################################################################################################
# This is the code to get the line-of-sight map of the 3D reconstructed gas maps.
########################################################################################################

import numpy as np
from astropy.io import fits
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
import argparse
from astropy.wcs import WCS
from matplotlib.ticker import MultipleLocator
from numpy import meshgrid
mpl.rc("text",usetex=True)
import h5py
import healpy as hp
from healpy.newvisufunc import projview, newprojplot
from scipy.interpolate import interpn

extension = "png"

## Constants
NA = 6.022e23                                                                      ## Avogadro constant
mol_weight_HI = 1.008                                                              ## in g
weight_HI = mol_weight_HI / NA                                                     ## mass of HI molecule in g
M_sun = 2e33                                                                       ## in g
HImassinMsol = weight_HI / M_sun                                                   ## mass of H2 molecule in M_sun
pc2cm = 3.086e18                                                                   ## pc to cm
Msolpc3_HI = HImassinMsol*(pc2cm**3)                                               ## conversion surface denisty to M_sun/pc^2 for HI
Msolpc3_H2 = 2.0*HImassinMsol*(pc2cm**3)                                           ## conversion surface denisty to M_sun/pc^2 for H2


## ========================================================
## Plot surface mass density for the gas cubes
## ========================================================
def make_surfmassdens_vLSR_plot(args):

    ## Get the distance to the Galactic center
    if(args.vLSR == "BEG03"):
        Ro = 8.
    elif(args.vLSR == "SBM15"):
        Ro = 8.15

    ## Get the gas cube to be plotted
    if(args.type == "HI"):
        hdul_gas = fits.open("HI_dens_mean_%s.fits" % (args.vLSR)) 
        cube_gas = hdul_gas[0].data

        ## Size of the gas cube
        x0 = -24.0 # kpc
        y0 = -24.0 # kpc
        z0 = -2.0  # kpc

    elif(args.type == "H2"):
        hdul_gas = fits.open("H2_dens_mean_%s.fits" % (args.vLSR)) 
        cube_gas = hdul_gas[0].data

        ## Size of the gas cube
        x0 = -16.0 # kpc
        y0 = -16.0 # kpc
        z0 = -0.5  # kpc

    ## Get the size of the gas cube
    nxM, nyM, nzM = cube_gas.shape

    ## Create a figure
    fig_height_inches = 4.7
    fig, ax = plt.subplots(figsize=[fig_height_inches/0.85, fig_height_inches])

    ## Choose the coordinates as in Mertsch & Phan 2022
    x0_min = x0
    x0_max = -x0
    y0_min = y0
    y0_max = -y0

    ## Compute the surface mass density
    if(args.type == "H2"):
        surfdens = Msolpc3_H2*(-2.0*z0*1.0e3/nzM)*np.sum(cube_gas, axis=2).T
    else:
        surfdens = Msolpc3_HI*(-2.0*z0*1.0e3/nzM)*np.sum(cube_gas, axis=2).T
    im = ax.imshow(surfdens, origin='lower', extent=[x0_min, x0_max, y0_min, y0_max], cmap='magma', vmin=0, vmax=1.5e1)
            
    ## Colourbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cax.set_ylabel(r"$\Sigma \, \mathrm{[M_{\odot} \, \mathrm{pc}^{-2}]}$")  

    ## Plot ranges, labels, grid
    ax.set_xlim([x0, -x0])
    ax.set_ylim([y0, -y0])
    ax.set_xlabel(r"$x\,{\rm [kpc]}")
    ax.set_ylabel(r"$y\,{\rm [kpc]}")
    plt.subplots_adjust(left=0.12, right=0.9, bottom=0.09, top=0.97)
    plt.savefig("surfmassdens_%s_%s.%s" % (args.type, args.vLSR, extension), dpi=600)
    plt.close()


## ========================================================
## Get the los integration at pixel ipix
## ========================================================
def get_slice(num, NSIDE, Ro, ipix, args):

    ## Get the gas cube to be plotted
    if(args.type == "HI"):
        hdul_gas = fits.open("HI_dens_mean_%s.fits" % (args.vLSR)) 
        cube_gas = hdul_gas[0].data

        ## Size of the gas cube
        x0 = -24.0 # kpc
        y0 = -24.0 # kpc
        z0 = -2.0  # kpc

    elif(args.type == "H2"):
        hdul_gas = fits.open("H2_dens_mean_%s.fits" % (args.vLSR)) 
        cube_gas = hdul_gas[0].data

        ## Size of the gas cube
        x0 = -16.0 # kpc
        y0 = -16.0 # kpc
        z0 = -0.5  # kpc

    nx, ny, nz = cube_gas.shape

    dx = -2*x0/nx
    dy = -2*y0/ny    
    dz = -2*z0/nz

    xp = np.linspace(x0+dx/2, -x0-dx/2, nx) ## 1D array of bin in x
    yp = np.linspace(y0+dy/2, -y0-dy/2, ny) ## 1D array of bin in y
    zp = np.linspace(z0+dz/2, -z0-dz/2, nz) ## 1D array of bin in z

    points = (xp, yp, zp)

    ## Adopt rescale_NSIDE times larger nside -> rescale_NSIDE^2 times more pixel; sample ls and bs from these finer pixels
    thetas, phis = hp.pix2ang(NSIDE, ipix, nest=True, lonlat=False) ## returns theta, phi in rad
 
    ls = np.repeat(phis,              num)
    bs = np.repeat(np.pi/2. - thetas, num)
    smax = -x0*1.8 # kpc -> Maximum distance up to which the line of sight integration is performed (~40 kpc for HI and 28 kpc for H2) 
    ss = smax * np.linspace(1.0/num,1.0,num) # array of distances for integration

    # ## Get the distance to the Galactic center
    # if(args.vLSR=='BEG03'):
    #     Ro = 8. ## as in BEG03
    # elif(args.vLSR=='SBM15'):
    #     Ro = 8.15 ## as in our version of SBM15

    ## Compute (x, y, z)
    ## Convention: Sun is at (x,y,z) = (Rsol,0,0) and ell=0 is pointing opposite to the x-direction
    xs = -ss * np.cos(ls) * np.cos(bs) + Ro
    ys = -ss * np.sin(ls) * np.cos(bs)
    zs = ss * np.sin(bs)

    ## Create collection of points where the gas density is known
    points_intr = np.vstack((xs, ys, zs)).T

    ## Create the line-of-sight integration for the pixel ipix
    los_int = np.sum(interpn(points, cube_gas, points_intr, bounds_error=False, fill_value=0.0))

    return los_int*smax*1.0e3/num


## ========================================================
## Create the healpix map for the los integration 
## ========================================================
def los_integrate_map(num, NSIDE, Ro, args):
    npix = hp.nside2npix(NSIDE)
    hpxmap = np.zeros(npix)

    print('Hopefully we get something')
    print('--------------------')
    for ipix in range(npix):
        hpxmap[ipix] = get_slice(num, NSIDE, Ro, ipix, args)
        if(ipix%int(npix/20)==0):
            print('-', end='')

    np.save('los_map_%s_%s_Ro_%.2f_NSIDE_%d_num_%d.npy' % (args.type, args.vLSR, Ro, NSIDE, num), hpxmap)

    return hpxmap


if __name__ == '__main__':

    ## ---------------------------------------------------------------------------------------
    ## Command line argument parser to specify the gas flow model and the data type (HI or H2) 
    ## ---------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Gaussian variational inference.')
    parser.add_argument('-vLSR', metavar='vLSR', type=str, default="BEG03", help='gas flow model')
    parser.add_argument('-type', metavar='type', type=str, default="HI",    help='type of the gas cube: HI, H2, or both') 
    args = parser.parse_args()

    ## Create the surface mass denisty plot 
    # make_surfmassdens_vLSR_plot(args)

    ## Specify the precision (num) and NSIDE of the los map  
    num = 5000 # This is chosen such that teh integration step ds <~ gas disk height/10   
    NSIDE = 32
    Ro = 8.5 # Position of the Solar System for the los map

    ## Create the los map
    # los_integrate_map(num, NSIDE, Ro, args)

    ## Plot the los map  
    npix = hp.nside2npix(NSIDE)
    hpxmap = np.zeros(npix)
    hpxmap = np.load('los_map_%s_%s_Ro_%.2f_NSIDE_%d_num_%d.npy' % (args.type, args.vLSR, Ro, NSIDE, num))*3.086e18

    projview(
        np.log10(hpxmap), 
        title=r'Log10 Column Density Map from Data',
        coord=["G"], cmap='magma',
        min=19, max=23,
        nest=True, 
        unit=r'$log_{10}N_{\rm HI}\, [{\rm cm}^{-2}]$',
        graticule=True, graticule_labels=True, 
        # xlabel=r'longitude (deg)',
        # ylabel=r'latitude (deg)',
        projection_type="mollweide"
    )
    plt.savefig('fg_los_map_%s_%s_Ro_%.2f_NSIDE_%d_num_%d.png' % (args.type, args.vLSR, Ro, NSIDE, num), dpi=150)
    plt.close()
