import matplotlib as mpl
mpl.rc("text",usetex=True)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import h5py
from scipy.interpolate import griddata

fs = 20

################################################################################################
# Function to create the Galactocentric profile of CRs
def func_CR_pro(map_type, Ekpr, CR_map, Rmax, NR):

    # Size of the CR map
    Nx, Ny=CR_map.shape

    # Create a grid of coordinates
    x, y=np.ogrid[:Nx, :Ny]
    Rkpc=np.linspace(0.0,Rmax,NR) 

    # Define the parameters of the ring
    center=(Nx // 2, Ny // 2)
    Rpix=Rkpc*center[0]/Rmax
    CR_pro=np.zeros([2*(NR-1),3])
    # CR_pro=np.zeros([(NR-1),3])

    # Make the Galactocentric profile
    for i in range(NR-1):
        # Create a circular mask
        mask=(x-center[0])**2+(y-center[1])** 2 <= Rpix[i+1]**2
        mask&=(x-center[0])**2+(y-center[1])** 2 >= Rpix[i]**2

        # Apply the mask to the image
        masked_image = np.copy(CR_map)
        masked_image[~mask] = np.nan  # Set pixels outside the ring to nan

        # CR_pro[i,:]=np.nanmean(masked_image), np.nanmin(masked_image), np.nanmax(masked_image)
        CR_pro[2*i,:]=np.nanmean(masked_image), np.nanmin(masked_image), np.nanmax(masked_image)
        CR_pro[2*i+1,:]=np.nanmean(masked_image), np.nanmin(masked_image), np.nanmax(masked_image)
        # CR_pro[2*i,:]= np.nanpercentile(masked_image,50), np.nanpercentile(masked_image,5), np.nanpercentile(masked_image,95)
        # CR_pro[2*i+1,:]= np.nanpercentile(masked_image,50), np.nanpercentile(masked_image,5), np.nanpercentile(masked_image,95)

    if(map_type=='CR'):
        if(Rmax==20.0):
            mp=0.938272 # GeV
            vp=np.sqrt((Ekpr+mp)**2-mp*mp)*3.0e10/(Ekpr+mp) # cm/s

            CR_pro*=1.0/Ekpr**2 
            CR_pro*=4.0*np.pi*1.0e13*1.0e-4/vp # 1.0e-13 GeV^-1 cm^-3
        if(Rmax==10.0):
            CR_pro*=1.0e22 # 1.0e-13 GeV^-1 cm^-3

    Rkpc_plot=np.sort(np.concatenate((Rkpc[0:-1], Rkpc[1:]-(Rkpc[1]-Rkpc[0])*1.0e-2), axis=0))

    return Rkpc_plot, CR_pro

# ################################################################################################
# Read the CR map from the point source simulations
with h5py.File('Hydrogen_1.h5', 'r') as hf:
    energy_key=list(hf['Data'].keys())
    nCR=np.array([hf['Data'][name][:] for name in energy_key]) # MeV cm^-2 s^-1 sr^-1

# Convert unit from MeV cm^-2 s^-1 sr^-1 to GeV m^-2 s^-1 sr^-1
nCR*=10.0

# Define the grid of the CR data cube
NE, Nz, Nx, Ny=nCR.shape
print('Data cube size (NE, Nz, Nx, Ny):',NE, Nz, Nx, Ny)
Ek=np.logspace(7,15,NE)*1.0e-9
x=np.linspace(-20,20,Nx)
y=np.linspace(-20,20,Ny)
z=np.linspace(-4,4,Nz)
Ekpr=30.0 # GeV -> Energy to print the map 

iE=np.argmax(Ek>Ekpr)
iz=Nz//2

print('Print map at E=',Ek[iE],'GeV')
CR_map=nCR[iE,iz,:,:]
in_map=2.0-np.log10(nCR[iE+2,iz,:,:]/nCR[iE,iz,:,:])/np.log10(Ek[iE+2]/Ek[iE])

# Create a finer grid
x_fine= np.linspace(-20,20,10*(Nx-1)+1)
y_fine= np.linspace(-20,20,10*(Nx-1)+1)
print('dx =',x[1]-x[0],'kpc -> dx_fine=',x_fine[1]-x_fine[0],'kpc')

# Interpolate the CR map of the disk to a finer grid
x, y=np.meshgrid(x,y)
x_fine, y_fine=np.meshgrid(x_fine,y_fine)
CR_map_fine1=griddata((x.flatten(),y.flatten()),nCR[iE,iz,:,:].flatten(),(x_fine,y_fine),method='cubic')
CR_map_fine2=griddata((x.flatten(),y.flatten()),nCR[iE+2,iz,:,:].flatten(),(x_fine,y_fine),method='cubic')

CR_map_fine=CR_map_fine1
in_map_fine=2.0-np.log10(CR_map_fine2/CR_map_fine1)/np.log10(Ek[iE+2]/Ek[iE])

################################################################################################
# Read the image from Andres
# img = mpimg.imread("fg_proton_10GeV.png")

# # Convert the image to the CR map
# img_array = np.mean(np.array(img), axis=2)
# scale =  4.223e3/np.max(img_array-np.min(img_array))
# CR_map = (img_array-np.min(img_array)) * scale
# CR_map = CR_map[0:-1,1:-2]

NR=101
Rkpc_plot, CR_pro=func_CR_pro('CR',Ekpr,CR_map_fine,20.0,NR)
Rkpc_plot, in_pro=func_CR_pro('in',Ekpr,in_map,20.0,NR)

################################################################################################
# Plot the CR profile
fig=plt.figure(figsize=(10, 8))
ax=plt.subplot(111)

Rg_dif, nCR_dif, err_Rg_dif=np.loadtxt("data_nCR_30GeV_gamma_dif.dat",unpack=True,usecols=[0,1,2])
Rg_mcl, nCR_mcl, err_nCR_mcl=np.loadtxt("data_nCR_30GeV_gamma_mcl.dat",unpack=True,usecols=[0,1,2])

ax.plot(Rkpc_plot,CR_pro[:,0],'r--',linewidth=3.0)
ax.fill_between(Rkpc_plot,CR_pro[:,1],CR_pro[:,2],color=[(255.0/255.0,102.0/255.0,102.0/255.0)],label=r'${\rm Andres}$')

ax.errorbar(Rg_dif,nCR_dif,nCR_dif*0.0,err_Rg_dif-Rg_dif,'o',color='royalblue',markersize=10.0,elinewidth=2.5,label=r'{\rm Diffuse emission}')
ax.errorbar(Rg_mcl,nCR_mcl,err_nCR_mcl-nCR_mcl,Rg_mcl*0.0,'go',markersize=10.0,elinewidth=2.5,label=r'{\rm Molecular clouds}')
Rg_dif_local=np.array([0.0, 15.0])
nCR_dif_local=np.array([0.453, 0.453])
ax.plot(Rg_dif_local, nCR_dif_local, 'k-', linewidth=2, label=r'{\rm Local CR data}')
ax.legend()
ax.set_xlabel(r'$R {\rm (kpc)}$',fontsize=fs)
ax.set_ylabel(r'$n_{\rm p}(30\,{\rm GeV})\,{ (10^{-13}\, {\rm GeV^{-1}\, cm^{-3}})}$',fontsize=fs)
for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
    label_ax.set_fontsize(fs)
ax.set_xlim(0,15)
ax.set_ylim(0,3.5)
ax.legend(loc='upper right', prop={"size":fs})
ax.grid(linestyle='--')

plt.savefig("fg_CR_profile_real.png")

################################################################################################
# Plot the CR index profile
fig=plt.figure(figsize=(10, 8))
ax=plt.subplot(111)

ax.plot(Rkpc_plot,in_pro[:,0],'r--',linewidth=3.0)
ax.fill_between(Rkpc_plot,in_pro[:,1],in_pro[:,2],color=[(255.0/255.0,102.0/255.0,102.0/255.0)],label=r'${\rm Andres}$')

ax.legend()
ax.set_xlabel(r'$R {\rm (kpc)}$',fontsize=fs)
ax.set_ylabel(r'$n_{\rm p}(30\,{\rm GeV})\,{ (10^{-13}\, {\rm GeV^{-1}\, cm^{-3}})}$',fontsize=fs)
for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
    label_ax.set_fontsize(fs)
ax.set_xlim(0,15)
ax.set_ylim(2.0,3.5)
ax.legend(loc='upper right', prop={"size":fs})
ax.grid(linestyle='--')

plt.savefig("fg_in_profile.png")

################################################################################################
# Plot the CR map 
fig=plt.figure(figsize=(10, 8))
ax=plt.subplot(111)

c = plt.imshow(CR_map_fine, cmap ='magma', extent =[-20, 20, -20, 20], interpolation ='nearest', origin ='lower') 
# c = plt.imshow(masked_image, cmap ='magma', extent =[-20, 20, -20, 20], interpolation ='nearest', origin ='lower') 

cbar = plt.colorbar(c)
cbar.set_label(r'$E_k^2\phi_{\rm CR}(E_k)\,({\rm GeV\, m^{-2}\, s^{-1}\, sr^{-1}})$', fontsize=fs)
cbar.ax.tick_params(labelsize=fs)

plt.title(r'$E_k=%.2f\, {\rm GeV}$' % Ek[iE], fontsize=fs) 
for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
    label_ax.set_fontsize(fs)

plt.savefig("fg_%dGeV_real.png" % int(Ek[iE]))