import redback 
import pandas as pd
from bilby.core.prior import Uniform, Gaussian, PriorDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from redback.constants import day_to_s
from redback.model_library import all_models_dict
from collections import namedtuple

plt.rcParams.update({'font.size': 15})
plt.rc('xtick', labelsize=15) 
plt.rc('ytick', labelsize=15) 
plt.rcParams["figure.figsize"] = (7, 6.2)

#Setup
model = 'general_magnetar_driven_supernova'
function = all_models_dict[model]
time_sim = np.logspace(2, 8, 1000)/day_to_s
time_temp = np.geomspace(1e-4, 1e9, 1000, endpoint=True)/day_to_s
redshift=0.1

#Defining Parameters
E_sn=1.0e51
ej_rad=1.0e11
kappa=0.1
n_ism=1.0e-5
nn=3.0
kappa_gamma=0.1
tf=5000
output_format='dynamics_output'
mag_output_format='magnitude'
frequencyg=redback.utils.bands_to_frequency('g')[0]
frequencyr=redback.utils.bands_to_frequency('r')[0]

#Getting the grid parameters set
P1=1
P2=3

n=41
bp=np.geomspace(10**12.5,10**15.5,n)
mej=np.geomspace(1,50,n)
[X, Y] = np.meshgrid(mej, bp)

ErotP1=2.6e52*(P1)**(-2)
ErotP2=2.6e52*(P2)**(-2)
tsdP1=1.3e5*(bp/1e14)**(-2)*P1**2
tsdP2=1.3e5*(bp/1e14)**(-2)*P2**2
l0P1=ErotP1/tsdP1
l0P2=ErotP2/tsdP2
[X, tP1mg] = np.meshgrid(mej, tsdP1)
[X, tP2mg] = np.meshgrid(mej, tsdP2)

#Making the grid
EkinP1=np.zeros([n,n])
EkinP2=np.zeros([n,n])
EradP1=np.zeros([n,n])
EradP2=np.zeros([n,n])
tdifP1=np.zeros([n,n])
tdifP2=np.zeros([n,n])
peaklumP1=np.zeros([n,n])
peaklumP2=np.zeros([n,n])
peaktimP1=np.zeros([n,n])
peaktimP2=np.zeros([n,n])
vejP1=np.zeros([n,n])
vejP2=np.zeros([n,n])
gmagP1=np.zeros([n,n])
gmagP2=np.zeros([n,n])
rmagP1=np.zeros([n,n])
rmagP2=np.zeros([n,n])
tdifvejP1=np.zeros([n,n])
tdifvejP2=np.zeros([n,n])
peakvphotP1=np.zeros([n,n])
peakvphotP2=np.zeros([n,n])

for i in range(0,n):
	for j in range(0,n):
		datP1=function(time_sim, redshift, mej=mej[j], E_sn=E_sn, ejecta_radius=ej_rad, kappa=kappa, n_ism=n_ism, l0=l0P1[i], tau_sd=tsdP1[i], nn=nn, kappa_gamma=kappa_gamma, frequency=frequencyg, temperature_floor=tf, output_format=output_format)
		datP2=function(time_sim, redshift, mej=mej[j], E_sn=E_sn, ejecta_radius=ej_rad, kappa=kappa, n_ism=n_ism, l0=l0P2[i], tau_sd=tsdP2[i], nn=nn, kappa_gamma=kappa_gamma, frequency=frequencyg, temperature_floor=tf, output_format=output_format)
		gmagP1[i,j]=np.min(function(time_sim, redshift, mej=mej[j], E_sn=E_sn, ejecta_radius=ej_rad, kappa=kappa, n_ism=n_ism, l0=l0P1[i], tau_sd=tsdP1[i], nn=nn, kappa_gamma=kappa_gamma, frequency=frequencyg, temperature_floor=tf, output_format=mag_output_format))
		gmagP2[i,j]=np.min(function(time_sim, redshift, mej=mej[j], E_sn=E_sn, ejecta_radius=ej_rad, kappa=kappa, n_ism=n_ism, l0=l0P2[i], tau_sd=tsdP2[i], nn=nn, kappa_gamma=kappa_gamma, frequency=frequencyg, temperature_floor=tf, output_format=mag_output_format))
		rmagP1[i,j]=np.min(function(time_sim, redshift, mej=mej[j], E_sn=E_sn, ejecta_radius=ej_rad, kappa=kappa, n_ism=n_ism, l0=l0P1[i], tau_sd=tsdP1[i], nn=nn, kappa_gamma=kappa_gamma, frequency=frequencyr, temperature_floor=tf, output_format=mag_output_format))
		rmagP2[i,j]=np.min(function(time_sim, redshift, mej=mej[j], E_sn=E_sn, ejecta_radius=ej_rad, kappa=kappa, n_ism=n_ism, l0=l0P2[i], tau_sd=tsdP2[i], nn=nn, kappa_gamma=kappa_gamma, frequency=frequencyr, temperature_floor=tf, output_format=mag_output_format))
		EkinP1[i,j]=np.nanmax(datP1.kinetic_energy)
		EkinP2[i,j]=np.nanmax(datP2.kinetic_energy)
		EradP1[i,j]=datP1.erad_total
		EradP2[i,j]=datP2.erad_total						
		tdifP1[i,j]=datP1.tau[np.where(datP1.tau < time_sim*day_to_s)[0][0]]/day_to_s
		tdifP2[i,j]=datP2.tau[np.where(datP2.tau < time_sim*day_to_s)[0][0]]/day_to_s
		peaklumP1[i,j]=np.nanmax(datP1.bolometric_luminosity)
		peaklumP2[i,j]=np.nanmax(datP2.bolometric_luminosity)
		peaktimP1[i,j]=time_temp[np.nanargmax(datP1.bolometric_luminosity)]
		peaktimP2[i,j]=time_temp[np.nanargmax(datP2.bolometric_luminosity)]
		vejP1[i,j]=np.nanmax(datP1.v_ej)
		vejP2[i,j]=np.nanmax(datP2.v_ej)
		peakrphotindP1=np.nanargmax(datP1.bolometric_luminosity)
		if peakrphotindP1 == 999:
			peakrphotindP1 = 998
		peakvphotP1[i,j]=(datP1.r_photosphere[peakrphotindP1+1]-datP1.r_photosphere[peakrphotindP1-1])/(time_temp[peakrphotindP1+1]-time_temp[peakrphotindP1-1])/day_to_s/redback.utils.km_cgs
		peakrphotindP2=np.nanargmax(datP2.bolometric_luminosity)
		if peakrphotindP2 == 999:
			peakrphotindP2 = 998		
		peakvphotP2[i,j]=(datP2.r_photosphere[peakrphotindP2+1]-datP2.r_photosphere[peakrphotindP2-1])/(time_temp[peakrphotindP2+1]-time_temp[peakrphotindP2-1])/day_to_s/redback.utils.km_cgs
		tdifvejP1[i,j]=datP1.v_ej[np.where(datP1.tau < time_sim*day_to_s)[0][0]]	
		tdifvejP2[i,j]=datP2.v_ej[np.where(datP2.tau < time_sim*day_to_s)[0][0]]	
		
#Erad/Ekin ratio
#Some models give nans for Ekin

levels=np.linspace(0.5,4.5,1000)
tratlevels=[-1,0,1]

plt.close('all')
ax = plt.gca()
ax.set_xscale('log')
plt.contourf(X, np.log10(Y), np.log10(EkinP1/EradP1),levels=levels, cmap='coolwarm')
cbar=plt.colorbar(ticks=[0.5,1.5,2.5,3.5,4.5])
CS=plt.contour(X, np.log10(Y), np.log10(tP1mg/tdifP1/day_to_s), levels=tratlevels,colors='k',linestyles='solid')
fmt = {}
strs = ['$\\zeta = 0.1$', '$\\zeta = 1$', '$\\zeta = 10$']
for l, s in zip(CS.levels, strs):
    fmt[l] = s
ax.clabel(CS, inline=True, fontsize=15, fmt=fmt)
ax.set_xticks([1, 5, 20, 50])
ax.set_xticklabels(['1', '5', '20', '50'])
ax.set_xlabel('Ejecta Mass ($M_{\\odot}$)')
ax.set_ylabel('log(Pulsar Magnetic Field) (G)')
cbar.set_label('log($E_{\\rm kin}/E_{\\rm rad}$)', rotation=270, labelpad=20)
plt.savefig('erat_p1.pdf')
#plt.show()

plt.close('all')
ax = plt.gca()
ax.set_xscale('log')
plt.contourf(X, np.log10(Y), np.log10(EkinP2/EradP2),levels=levels, cmap='coolwarm')
cbar=cbar=plt.colorbar(ticks=[0.5,1.5,2.5,3.5,4.5])
CS=plt.contour(X, np.log10(Y), np.log10(tP2mg/tdifP2/day_to_s), levels=tratlevels,colors='k',linestyles='solid')
fmt = {}
strs = ['$\\zeta = 0.1$', '$\\zeta = 1$', '$\\zeta = 10$']
for l, s in zip(CS.levels, strs):
    fmt[l] = s
ax.clabel(CS, inline=True, fontsize=15, fmt=fmt)
ax.set_xticks([1, 5, 20, 50])
ax.set_xticklabels(['1', '5', '20', '50'])
ax.set_xlabel('Ejecta Mass ($M_{\\odot}$)')
ax.set_ylabel('log(Pulsar Magnetic Field) (G)')
cbar.set_label('log($E_{\\rm kin}/E_{\\rm rad}$)', rotation=270, labelpad=20)
plt.savefig('erat_p2.pdf')
#plt.show()

#Peak Luminosity

lumlev=np.linspace(40,46,1000)

plt.close('all')
ax = plt.gca()
ax.set_xscale('log')
plt.contourf(X, np.log10(Y), np.log10(peaklumP1),levels=lumlev, cmap='coolwarm')
cbar=plt.colorbar(ticks=[40,42,44,46])
ax.set_xticks([1, 5, 20, 50])
ax.set_xticklabels(['1', '5', '20', '50'])
ax.set_xlabel('Ejecta Mass ($M_{\\odot}$)')
ax.set_ylabel('log(Pulsar Magnetic Field) (G)')
cbar.set_label('log(Bolometric Luminsoity) (erg/s)', rotation=270, labelpad=20)
plt.savefig('peaklum_p1.pdf')
#plt.show()

plt.close('all')
ax = plt.gca()
ax.set_xscale('log')
plt.contourf(X, np.log10(Y), np.log10(peaklumP2),levels=lumlev, cmap='coolwarm')
cbar=plt.colorbar(ticks=[40,42,44,46])
ax.set_xticks([1, 5, 20, 50])
ax.set_xticklabels(['1', '5', '20', '50'])
ax.set_xlabel('Ejecta Mass ($M_{\\odot}$)')
ax.set_ylabel('log(Pulsar Magnetic Field) (G)')
cbar.set_label('log(Bolometric Luminsoity) (erg/s)', rotation=270, labelpad=20)
plt.savefig('peaklum_p2.pdf')
#plt.show()

#g-band peak

glev=np.linspace(-28,-15,1000)

plt.close('all')
fig = plt.figure()
ax = plt.gca()
ax.set_xscale('log')
cax=plt.contourf(X, np.log10(Y), -gmagP1,levels=glev, cmap='coolwarm')
cbar=fig.colorbar(cax, ticks=[-28, -25, -22, -19, -16])
cbar.ax.set_yticklabels(['28', '25','22', '19','16'])
ax.set_xticks([1, 5, 20, 50])
ax.set_xticklabels(['1', '5', '20', '50'])
ax.set_xlabel('Ejecta Mass ($M_{\\odot}$)')
ax.set_ylabel('log(Pulsar Magnetic Field) (G)')
cbar.set_label('g-band Magnitude', rotation=270, labelpad=20)
plt.savefig('peakgmag_p1.pdf')
#plt.show()

plt.close('all')
fig = plt.figure()
ax = plt.gca()
ax.set_xscale('log')
cax=plt.contourf(X, np.log10(Y), -gmagP2,levels=glev, cmap='coolwarm')
cbar=fig.colorbar(cax, ticks=[-28, -25, -22, -19, -16])
cbar.ax.set_yticklabels(['28', '25','22', '19','16'])
ax.set_xticks([1, 5, 20, 50])
ax.set_xticklabels(['1', '5', '20', '50'])
ax.set_xlabel('Ejecta Mass ($M_{\\odot}$)')
ax.set_ylabel('log(Pulsar Magnetic Field) (G)')
cbar.set_label('g-band Magnitude', rotation=270, labelpad=20)
plt.savefig('peakgmag_p2.pdf')
#plt.show()

#g-band peak

rlev=np.linspace(-28,-15,1000)

plt.close('all')
fig = plt.figure()
ax = plt.gca()
ax.set_xscale('log')
cax=plt.contourf(X, np.log10(Y), -rmagP1,levels=rlev, cmap='coolwarm')
cbar=fig.colorbar(cax, ticks=[-28, -25, -22, -19, -16])
cbar.ax.set_yticklabels(['28', '25','22', '19','16'])
ax.set_xticks([1, 5, 20, 50])
ax.set_xticklabels(['1', '5', '20', '50'])
ax.set_xlabel('Ejecta Mass ($M_{\\odot}$)')
ax.set_ylabel('log(Pulsar Magnetic Field) (G)')
cbar.set_label('r-band Magnitude', rotation=270, labelpad=20)
plt.savefig('peakrmag_p1.pdf')
#plt.show()

plt.close('all')
fig = plt.figure()
ax = plt.gca()
ax.set_xscale('log')
cax=plt.contourf(X, np.log10(Y), -rmagP2,levels=rlev, cmap='coolwarm')
cbar=fig.colorbar(cax, ticks=[-28, -25, -22, -19, -16])
cbar.ax.set_yticklabels(['28', '25','22', '19','16'])
ax.set_xticks([1, 5, 20, 50])
ax.set_xticklabels(['1', '5', '20', '50'])
ax.set_xlabel('Ejecta Mass ($M_{\\odot}$)')
ax.set_ylabel('log(Pulsar Magnetic Field) (G)')
cbar.set_label('r-band Magnitude', rotation=270, labelpad=20)
plt.savefig('peakrmag_p2.pdf')
#plt.show()

#Peak Timescale

timlev=np.linspace(1,70,1000)

plt.close('all')
ax = plt.gca()
ax.set_xscale('log')
plt.contourf(X, np.log10(Y), peaktimP1,levels=timlev, cmap='coolwarm')
cbar=plt.colorbar(ticks=[1,10,30,50,70])
ax.set_xticks([1, 5, 20, 50])
ax.set_xticklabels(['1', '5', '20', '50'])
ax.set_xlabel('Ejecta Mass ($M_{\\odot}$)')
ax.set_ylabel('log(Pulsar Magnetic Field) (G)')
cbar.set_label('Bolometric Peak Timescale (days)', rotation=270, labelpad=20)
plt.savefig('peaktim_p1.pdf')
#plt.show()

plt.close('all')
ax = plt.gca()
ax.set_xscale('log')
plt.contourf(X, np.log10(Y), peaktimP2,levels=timlev, cmap='coolwarm')
cbar=plt.colorbar(ticks=[1,10,30,50,70])
ax.set_xticks([1, 5, 20, 50])
ax.set_xticklabels(['1', '5', '20', '50'])
ax.set_xlabel('Ejecta Mass ($M_{\\odot}$)')
ax.set_ylabel('log(Pulsar Magnetic Field) (G)')
cbar.set_label('Bolometric Peak Timescale (days)', rotation=270, labelpad=20)
plt.savefig('peaktim_p2.pdf')
#plt.show()

#Ejecta Velocity

vejlev=np.linspace(1000,40000,1000)

plt.close('all')
ax = plt.gca()
ax.set_xscale('log')
plt.contourf(X, np.log10(Y), vejP1,levels=vejlev, cmap='coolwarm')
cbar=plt.colorbar(ticks=[1000,5000,10000,15000, 20000, 25000, 30000, 35000, 40000])
ax.set_xticks([1, 5, 20, 50])
ax.set_xticklabels(['1', '5', '20', '50'])
ax.set_xlabel('Ejecta Mass ($M_{\\odot}$)')
ax.set_ylabel('log(Pulsar Magnetic Field) (G)')
cbar.set_label('Final Ejecta Velocity (km/s)', rotation=270, labelpad=20)
plt.savefig('vej_p1.pdf')
#plt.show()

plt.close('all')
ax = plt.gca()
ax.set_xscale('log')
plt.contourf(X, np.log10(Y), vejP2,levels=vejlev, cmap='coolwarm')
cbar=plt.colorbar(ticks=[1000,5000,10000,15000, 20000, 25000, 30000, 35000, 40000])
ax.set_xticks([1, 5, 20, 50])
ax.set_xticklabels(['1', '5', '20', '50'])
ax.set_xlabel('Ejecta Mass ($M_{\\odot}$)')
ax.set_ylabel('log(Pulsar Magnetic Field) (G)')
cbar.set_label('Final Ejecta Velocity (km/s)', rotation=270, labelpad=20)
plt.savefig('vej_p2.pdf')
#plt.show()

#Velocity Comparison

vcomplev=np.linspace(-1,1,1000)
peakvphotP1[np.where(peakvphotP1 != peakvphotP1)[0],np.where(peakvphotP1 != peakvphotP1)[1]]=1.0
peakvphotP2[np.where(peakvphotP2 != peakvphotP2)[0],np.where(peakvphotP2 != peakvphotP2)[1]]=1.0

plt.close('all')
fig = plt.figure()
ax = plt.gca()
ax.set_xscale('log')
cax=plt.contourf(X, np.log10(Y), np.log10(np.abs(peakvphotP1)/tdifvejP1), levels=vcomplev, cmap='coolwarm')
cbar=fig.colorbar(cax, ticks=[-1,-0.5,0,0.5,1])
ax.set_xticks([1, 5, 20, 50])
ax.set_xticklabels(['1', '5', '20', '50'])
ax.set_xlabel('Ejecta Mass ($M_{\\odot}$)')
ax.set_ylabel('log(Pulsar Magnetic Field) (G)')
cbar.set_label('log(vcomp)', rotation=270, labelpad=20)
plt.savefig('vcomp_p1.pdf')
#plt.show()

plt.close('all')
fig = plt.figure()
ax = plt.gca()
ax.set_xscale('log')
cax=plt.contourf(X, np.log10(Y), np.log10(np.abs(peakvphotP2)/tdifvejP2), levels=vcomplev, cmap='coolwarm')
cbar=fig.colorbar(cax, ticks=[-1,-0.5,0,0.5,1])
ax.set_xticks([1, 5, 20, 50])
ax.set_xticklabels(['1', '5', '20', '50'])
ax.set_xlabel('Ejecta Mass ($M_{\\odot}$)')
ax.set_ylabel('log(Pulsar Magnetic Field) (G)')
cbar.set_label('log(vcomp)', rotation=270, labelpad=20)
plt.savefig('vcomp_p2.pdf')
#plt.show()
