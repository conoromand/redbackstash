import redback 
import pandas as pd
from bilby.core.prior import Uniform, Gaussian, PriorDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from redback.constants import day_to_s
from redback.model_library import all_models_dict

smdat=np.loadtxt('lc_L46large_S21.txt')

#model = 'slsn'
model = 'general_magnetar_driven_supernova'
function = all_models_dict[model]
time_sim = np.logspace(2, 7, 1000)/day_to_s
time_temp = np.geomspace(1e-4, 1e9, 1000, endpoint=True)/day_to_s
redshift=0.1

mej=0.05
E_sn=1.0e51
ej_rad=1.0e9
kappa=0.1
n_ism=1.0e-5
nn=3.0
kappa_gamma=0.1
tf=6000
output_format='dynamics_output'
frequency=redback.utils.bands_to_frequency('g')[0]

#L46_small

L0=1.0e50
tsd=1.0e2#/redback.utils.day_to_s

kntest=function(time_sim, redshift, mej=mej, E_sn=E_sn, ejecta_radius=ej_rad, kappa=kappa, n_ism=n_ism, l0=L0, tau_sd=tsd, nn=nn, kappa_gamma=kappa_gamma, frequency=frequency, temperature_floor=tf, output_format=output_format) #, p0=p0, bp=bp, mass_ns=mass_ns, theta_pb = theta_pb

plt.close('all')
plt.loglog(time_temp, kntest.kinetic_energy)
plt.xlabel("Time (days)")
plt.ylabel("Kinetic energy (erg)")
plt.show()

plt.close('all')
plt.semilogy(time_temp, kntest.bolometric_luminosity)
plt.xlim(-5,20)
plt.ylim(1e40,1e46)
plt.xlabel("Time (days)")
plt.ylabel("Bolometric Luminosity (erg s$^{-1}$)")
plt.show()

plt.close('all')
plt.loglog(time_temp, kntest.tau,'r-')
plt.loglog([1e-3,1e10],[1e-3,1e10],'k:')
plt.xlabel("Time (days)")
plt.ylabel("Diffusion Time (days)")
#plt.show()

plt.close('all')
plt.loglog(time_temp*day_to_s, kntest.magnetar_luminosity)
plt.xlabel("Time (seconds)")
plt.ylabel("Magnetar Luminosity (erg s$^{-1}$)")
#plt.show()

plt.close('all')
plt.loglog(time_temp, kntest.v_ej)
plt.xlabel("Time (days)")
plt.ylabel("Ejecta Velocity (km s$^{-1}$)")
plt.show()

#lofac= (1. + time_sim / tsd46_s) ** ((1. + nn) / (1. - nn))
