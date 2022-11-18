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

mej=10.0
E_sn=1.0e51
ej_rad=1.0e11
kappa=0.1
n_ism=1.0e-5
nn=3.0
kappa_gamma=0.1
tf=6000
output_format='dynamics_output'
frequency=redback.utils.bands_to_frequency('g')[0]

#L46_small

L046=1.0e46
L048=1.0e48
tsd46_s=1.0e5#/redback.utils.day_to_s
tsd46_m=3.0e5#/redback.utils.day_to_s
tsd46_l=1.0e6#/redback.utils.day_to_s
tsd48_s=1.0e3#/redback.utils.day_to_s
tsd48_m=3.0e3#/redback.utils.day_to_s
tsd48_l=1.0e4#/redback.utils.day_to_s

l46_s=function(time_sim, redshift, mej=mej, E_sn=E_sn, ejecta_radius=ej_rad, kappa=kappa, n_ism=n_ism, l0=L046, tau_sd=tsd46_s, nn=nn, kappa_gamma=kappa_gamma, frequency=frequency, temperature_floor=tf, output_format=output_format) #, p0=p0, bp=bp, mass_ns=mass_ns, theta_pb = theta_pb
l46_m=function(time_sim, redshift, mej=mej, E_sn=E_sn, ejecta_radius=ej_rad, kappa=kappa, n_ism=n_ism, l0=L046, tau_sd=tsd46_m, nn=nn, kappa_gamma=kappa_gamma, frequency=frequency, temperature_floor=tf, output_format=output_format)
l46_l=function(time_sim, redshift, mej=mej, E_sn=E_sn, ejecta_radius=ej_rad, kappa=kappa, n_ism=n_ism, l0=L046, tau_sd=tsd46_l, nn=nn, kappa_gamma=kappa_gamma, frequency=frequency, temperature_floor=tf, output_format=output_format)

l48_s=function(time_sim, redshift, mej=mej, E_sn=E_sn, ejecta_radius=ej_rad, kappa=kappa, n_ism=n_ism, l0=L048, tau_sd=tsd48_s, nn=nn, kappa_gamma=kappa_gamma, frequency=frequency, temperature_floor=tf, output_format=output_format)
l48_m=function(time_sim, redshift, mej=mej, E_sn=E_sn, ejecta_radius=ej_rad, kappa=kappa, n_ism=n_ism, l0=L048, tau_sd=tsd48_m, nn=nn, kappa_gamma=kappa_gamma, frequency=frequency, temperature_floor=tf, output_format=output_format)
l48_l=function(time_sim, redshift, mej=mej, E_sn=E_sn, ejecta_radius=ej_rad, kappa=kappa, n_ism=n_ism, l0=L048, tau_sd=tsd48_l, nn=nn, kappa_gamma=kappa_gamma, frequency=frequency, temperature_floor=tf, output_format=output_format)

plt.close('all')
plt.loglog(time_temp, l46_s.kinetic_energy,label='L46s')
plt.loglog(time_temp, l46_m.kinetic_energy,label='L46m')
plt.loglog(time_temp, l46_l.kinetic_energy,label='L46l')
plt.loglog(time_temp, l48_s.kinetic_energy,label='L48s')
plt.loglog(time_temp, l48_m.kinetic_energy,label='L48m')
plt.loglog(time_temp, l48_l.kinetic_energy,label='L48l')
plt.xlabel("Time (days)")
plt.ylabel("Kinetic energy (erg)")
plt.legend()
plt.show()

plt.close('all')
plt.semilogy(time_temp, l46_s.bolometric_luminosity,'r-',label='L46s')
plt.semilogy(time_temp, l46_m.bolometric_luminosity,'b-',label='L46m')
plt.semilogy(time_temp, l46_l.bolometric_luminosity,'g-',label='L46l')
plt.semilogy(time_temp, l48_s.bolometric_luminosity,'m-',label='L48s')
plt.semilogy(time_temp, l48_m.bolometric_luminosity,'c-',label='L48m')
plt.semilogy(time_temp, l48_l.bolometric_luminosity,'y-',label='L48l')
plt.semilogy([1,1],[1,1],'w-',label=' ')
plt.semilogy(smdat[:,0]/day_to_s, smdat[:,1]+smdat[:,12],'k-',label='SM21 0$^{\\rm o}$')
plt.semilogy(smdat[:,0]/day_to_s, smdat[:,6]+smdat[:,17],'k:',label='SM21 90$^{\\rm o}$')
plt.xlim(-5,200)
plt.ylim(1e38,1e46)
plt.xlabel("Time (days)")
plt.ylabel("Bolometric Luminosity (erg s$^{-1}$)")
plt.legend()
plt.savefig('newmodbolo.pdf')
plt.show()

plt.close('all')
plt.loglog(time_temp, l46_s.tau,'r-',label='L46s')
plt.loglog(time_temp, l46_m.tau,'b-',label='L46m')
plt.loglog(time_temp, l46_l.tau,'g-',label='L46l')
plt.loglog(time_temp, l48_s.tau,'m-',label='L48s')
plt.loglog(time_temp, l48_m.tau,'c-',label='L48m')
plt.loglog(time_temp, l48_l.tau,'y-',label='L48l')
plt.loglog([1e-3,1e10],[1e-3,1e10],'k:')
plt.xlabel("Time (days)")
plt.ylabel("Diffusion Time (days)")
plt.legend()
#plt.show()

plt.close('all')
plt.loglog(time_temp*day_to_s, l46_s.magnetar_luminosity,label='L46s')
plt.loglog(time_temp*day_to_s, l46_m.magnetar_luminosity,label='L46m')
plt.loglog(time_temp*day_to_s, l46_l.magnetar_luminosity,label='L46l')
plt.loglog(time_temp*day_to_s, l48_s.magnetar_luminosity,label='L48s')
plt.loglog(time_temp*day_to_s, l48_m.magnetar_luminosity,label='L48m')
plt.loglog(time_temp*day_to_s, l48_l.magnetar_luminosity,label='L48l')
plt.xlabel("Time (seconds)")
plt.ylabel("Magnetar Luminosity (erg s$^{-1}$)")
plt.legend()
plt.show()

plt.close('all')
plt.loglog(time_temp, l46_s.v_ej,label='L46s')
plt.loglog(time_temp, l46_m.v_ej,label='L46m')
plt.loglog(time_temp, l46_l.v_ej,label='L46l')
plt.loglog(time_temp, l48_s.v_ej,label='L48s')
plt.loglog(time_temp, l48_m.v_ej,label='L48m')
plt.loglog(time_temp, l48_l.v_ej,label='L48l')
plt.xlabel("Time (days)")
plt.ylabel("Ejecta Velocity (km s$^{-1}$)")
plt.legend()
plt.show()

#lofac= (1. + time_sim / tsd46_s) ** ((1. + nn) / (1. - nn))
