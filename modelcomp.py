import redback 
import pandas as pd
from bilby.core.prior import Uniform, Gaussian, PriorDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from redback.constants import day_to_s
from redback.model_library import all_models_dict

#ZTF20acphdcg/SN2020znr

#import data from csv file
sne = 'ZTF20acphdcg'
data = pd.read_csv('ZTF20acphdcg_atlas_ztf_mosfit.csv')
data = data.sort_values(by='time')

#structure the data
data['band'].iloc[data['band'] == "sdssr"] = 'r'
data['band'].iloc[data['band'] == "sdssg"] = 'g'
data=data[data['band']!='c']
time_d = data['time'].values
magnitude = data['magnitude'].values
magnitude_err = data['e_magnitude'].values
bands = data['band'].values
fd=redback.utils.calc_flux_density_from_ABmag(magnitude).value
fde=redback.utils.calc_flux_density_error(magnitude, magnitude_err, reference_flux = 3631, magnitude_system='AB')

#data = redback.get_data.get_supernova_data_from_open_transient_catalog_data(transient=sne)
#data[data['band']=='i']

#plot to check data
data_mode = 'flux_density'
sn_obj = redback.supernova.Supernova(name=sne, data_mode=data_mode, time_mjd=time_d, flux_density=fd,
                                     flux_density_err=fde, bands=bands, use_phase_model=True)
#ax = sn_obj.plot_data()

#model = 't0_base_model'
model = 'slsn'
function = all_models_dict[model]
time_sim = np.logspace(2, 8, 100)/day_to_s
redshift=0.1

rb_p0=2.34
rb_bp=0.30
rb_mns=1.65
rb_tpb=0.66
rb_mej=1.13
rb_vej=6735
rb_kappa=1.05
rb_kg=0.34
rb_tf=13173
rb_t0=59152

mf_p0=2.80
mf_bp=0.51
mf_mns=1.68
mf_tpb=1.57
mf_mej=21.4
mf_vej=5560
mf_kappa=0.19
mf_kg=2e-3
mf_tf=10000
mf_t0=59149

rb_g=function(time_sim, redshift, p0=rb_p0, bp=rb_bp, mass_ns=rb_mns, theta_pb=rb_tpb, mej=rb_mej, vej=rb_vej, kappa=rb_kappa, kappa_gamma=rb_kg, temperature_floor=rb_tf, frequency=redback.utils.bands_to_frequency('g')[0], output_format='magnitude')
rb_o=function(time_sim, redshift, p0=rb_p0, bp=rb_bp, mass_ns=rb_mns, theta_pb=rb_tpb, mej=rb_mej, vej=rb_vej, kappa=rb_kappa, kappa_gamma=rb_kg, temperature_floor=rb_tf, frequency=redback.utils.bands_to_frequency('o')[0], output_format='magnitude')
rb_r=function(time_sim, redshift, p0=rb_p0, bp=rb_bp, mass_ns=rb_mns, theta_pb=rb_tpb, mej=rb_mej, vej=rb_vej, kappa=rb_kappa, kappa_gamma=rb_kg, temperature_floor=rb_tf, frequency=redback.utils.bands_to_frequency('r')[0], output_format='magnitude')

mf_g=function(time_sim, redshift, p0=mf_p0, bp=mf_bp, mass_ns=mf_mns, theta_pb=mf_tpb, mej=mf_mej, vej=mf_vej, kappa=mf_kappa, kappa_gamma=mf_kg, temperature_floor=mf_tf, frequency=redback.utils.bands_to_frequency('g')[0], output_format='magnitude')
mf_o=function(time_sim, redshift, p0=mf_p0, bp=mf_bp, mass_ns=mf_mns, theta_pb=mf_tpb, mej=mf_mej, vej=mf_vej, kappa=mf_kappa, kappa_gamma=mf_kg, temperature_floor=mf_tf, frequency=redback.utils.bands_to_frequency('o')[0], output_format='magnitude')
mf_r=function(time_sim, redshift, p0=mf_p0, bp=mf_bp, mass_ns=mf_mns, theta_pb=mf_tpb, mej=mf_mej, vej=mf_vej, kappa=mf_kappa, kappa_gamma=mf_kg, temperature_floor=mf_tf, frequency=redback.utils.bands_to_frequency('r')[0], output_format='magnitude')

plt.plot(time_sim, rb_g, 'g-')
plt.plot(time_sim, rb_o, 'b-')
plt.plot(time_sim, rb_r, 'r-')

plt.plot(time_sim, mf_g, 'g:')
plt.plot(time_sim, mf_o, 'b:')
plt.plot(time_sim, mf_r, 'r:')

rind=np.where(bands == 'r')[0]
oind=np.where(bands == 'o')[0]
gind=np.where(bands == 'g')[0]

plt.errorbar(time_d[rind]-rb_t0, magnitude[rind], magnitude_err[rind],linewidth=0,marker='o',c='r')
plt.errorbar(time_d[oind]-rb_t0, magnitude[oind], magnitude_err[oind],linewidth=0,marker='o',c='b')
plt.errorbar(time_d[gind]-rb_t0, magnitude[gind], magnitude_err[gind],linewidth=0,marker='o',c='g')

plt.plot([1,1],[1,1],'k-',label='Redback params')
plt.plot([1,1],[1,1],'k:',label='MOSFiT params')
plt.errorbar([1,1],[1,1],[1,1],linewidth=0,marker='o',c='k',label='Data')

plt.xlabel('Time [days]')
plt.ylabel('Magnitude')
plt.xlim(4,500)
plt.ylim(22,14.8)
plt.legend()
plt.savefig('modelcomp.png')
plt.show()

sne = "SN1998bw" #I, R, V, B data
data = redback.get_data.get_supernova_data_from_open_transient_catalog_data(sne)
time_d = data['time'].values
magnitude = data['magnitude'].values
magnitude_err = data['e_magnitude'].values
bands = data['band'].values

data_mode = 'magnitude'
sn_obj = redback.supernova.Supernova(name=sne, data_mode=data_mode, time_mjd=time_d, magnitude=magnitude,
                                     magnitude_err=magnitude_err, bands=bands, use_phase_model=True)
#ax = sn_obj.plot_data()
model = 'arnett'
function = all_models_dict[model]
time_sim = np.logspace(2, 8, 100)/day_to_s
redshift=0.01

mf_fni=10**(-0.04)
mf_mej=10**(-0.13)
mf_vej=10**4.11
mf_kappa=0.2
mf_kg=0.1
mf_tf=10**(3.71)
mf_t0=50940
kwargs={}
#kwargs['interaction_process']=None

mf_I=function(time_sim, redshift, f_nickel=mf_fni, mej=mf_mej, vej=mf_vej, kappa=mf_kappa, kappa_gamma=mf_kg, frequency=redback.utils.bands_to_frequency('I')[0], output_format='magnitude', temperature_floor=mf_tf)#, , 
mf_R=function(time_sim, redshift, f_nickel=mf_fni, frequency=redback.utils.bands_to_frequency('R')[0], output_format='magnitude', **kwargs)#, mej=mf_mej, vej=mf_vej, kappa=mf_kappa, kappa_gamma=mf_kg, temperature_floor=mf_tf
mf_V=function(time_sim, redshift, f_nickel=mf_fni, frequency=redback.utils.bands_to_frequency('V')[0], output_format='magnitude', **kwargs)#, mej=mf_mej, temperature_floor=mf_tf, vej=mf_vej, kappa=mf_kappa, kappa_gamma=mf_kg
mf_B=function(time_sim, redshift, f_nickel=mf_fni, frequency=redback.utils.bands_to_frequency('B')[0], output_format='magnitude', **kwargs)#, mej=mf_mej, temperature_floor=mf_tf, vej=mf_vej, kappa=mf_kappa, kappa_gamma=mf_kg

plt.plot(time_sim, mf_I, 'r-')
plt.plot(time_sim, mf_R, 'y-')
plt.plot(time_sim, mf_V, 'g-')
plt.plot(time_sim, mf_B, 'b-')

Iind=np.where(bands == 'I')[0]
Rind=np.where(bands == 'R')[0]
Vind=np.where(bands == 'V')[0]
Bind=np.where(bands == 'B')[0]

plt.errorbar(time_d[Iind]-mf_t0, magnitude[Iind], magnitude_err[Iind],linewidth=0,marker='o',c='r')
plt.errorbar(time_d[Rind]-mf_t0, magnitude[Rind], magnitude_err[Rind],linewidth=0,marker='o',c='y')
plt.errorbar(time_d[Vind]-mf_t0, magnitude[Vind], magnitude_err[Vind],linewidth=0,marker='o',c='g')
plt.errorbar(time_d[Bind]-mf_t0, magnitude[Bind], magnitude_err[Bind],linewidth=0,marker='o',c='b')

plt.plot([1,1],[1,1],'k-',label='MOSFiT params')
plt.errorbar([1,1],[1,1],[1,1],linewidth=0,marker='o',c='k',label='Data')

plt.xlabel('Time [days]')
plt.ylabel('Magnitude')
plt.xlim(0,200)
plt.ylim(22,14)
plt.legend()
plt.savefig('modeltest_arnett.png')
plt.show()
