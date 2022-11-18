import redback 
import pandas as pd
from bilby.core.prior import Uniform, Gaussian, PriorDict

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

#Time to do inference
sampler = 'dynesty'

model = 't0_base_model'
base_model = 'slsn'
priors = redback.priors.get_priors(model=model)
priors.update(redback.priors.get_priors(model=base_model))
priors['redshift'] = 1.0e-1
priors['t0'] = Gaussian(data['time'].iloc[0], sigma=10, name='t0', latex_label=r'$T_{\rm{0}}$')
#priors['av'] = Uniform(minimum=0, maximum=5, name='av', latex_label=r'$A_{\rm V}$')
model_kwargs = dict(frequency=redback.utils.bands_to_frequency(bands), output_format='flux_density', base_model=base_model)

# returns a supernova result object

result = redback.fit_model(transient=sn_obj, model=model, sampler=sampler, model_kwargs=model_kwargs,
                           prior=priors, sample='rwalk', nlive=1000, resume=False)
result.plot_corner()
