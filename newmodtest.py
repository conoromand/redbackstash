_sed = kwargs.get("sed", sed.Blackbody)
sed_1 = _sed(time=time, luminosity=output.bolometric_luminosity, temperature=temp,
                r_photosphere=rad, frequency=frequency, luminosity_distance=dl)
sed_bb=sed_1                
plt.loglog(time, sed_bb.flux_density.to(uu.mJy).value*1e3,'k')
plt.loglog(time, sed_bb.r_photosphere/1e14,'r')
plt.loglog(time, sed_bb.temperature/1e4,'b')

_sed = kwargs.get("sed", sed.CutoffBlackbody)
sed_1 = _sed(time=time, luminosity=output.bolometric_luminosity, temperature=temp,
                r_photosphere=rad, frequency=frequency, luminosity_distance=dl,
                cutoff_wavelength=cutoff_wavelength)
sed_co=sed_1                
plt.loglog(time, sed_co.flux_density.to(uu.mJy).value*1e3,'k:')
plt.loglog(time, sed_co.r_photosphere/1e14,'r:')
plt.loglog(time, sed_co.temperature/1e4,'b:')

plt.show()


#6000-9000
lam=np.linspace(6000,9000,100)
nus=redback.utils.lambda_to_nu(lam)
time_sim = np.logspace(6, 6, 100)/day_to_s
time_sim = np.logspace(2, 8, 100)/day_to_s

#model = 't0_base_model'
model = 'arnett'
function = all_models_dict[model]
redshift=0.1

mej=2
vej=6000
kappa=0.1
kg=0.1
tf=3000
f_ni=0.5

ar_fd=function(time_sim, redshift, f_nickel=f_ni, mej=mej, vej=vej, kappa=kappa, kappa_gamma=kg, temperature_floor=tf, frequency=1e14, output_format='flux_density')
