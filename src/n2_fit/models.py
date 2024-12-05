from lmfit.models import LinearModel, SkewedVoigtModel, Model


class Models:

    def config_SkewedVoigtModel(self, model_name, prefix_, value_center, value_center_min, 
                                value_center_max, value_sigma, value_sigma_min, 
                                value_sigma_max, value_amp, value_amp_min, value_amp_max, value_gamma, 
                                value_skew, pars, vary_gamma=False):
        """
        Configure a SkewdVoigtModel to be used in the fit when fix_parameters = False

        Parameters
        ----------
        model_name : string
             the name of the model
        prefix_    : string
             the name of the skewdvoigt peak
        value_center: float
             the center of the peak
        value_center_min: float
             the lower bound for the center of the peak for the fit routine
        value_center_max: float
             the upper bound for the center of the peak for the fit routine
        value_sigma: float
             the sigma of the gaussian component of the voigt peak
        value_sigma_min: float
             the lower bound for the sigma for the fit routine
        value_sigma_max: float
             the upper bound for the sigma for the fit routine
        value_amp: float
             the value for the amplitude of the peak
        value_amp_min: float
             the lower bound for the amplitude for the fit routine
        value_gamma: float
             the gamma value for the loretzian component of the voigt peak. This parameter is not fitted.
        value_skew: float
             the skew parameter for the voigt peak (defines peak asimmetry)
        pars: lmfit parameter class of a model


        Return
        --------
        x,y : np.array
            two arrays without negative values
        """
        value_amp_min = value_amp_min if value_amp_min>=0 else 0
        value_amp = value_amp if value_amp>=0 else 0
        locals()[model_name] = SkewedVoigtModel(prefix=prefix_)   
        return_model_name = locals()[model_name]

        pars.update(getattr(return_model_name,'make_params')())

        pars[''.join((prefix_, 'center'))].set(     value=value_center, min=value_center_min, max=value_center_max                    )
        pars[''.join((prefix_, 'sigma'))].set(      value=value_sigma,  min=value_sigma_min,  max=value_sigma_max                     )
        pars[''.join((prefix_, 'amplitude'))].set(  value=value_amp, min=value_amp_min, max = value_amp_max  )
        pars[''.join((prefix_, 'gamma'))].set(      value=value_gamma,  vary=vary_gamma, expr='')
        pars[''.join((prefix_, 'skew'))].set(       value=value_skew                                                                  )
        return return_model_name