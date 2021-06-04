import numpy as np
from scipy.interpolate import interp1d

class SpectrumInterpolator:
    """
    Class for interpolating spectra.
    Extracted from original CosmoSIS 2pt code with some
    other stuff deleted.  Also renamed.
    """
    def __init__(self, angle, spec, bounds_error=False):
        assert np.all(angle>=0)
        #Check if the angle array starts with zero - 
        #this will affect how we interpolate
        starts_with_zero=False
        self.spec0 = 0.

        if angle[0]<1.e-9:
            #if the first angle value is 0,
            #record the first spec value as self.spec0
            #and then set angle and spec arrays to 
            #skip the first element.
            starts_with_zero=True
            self.spec0 = spec[0]
            angle, spec = angle[1:], spec[1:]

        if np.all(spec > 0):
            self.interp_func = interp1d(np.log(angle), np.log(
                spec), bounds_error=bounds_error, fill_value=-np.inf)
            self.interp_type = 'loglog'
            self.x_func = np.log
            self.y_func = np.exp

        elif np.all(spec < 0):
            self.interp_func = interp1d(
                np.log(angle), np.log(-spec), bounds_error=bounds_error, fill_value=-np.inf)
            self.interp_type = 'minus_loglog'
            self.x_func = np.log
            self.y_func = lambda y: -np.exp(y)
        else:
            self.interp_func = interp1d(
                np.log(angle), spec, bounds_error=bounds_error, fill_value=0.)
            self.interp_type = "log_ang"
            self.x_func = np.log
            self.y_func = lambda y: y

    def __call__(self, angle):
        non_zero = angle>1.e-12
        interp_vals = self.x_func(angle)
        try:
            spec = self.y_func( self.interp_func(interp_vals) )
        except ValueError:
            interp_vals[0] *= 1+1.e-9
            interp_vals[-1] *= 1-1.e-9
            spec = self.y_func( self.interp_func(interp_vals) )
        return np.where(non_zero, spec, self.spec0)
