from cosmosis.datablock import DataBlock, names, option_section
from cosmosis.gaussian_likelihood import GaussianLikelihood

import sacc
import numpy as np
from spec_tools import SpectrumInterpolator

class ClLikelihood(GaussianLikelihood):
    like_name = "cl"

    def __init__(self, options):

        # Load data and read the data types we want to use
        # in this analysis
        self.sacc_data = sacc.Sacc.load_fits(options['sacc_file'])
        self.data_types = options['data_types'].split()

        # Cut down the obs data to our required data types.
        # Can do additional cuts here, e.g. on ell ranges
        good = []
        for data_type in self.data_types:
            good.append(self.sacc_data.indices(data_type))
        self.sacc_data.keep_indices(np.sort(np.concatenate(good)))    

        # This calls the other methods below
        super().__init__(options)

    def build_data(self):
        # add cuts here
        data = self.sacc_data

        # generate the main data vector and auxiliary info
        mu = data.get_mean()

        # Get the global info telling use
        # what we need to load
        self.bins = []
        for dt in data.get_data_types():
            for t1, t2 in data.get_tracer_combinations(dt):
                self.bins.append((dt, t1, t2))

        # Get the per-data-point information we will loop through
        self.tracer1 = np.array([d.tracers[0] for d in data.data])
        self.tracer2 = np.array([d.tracers[1] for d in data.data])
        self.ell = np.array([d.get_tag('ell') for d in data.data])
        self.windows = [d.get_tag('window') for d in data.data]
        self.window_index = [d.get_tag('window_ind') for d in data.data]
        self.data_point_types = [d.data_type for d in data.data]

        # The None is for supporting other cases
        # where we have a simple x-y relation
        return None, mu

    def build_covariance(self):
        return self.sacc_data.covariance.dense

    def build_inverse_covariance(self):
        return self.sacc_data.covariance.inverse

    def extract_theory_points(self, block):
        N = self.data_y.size
        x = np.zeros(N)

        # Read all the theory values
        theory = {}
        for (dt, t1, t2) in self.bins:
            # Extract data from CosmoSIS data block
            ell = block[dt, f'ell_{t1}_{t2}']
            cl = block[dt, f'bin_{t1}_{t2}']
            # This interpolator tries to do logs if cl is all
            # positive but otherwise falls back to linear
            theory[dt, t1, t2] = SpectrumInterpolator(ell, cl)

        # For each data point find the corresponding theory
        for i in range(N):
            # Pull out the relevant theory info from the dict above
            # I could do all these with an enumerate/zip but it was
            # getting a bit long
            t1 = self.tracer1[i]
            t2 = self.tracer2[i]
            ell_obs = self.ell[i]
            dt = self.data_point_types[i]
            win = self.windows[i]
            win_index = self.window_index[i]
            # get the interpolator object above
            interpolator = theory[dt, t1, t2]
            interpolated_cl_theory = interpolator(win.values)

            # Assumes the we
            x[i] = (win.weight[:, win_index] @ interpolated_cl_theory)


        # It's also handy to save this auxiliary information
        block[names.data_vector, self.like_name + "_angle"] = self.ell
        # Need to update cosmosis to save vectors of strings
        #block[names.data_vector, self.like_name + "_bin1"] = self.tracer1
        #block[names.data_vector, self.like_name + "_bin2"] = self.tracer2

        return x


setup, execute, cleanup = ClLikelihood.build_module()
