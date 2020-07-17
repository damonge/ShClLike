import numpy as np
import pyccl as ccl
from cobaya.theory import Theory


class CCL(Theory):
    transfer_function: str = 'boltzmann_camb'
    matter_power_spectrum: str = 'halofit'
    baryons_power_spectrum: str = 'nobaryons'
    params = {'Omega_c': None,
              'Omega_b': None,
              'h': None,
              'n_s': None,
              'A_sE9': None,
              'm_nu': None}

    def initialize(self):
        self._required_results = {}

    def get_requirements(self):
        return {}

    def must_provide(self, **requirements):
        if 'CCL' not in requirements:
            return {}
        options = requirements.get('CCL') or {}
        if 'methods' in options:
            self._required_results.update(options['methods'])

        return {}

    def get_can_provide_params(self):
        return ['sigma8']

    def get_can_support_params(self):
        return []

    def calculate(self, state, want_derived=True, **params_values_dict):
        cosmo = ccl.Cosmology(Omega_c=self.provider.get_param('Omega_c'),
                              Omega_b=self.provider.get_param('Omega_b'),
                              h=self.provider.get_param('h'),
                              n_s=self.provider.get_param('n_s'),
                              A_s=self.provider.get_param('A_sE9')*1E-9,
                              T_CMB=2.7255,
                              m_nu=self.provider.get_param('m_nu'),
                              transfer_function=self.transfer_function,
                              matter_power_spectrum=self.matter_power_spectrum,
                              baryons_power_spectrum=self.baryons_power_spectrum)

        state['CCL'] = {'cosmo': cosmo}
        state['sigma8'] = ccl.sigma8(cosmo)
        for req_res, method in self._required_results.items():
            state['CCL'][req_res] = method(cosmo)

    def get_CCL(self):
        return self._current_state['CCL']
