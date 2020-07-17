import numpy as np
import pyccl as ccl
from typing import Sequence, Union
from cobaya.theory import Theory


class CLCCL(Theory):
    kmax: float = 0
    z_pk: Union[Sequence, np.ndarray] = []
    z_bg: Union[Sequence, np.ndarray] = []
    transfer_function: str = 'boltzmann_camb'
    matter_power_spectrum: str = 'halofit'
    baryons_power_spectrum: str = 'nobaryons'
    external_nonlin_pk: bool = True

    _default_z_pk_sampling = np.linspace(0, 5, 100)
    _default_z_bg_sampling = np.concatenate((np.linspace(0, 10, 100),
                                             np.geomspace(10, 1500, 50)))

    def initialize(self):
        self._var_pairs = set()
        self._required_results = {}

    def get_requirements(self):
        return {'omch2', 'ombh2', 'ns', 'As', 'mnu'}

    def must_provide(self, **requirements):
        if 'CCL' not in requirements:
            return {}
        options = requirements.get('CCL') or {}
        if 'methods' in options:
            self._required_results.update(options['methods'])

        self.kmax = max(self.kmax, options.get('kmax', self.kmax))
        self.z_pk = np.unique(np.concatenate(
            (np.atleast_1d(options.get("z_pk", self._default_z_pk_sampling)),
             np.atleast_1d(self.z_pk))))
        self.z_bg = np.unique(np.concatenate(
            (np.atleast_1d(options.get("z_bg", self._default_z_bg_sampling)),
             np.atleast_1d(self.z_bg))))

        needs = {}

        if self.kmax:
            self.external_nonlin_pk = self.external_nonlin_pk or options.get('external_nonlin_pk',
                                                                             False)
            self._var_pairs.update(
                set((x, y) for x, y in
                    options.get('vars_pairs', [('delta_tot', 'delta_tot')])))

            needs['Pk_grid'] = {
                'vars_pairs': self._var_pairs or [('delta_tot', 'delta_tot')],
                'nonlinear': (True, False) if self.external_nonlin_pk else False,
                'z': self.z_pk,
                'k_max': self.kmax}

        needs['Hubble'] = {'z': self.z_bg}
        needs['comoving_radial_distance'] = {'z': self.z_bg}

        assert len(self._var_pairs) < 2, "CCL doesn't support other Pks yet"
        return needs

    def get_can_provide_params(self):
        return ['sigma8']

    def get_can_support_params(self):
        return []

    def calculate(self, state, want_derived=True, **params_values_dict):
        distance = self.provider.get_comoving_radial_distance(self.z_bg)
        hubble_z = self.provider.get_Hubble(self.z_bg)
        H0 = hubble_z[0]
        E_of_z = hubble_z / H0
        distance = np.flip(distance)
        E_of_z = np.flip(E_of_z)

        h = H0 * 0.01
        Omega_c = self.provider.get_param('omch2') / h**2
        Omega_b = self.provider.get_param('ombh2') / h**2

        a = 1. / (1+self.z_bg[::-1])
        cosmo = ccl.Cosmology(Omega_c=Omega_c, Omega_b=Omega_b, h=h,
                              n_s=self.provider.get_param('ns'),
                              A_s=self.provider.get_param('As'),
                              T_CMB=2.7255,
                              m_nu=self.provider.get_param('mnu'),
                              transfer_function=self.transfer_function,
                              matter_power_spectrum=self.matter_power_spectrum,
                              baryons_power_spectrum=self.baryons_power_spectrum)
        cosmo._set_background_from_arrays(a_array=a,
                                          chi_array=distance,
                                          hoh0_array=E_of_z)

        if self.kmax:
            for pair in self._var_pairs:
                k, z, Pk_lin = self.provider.get_Pk_grid(var_pair=pair, nonlinear=False)
                Pk_lin = np.flip(Pk_lin, axis=0)
                a = 1./(1+np.flip(z))
                cosmo._set_linear_power_from_arrays(a, k, Pk_lin)

                if self.external_nonlin_pk:
                    k, z, Pk_nl = self.provider.get_Pk_grid(var_pair=pair, nonlinear=True)
                    Pk_nl = np.flip(Pk_nl, axis=0)
                    a = 1./(1+np.flip(z))
                    cosmo._set_nonlin_power_from_arrays(a, k, Pk_nl)

        state['CCL'] = {'cosmo': cosmo}
        state['sigma8'] = ccl.sigma8(cosmo)
        for req_res, method in self._required_results.items():
            state['CCL'][req_res] = method(cosmo)

    def get_CCL(self):
        return self._current_state['CCL']
