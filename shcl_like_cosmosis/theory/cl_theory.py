from cosmosis.datablock import DataBlock, names, SectionOptions
import pyccl as ccl
import sacc
import numpy as np
import collections


class CLCalculator:
    def __init__(self, options):
        self.options = options
        self.transfer_function  = options.get_string('transfer_function', 'boltzmann_camb')
        self.matter_pk  = options.get_string('matter_pk', 'halofit')
        self.baryons_pk  = options.get_string('baryons_pk', 'nobaryons')
        self.run_boltzmann = options.get_bool('run_boltzmann', True)
        sacc_file  = options.get_string('sacc_file', '')
        read_sacc  = options.get_bool('read_sacc', True)

        self.tracer_names = collections.defaultdict(set)
        self.calculations = {}
        self.ell = collections.defaultdict(dict)

        if sacc_file and read_sacc:
            self.setup_from_sacc(options, sacc_file)
        else:
            self.setup_from_options(options)




    def setup_from_sacc(self, options, sacc_file):
        sacc_data = sacc.Sacc.load_fits(sacc_file)

        # shear-shear - could do a loop over data types here
        data_type = 'galaxy_shear_cl_ee' # because we are lucky enough not to be CMB people
        self.calculations[data_type] = sacc_data.get_tracer_combinations(data_type)

        # assume band powers - seems to be very likely here
        # for C_ell.  Note that for xi bandpowers can be used
        # when doing the transform from C_ell instead.
        ell_per_decade = options.get_int('ell_per_decade', 30)

        # For now use a global value of ell for everything.
        # We could easily modify this per data-type, and in
        # fact will probably have to set per-data-type limber
        # ranges.
        # Also note that we should be cuttng 
        windows = sacc_data.get_tag('window') 
        ell_min = np.min([win.values.min() for win in windows])
        ell_max = np.max([win.values.max() for win in windows])
        ell_min = max(ell_min, 2)
        n_ell = int(np.log10(ell_max / ell_min)) * ell_per_decade
        ell = np.unique(np.geomspace(ell_min, ell_max, n_ell).astype(int))
        n_ell = len(ell)
        print(f"Calculating {data_type} at {n_ell} ell values from {ell_min} .. {ell_max}")

        # pull out the names of our tracers to save doing it every time
        for t1, t2 in self.calculations[data_type]:
            self.tracer_names[data_type].add(t1)
            self.tracer_names[data_type].add(t2)

            # Shared ell value for everything right now
            self.ell[data_type][t1, t2] = ell


    def setup_from_options(options):

        nz_name = options['options_section', 'nz_name']

        ell_min = options['options_section', 'ell_min']
        ell_max = options['options_section', 'ell_max']
        n_ell = options['options_section', 'n_ell']
        ell = np.geomspace(ell_min, ell_max, n_ell)

        data_type = 'galaxy_shear_cl_ee' # Much, much better

        pairs = options[options_section, 'shear_shear']
        pairs = [pair.split('-') for pair in pairs.split()]    

        self.calculations[data_type] = pairs

        # pull out the names of our tracers to save doing it every time
        for t1, t2 in self.calculations[data_type]:
            self.tracer_names[data_type].add(t1)
            self.tracer_names[data_type].add(t2)

            self.ell[data_type][t1, t2] = ell

    def execute(self, block):
        inputs = self.read_inputs(block)
        results = self.run(inputs)
        self.write_outputs(block, results)
        return 0


    def read_inputs(self, block):

        # Translate into CCL parameters
        h = block[names.cosmological_parameters, 'h0']
        Omega_c = block[names.cosmological_parameters, 'Omega_c']
        Omega_b = block[names.cosmological_parameters, 'Omega_b']
        n_s = block[names.cosmological_parameters, 'n_s']
        A_s = block[names.cosmological_parameters, 'A_s']
        m_nu = block[names.cosmological_parameters, 'm_nu']

        nz = {}
        ia = {}
        for dt, tracer_names in self.tracer_names.items():
            for name in tracer_names:
                if name in nz:
                    continue
                sec, key = name.split("_", 1)
                nz[name] = (
                        block[f'nz_{sec}', f'z_{key}'],
                        block[f'nz_{sec}', f'bin_{key}']
                )

                ia[name] = (
                    nz[name][0], # z assumed same as n(z)
                    block[f'ia_{sec}', f'ia_{key}']
                )


        # In this case CCL runs everything.
        # Otherwise we assume earlier stages
        # have run the Boltzmann code and re-start
        # CCL from there.
        if self.run_boltzmann:
            return locals()

        # comoving distance
        z_bg = block[names.distances, 'z']
        distance = block[names.distances, 'D_M']

        # H(z), in CAMB's units of Mpc^-1, but that doesn't matter
        # because we are about to normalize
        E_of_z = block[names.distances, 'H']
        E_of_z /= E_of_z[0]

        # Generate cosmology and populate background
        a_bg = 1. / (1 + z_bg)

        z_lin, k_lin, Pk_lin = block.get_grid('matter_power_lin', 'z', 'k_h', 'P_k')
        z_nl, k_nl, Pk_nl = block.get_grid('matter_power_nl', 'z', 'k_h', 'P_k')
        a_lin = 1. / (1 + z_lin)
        a_nl = 1. / (1 + z_nl)

        # only support shared z here
        z_lin = np.flip(z_lin)
        Pk_lin = np.flip(Pk_lin, axis=0)
        a_lin = np.flip(a_lin)

        z_nl = np.flip(z_nl)
        Pk_nl = np.flip(Pk_nl, axis=0)
        a_nl = np.flip(a_nl)

        E_of_z = np.flip(E_of_z)
        z_bg = np.flip(z_bg)
        a_bg = np.flip(a_bg)
        distance = np.flip(distance)

        return locals() # I'm tired, okay, it's the night after
                        # the 2020 US election


    def run(self, inputs):
        cosmo = ccl.Cosmology(Omega_c=inputs["Omega_c"],
                              Omega_b=inputs["Omega_b"],
                              h=inputs["h"],
                              n_s=inputs["n_s"],
                              A_s=inputs["A_s"],
                              T_CMB=2.7255,
                              m_nu=inputs["m_nu"],
                              transfer_function=self.transfer_function,
                              matter_power_spectrum=self.matter_pk,
                              baryons_power_spectrum=self.baryons_pk)

        if not self.run_boltzmann:
            # Ingest pre-calculated distances
            cosmo._set_background_from_arrays(a_array=inputs["a_bg"],
                                              chi_array=inputs["distance"],
                                              hoh0_array=inputs["E_of_z"])

            # Ingest pre-calculated P(k)
            cosmo._set_linear_power_from_arrays(inputs["a_lin"], 
                inputs["k_lin"], inputs["Pk_lin"])
            cosmo._set_nonlin_power_from_arrays(inputs["a_nl"], 
                inputs["k_nl"], inputs["Pk_nl"])

        data_type = 'galaxy_shear_cl_ee'


        nz_info = inputs['nz']
        ia_info = inputs['ia']
        tracers = {}
        for b in self.tracer_names[data_type]:
            tracers[data_type, b] = ccl.WeakLensingTracer(
                                                cosmo, 
                                                nz_info[b],
                                                ia_bias=ia_info[b]
                                    )

        results = {}
        results[data_type] = {}
        for bin1, bin2 in self.calculations[data_type]:
            ell = self.ell[data_type][bin1, bin2]
            T1 = tracers[data_type, bin1]
            T2 = tracers[data_type, bin2]
            cl = ccl.angular_cl(cosmo, T1, T2, ell)
            results[data_type][bin1, bin2] = cl

        return results

    def write_outputs(self, block, results):
        data_type = 'galaxy_shear_cl_ee'
        for data_type, bin_results in results.items():
            for (b1, b2), cl in bin_results.items():
                block[data_type, f'bin_{b1}_{b2}'] = cl
                block[data_type, f'ell_{b1}_{b2}'] = self.ell[data_type][b1, b2]



def execute(block: DataBlock, config: CLCalculator) -> int:
    # rename for clarity
    calculator = config
    return calculator.execute(block)

# CosmoSIS setup functions can return any object.
def setup(options: DataBlock) -> CLCalculator:
    options = SectionOptions(options)
    return CLCalculator(options)

