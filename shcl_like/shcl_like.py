import pyccl as ccl
import numpy as np
from cobaya.likelihood import Likelihood
from cobaya.log import LoggedError
from scipy.interpolate import interp1d


class ShClLike(Likelihood):
    # All parameters starting with this will be
    # identified as belonging to this stage.
    input_params_prefix: str = ""
    # Input sacc file
    input_file: str = ""
    # IA model name. Currently all of these are
    # just flags, but we could turn them into
    # homogeneous systematic classes.
    ia_model: str = "IANone"
    # N(z) model name
    nz_model: str = "NzNone"
    # P(k) model name
    pk_model: str = "PkDefault"
    # List of bin names
    bins: list = []
    # List of default settings (currently only scale cuts)
    defaults: dict = {}
    # List of two-point functions that make up the data vector
    twopoints: list = []

    def initialize(self):
        # Read SACC file
        self._read_data()
        # Ell sampling for interpolation
        self._get_ell_sampling()

    def _read_data(self):
        # Reads sacc file
        # Selects relevant data.
        # Applies scale cuts
        # Reads tracer metadata (N(z))
        # Reads covariance
        import sacc
        s = sacc.Sacc.load_fits(self.input_file)
        self.bin_properties = {}
        for b in self.bins:
            if b['name'] not in s.tracers:
                raise LoggedError(self.log, "Unknown tracer %s" % b['name'])
            t = s.tracers[b['name']]
            self.bin_properties[b['name']] = {'z_fid': t.z,
                                              'nz_fid': t.nz}

        indices = []
        for cl in self.twopoints:
            lmin = cl.get('lmin', self.defaults.get('lmin', 2))
            lmax = cl.get('lmax', self.defaults.get('lmax', 1E30))
            ind = s.indices('cl_ee', (cl['bins'][0], cl['bins'][1]),
                            ell__gt=lmin, ell__lt=lmax)
            indices += list(ind)
        s.keep_indices(np.array(indices))

        indices = []
        self.cl_meta = []
        id_sofar = 0
        self.used_tracers = []
        self.l_min_sample = 1E30
        self.l_max_sample = -1E30
        for cl in self.twopoints:
            l, c_ell, cov, ind = s.get_ell_cl('cl_ee',
                                              cl['bins'][0],
                                              cl['bins'][1],
                                              return_cov=True,
                                              return_ind=True)
            if c_ell.size > 0:
                if cl['bins'][0] not in self.used_tracers:
                    self.used_tracers.append(cl['bins'][0])
                if cl['bins'][1] not in self.used_tracers:
                    self.used_tracers.append(cl['bins'][1])

            bpw = s.get_bandpower_windows(ind)
            if np.amin(bpw.values) < self.l_min_sample:
                self.l_min_sample = np.amin(bpw.values)
            if np.amax(bpw.values) > self.l_max_sample:
                self.l_max_sample = np.amax(bpw.values)

            self.cl_meta.append({'bin_1': cl['bins'][0],
                                 'bin_2': cl['bins'][1],
                                 'l_eff': l,
                                 'cl': c_ell,
                                 'cov': cov,
                                 'inds': (id_sofar +
                                          np.arange(c_ell.size,
                                                    dtype=int)),
                                 'l_bpw': bpw.values,
                                 'w_bpw': bpw.weight.T})
            indices += list(ind)
            id_sofar += c_ell.size
        indices = np.array(indices)
        self.data_vec = s.mean[indices]
        self.cov = s.covariance.covmat[indices][:, indices]
        self.inv_cov = np.linalg.inv(self.cov)
        self.ndata = len(self.data_vec)

    def _get_ell_sampling(self, nl_per_decade=30):
        # Selects ell sampling.
        # Ell max/min are set by the bandpower window ells.
        # It currently uses simple log-spacing.
        # nl_per_decade is currently fixed at 30
        if self.l_min_sample == 0:
            l_min_sample_here = 2
        else:
            l_min_sample_here = self.l_min_sample
        nl_sample = int(np.log10(self.l_max_sample / l_min_sample_here) *
                        nl_per_decade)
        l_sample = np.unique(np.geomspace(l_min_sample_here,
                                          self.l_max_sample+1,
                                          nl_sample).astype(int)).astype(float)

        if self.l_min_sample == 0:
            self.l_sample = np.concatenate((np.array([0.]), l_sample))
        else:
            self.l_sample = l_sample

    def _eval_interp_cl(self, cl_in, l_bpw, w_bpw):
        # Interpolates spectrum and evaluates it at bandpower window
        # ell values.
        f = interp1d(self.l_sample, cl_in)
        cl_unbinned = f(l_bpw)
        cl_binned = np.dot(w_bpw, cl_unbinned)
        return cl_binned

    def _get_nz(self, cosmo, name, **pars):
        # Get an N(z) for tracer with name `name`
        z = self.bin_properties[name]['z_fid']
        nz = self.bin_properties[name]['nz_fid']
        if self.nz_model == 'NzShift':
            z = z + pars[self.input_params_prefix + '_' + name + '_dz']
            msk = z >= 0
            z = z[msk]
            nz = nz[msk]
        elif self.nz_model != 'NzNone':
            raise LoggedError(self.log, "Unknown Nz model %s" % self.nz_model)
        return (z, nz)

    def _get_ia_bias(self, cosmo, name, **pars):
        # Get an IA amplitude for tracer with name `name`
        if self.ia_model == 'IANone':
            return None
        else:
            z = self.bin_properties[name]['z_fid']
            if self.ia_model == 'IAPerBin':
                A = pars[self.input_params_prefix + '_' + name + '_A_IA']
                A_IA = np.ones_like(z) * A
            elif self.ia_model == 'IADESY1':
                A0 = pars[self.input_params_prefix + '_A_IA']
                eta = pars[self.input_params_prefix + '_eta_IA']
                A_IA = A0 * ((1+z)/1.62)**eta
            else:
                raise LoggedError(self.log, "Unknown IA model %s" %
                                  self.ia_model)
            return (z, A_IA)

    def _get_tracer(self, cosmo, name, **pars):
        # Get CCL tracer for tracer with name `name`
        nz = self._get_nz(cosmo, name, **pars)
        ia = self._get_ia_bias(cosmo, name, **pars)
        t = ccl.WeakLensingTracer(cosmo, nz, ia_bias=ia)
        return t

    def _get_pk(self, cosmo):
        # Get P(k) to integrate over
        if self.pk_model == 'PkDefault':
            return None
        elif self.pk_model == 'PkHModel':
            mdef = ccl.halos.MassDef200c()
            hmf = ccl.halos.MassFuncTinker08(cosmo, mass_def=mdef)
            hbf = ccl.halos.HaloBiasTinker10(cosmo, mass_def=mdef)
            hmc = ccl.halos.HMCalculator(cosmo, hmf, hbf, mdef)
            cM = ccl.halos.ConcentrationDuffy08(mdef=mdef)
            prof = ccl.halos.HaloProfileNFW(cM)
            lk_s = np.log(np.geomspace(1E-4, 1E2, 256))
            a_s = 1./(1+np.linspace(0., 3., 20)[::-1])
            pk2d = ccl.halos.halomod_Pk2D(cosmo, hmc, prof,
                                          lk_arr=lk_s, a_arr=a_s,
                                          normprof1=True)
            return pk2d
        else:
            raise LoggedError("Unknown power spectrum model %s" %
                              self.pk_model)

    def _get_cl_wl(self, cosmo, pk, **pars):
        # Compute all C_ells without multiplicative bias
        trs = {}
        for tn in self.used_tracers:
            trs[tn] = self._get_tracer(cosmo, tn, **pars)
        cls = []
        for clm in self.cl_meta:
            cl = ccl.angular_cl(cosmo, trs[clm['bin_1']], trs[clm['bin_2']],
                                self.l_sample, p_of_k_a=pk)
            clb = self._eval_interp_cl(cl, clm['l_bpw'], clm['w_bpw'])
            cls.append(clb)
        return cls

    def _get_theory(self, **pars):
        # Compute theory vector
        res = self.provider.get_CCL()
        cosmo = res['cosmo']
        pk = res['pk']
        cls = self._get_cl_wl(cosmo, pk, **pars)
        cl_out = np.zeros(self.ndata)
        for clm, cl in zip(self.cl_meta, cls):
            m1 = pars[self.input_params_prefix + '_' + clm['bin_1'] + '_m']
            m2 = pars[self.input_params_prefix + '_' + clm['bin_2'] + '_m']
            prefac = (1+m1) * (1+m2)
            cl_out[clm['inds']] = cl * prefac
        return cl_out

    def get_requirements(self):
        # By selecting `self._get_pk` as a `method` of CCL here,
        # we make sure that this function is only run when the
        # cosmological parameters vary.
        return {'CCL': {'methods': {'pk': self._get_pk}}}

    def logp(self, **pars):
        """
        Simple Gaussian likelihood.
        """
        t = self._get_theory(**pars)
        r = t - self.data_vec
        chi2 = np.dot(r, self.inv_cov.dot(r))
        return -0.5*chi2
