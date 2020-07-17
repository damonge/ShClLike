import numpy as np
import sacc
import pyccl as ccl
import matplotlib.pyplot as plt

ndens_arcmin = 3.
sigma_gamma = 0.28

# Bandpowers
d_ell = 30
n_bpw = 60
ells = np.arange(3002)
l_eff = (np.arange(n_bpw) + 0.5)*d_ell
n_ell = len(ells)

# Cosmology
cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.67, n_s=0.96, A_s=2E-9, T_CMB=2.7255)

# N(z)s
z_centers = [0.4, 0.7, 1.0, 1.3]
n_bins = len(z_centers)
z = np.linspace(0., 2., 512)
nzs = np.array([np.exp(-0.5*((z-zc)/0.1)**2) for zc in z_centers])
trs = [ccl.WeakLensingTracer(cosmo, (z, n)) for n in nzs]

# C_ells
n_ells = np.ones(n_ell) * sigma_gamma**2 / (ndens_arcmin * (180 * 60 / np.pi)**2)
c_ells_unbinned = np.zeros([n_bins, n_bins, n_ell])
n_ells_unbinned = np.zeros([n_bins, n_bins, n_ell])
for i1, t1 in enumerate(trs):
    for i2, t2 in enumerate(trs):
        c_ells_unbinned[i1, i2, :] = ccl.angular_cl(cosmo, t1, t2, ells)
        if i1 == i2:
            n_ells_unbinned[i1, i2, :] = n_ells

# Bandpower windows
n_ell_large = 3001
ells_large = np.arange(n_ell_large)
window_single = np.zeros([n_bpw, n_ell])
for i in range(n_bpw):
    window_single[i, i * d_ell : (i + 1) * d_ell] = 1./d_ell

c_ells_binned = np.sum(c_ells_unbinned[:, :, None, :] * window_single[None, None, :, :], axis=-1)
n_ells_binned = np.sum(n_ells_unbinned[:, :, None, :] * window_single[None, None, :, :], axis=-1)

# Covariance
fsky = 0.1
n_cross = (n_bins * (n_bins + 1)) // 2
covar = np.zeros([n_cross, n_bpw, n_cross, n_bpw])
id_i = 0
for i1 in range(n_bins):
    for i2 in range(i1, n_bins):
        id_j = 0
        for j1 in range(n_bins):
            for j2 in range(j1, n_bins):
                cl_i1j1 = c_ells_binned[i1, j1, :]+n_ells_binned[i1, j1, :]
                cl_i1j2 = c_ells_binned[i1, j2, :]+n_ells_binned[i1, j2, :]
                cl_i2j1 = c_ells_binned[i2, j1, :]+n_ells_binned[i2, j1, :]
                cl_i2j2 = c_ells_binned[i2, j2, :]+n_ells_binned[i2, j2, :]
                # Knox formula
                cov = (cl_i1j1 * cl_i2j2 + cl_i1j2 * cl_i2j1) / (d_ell * fsky * (2 * l_eff + 1))
                covar[id_i, :, id_j, :] = np.diag(cov)
                id_j += 1
        id_i += 1
covar = covar.reshape([n_cross * n_bpw, n_cross * n_bpw])

# Open sacc
s = sacc.Sacc()
# Add tracers
for i, n in enumerate(nzs):
    s.add_tracer('NZ', 'wl_%d' % i,  # Name
                 quantity='galaxy_shear',  # Quantity
                 spin=2,  # Spin
                 z=z,  # z
                 nz=n,  # nz
                 sigma_g=0.28)  # You can add any extra information as **kwargs
# Add C_ells
wins = sacc.BandpowerWindow(ells, window_single.T)
for i1 in range(n_bins):
    for i2 in range(i1, n_bins):
        s.add_ell_cl('cl_ee', 'wl_%d' % i1, 'wl_%d' % i2, l_eff, c_ells_binned[i1, i2, :],
                     window=wins)
# Add covariance
s.add_covariance(covar)

s.save_fits("cls_shear.fits", overwrite=True)
