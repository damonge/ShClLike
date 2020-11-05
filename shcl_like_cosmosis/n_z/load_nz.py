from cosmosis.datablock import DataBlock, names, option_section
import sacc
import numpy as np

# CosmoSIS setup functions can return any
# object.  In this case we use a dict.
# We could also 
def setup(options: DataBlock) -> dict:
    sacc_file = options[option_section, 'sacc_file']
    nz_name = options[option_section, 'nz_name']

    sacc_data = sacc.Sacc.load_fits(sacc_file)

    nz = {}
    z = None
    for name, tracer in sacc_data.tracers.items():

        # Currently assume wl_0, wl_1, wl_2 etc.
        if name.startswith(nz_name + "_"):
            suffix = name[len(nz_name)+1:]
            nz[suffix] = tracer.nz

            if (z is not None) and not np.allclose(z, tracer.z):
                raise ValueError("Expecting same redshift z for each tracer group (for now)")
            z = tracer.z
    if not nz:
        raise ValueError(f"No tracers found with given name in {sacc_file}")

    config = {}
    config['name'] = f'nz_{nz_name}'
    config['nz'] = nz
    config['z'] = z

    return config


def execute(block: DataBlock, config: dict) -> int:
    section = config['name']
    nz = config['nz']
    z = config['z']

    # Store a different z for each n(z), so that
    # it's easier to apply z shifts.
    block[section, 'nbin'] = len(nz)
    for b, nz_b in nz.items():
        block[section, f'z_{b}'] = z
        block[section, f'bin_{b}'] = nz_b


    return 0