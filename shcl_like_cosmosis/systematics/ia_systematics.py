from cosmosis.datablock import SectionOptions, DataBlock
import numpy as np


ia_model_none = "none"
ia_model_desy1 = "des-y1"



# A more CosmoSIS-native model would be to give
# each of these models its own module.
# We could also give each its own class, that would
# clearly work better than this, but I've already
# re-written this quick prototype twice so I'll stop
# there.
class IASystematics:

    def __init__(self, options):
        self.model = options["model"]
        self.tracer = options["tracer"]

    def execute(self, block):
        inputs = self.read_inputs(block)
        results = self.run(inputs)
        self.write_outputs(block, results)
        return 0

    def read_ia_params(block):
        params = {}

        return params

    def read_inputs(self, block):
        z = {}
        nz = {}

        keys = block.keys(f'nz_{self.tracer}')
        bins = [k[4:] for _, k in keys if k.startswith('bin_')]

        inputs = {
            "z": z,
            "nz": nz,
        }
        # Read the n(z) z values.  We will use these to specify
        # the IA(z)
        section = f"nz_{self.tracer}"
        for b in bins:
            z[b] = block[section, f'z_{b}']

        if self.model == ia_model_none:
            pass
        elif self.model == ia_model_desy1:
            section = f'ia_{self.tracer}'
            inputs['A0'] = block[section, "A"]
            inputs['eta'] = block[section, "eta"]
        else:
            raise ValueError(f"Unknown IA model {self.model}.")

        return inputs

    def run(self, inputs):
        results = {}
        if self.model == ia_model_none:
            for b, z in inputs['z'].items():
                results[f'z_{b}'] = z
                results[f'IA_{b}'] = np.zeros_like(z)
        elif self.model == ia_model_desy1:
            A0 = inputs["A0"]
            eta = inputs["eta"]
            for b, z in inputs['z'].items():
                results[f'z_{b}'] = z
                results[f'IA_{b}'] = A0 * ((1+z)/1.62)**eta
        else:
            raise ValueError(f"Unknown IA model {self.model}.")

        return results

    def write_outputs(self, block, results):
        bins = [key[2:] for key in results if key.startswith('z_')]
        section = f"ia_{self.tracer}"
        for b in bins:
            block[section, f'z_{b}'] = results[f"z_{b}"]
            block[section, f'IA_{b}'] = results[f"IA_{b}"]



# CosmoSIS setup functions can return any object.
def setup(options: DataBlock) -> IASystematics:
    options = SectionOptions(options)
    return IASystematics(options)



def execute(block: DataBlock, config: IASystematics) -> int:
    # rename for clarity
    calculator = config
    return calculator.execute(block)
