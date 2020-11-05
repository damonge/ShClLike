from cosmosis.datablock import SectionOptions, DataBlock

nz_model_shift = "shift"
nz_model_none = "none"



# TODO: Should probably split this into
# two different modules, IA and N(z)
class NZSystematics:

    def __init__(self, options):
        self.model = options['model']
        self.tracer = options["tracer"]

    def execute(self, block):
        inputs = self.read_inputs(block)
        results = self.run(inputs)
        self.write_outputs(block, results)
        return 0


    def read_inputs(self, block):
        z = {}
        nz = {}

        keys = block.keys(f'nz_{self.tracer}')
        bins = [k[4:] for _, k in keys if k.startswith('bin_')]
        # Read the n(z)
        section = f"nz_{self.tracer}"
        for b in bins:
            z[b] = block[section, f'z_{b}']
            nz[b] = block[section, f'bin_{b}']

        inputs = {
            "z": z,
            "nz": nz,
            "bins": bins
        }

        if self.model == nz_model_none:
            pass
        elif self.model == nz_model_shift:
            section = f'nz_params_{self.tracer}'
            for b in bins:
                inputs[f"dz_{b}"] = block[section, f"dz_{b}"]
        else:
            raise ValueError(f"Unknown nz model {self.model}.")

        return inputs

    def run(self, inputs):
        bins = inputs["bins"]
        results = {"bins": bins}
        if self.model == nz_model_none:
            for b in bins:
                results[f"z_{b}"] = inputs["z"][b]
                results[f"nz_{b}"] = inputs["nz"][b]
        elif self.model == nz_model_shift:
            for b in bins:
                dz = inputs[f"dz_{b}"]
                results[f"z_{b}"] = inputs["z"][b] - dz
                results[f"nz_{b}"] = inputs["nz"][b]
        else:
            raise ValueError(f"Unknown nz model {self.model}.")

        return results



    def write_outputs(self, block, results):
        section = f"nz_{self.tracer}"
        for b in results["bins"]:
            z = results[f"z_{b}"]
            nz = results[f"nz_{b}"]
            cut = z >= 0
            z = z[cut]
            nz = nz[cut]
            block[section, f"z_{b}"] = z
            block[section, f"bin_{b}"] = nz




# CosmoSIS setup functions can return any object.
def setup(options: DataBlock) -> NZSystematics:
    options = SectionOptions(options)
    return NZSystematics(options)



def execute(block: DataBlock, config: NZSystematics) -> int:
    # rename for clarity
    calculator = config
    return calculator.execute(block)
