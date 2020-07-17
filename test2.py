from shcl_like.shcl_like import ShClLike
from shcl_like.clccl import CLCCL
from cobaya.model import get_model
import yaml


info = {'params': {},
        'likelihood': {'shcl':
                       {'external': ShClLike}},
        'theory': {'camb': None,
                   'clccl': {"external": CLCCL}},
        "debug": True}
with open('test_data/params_test2.yml', "r") as fin:
    pp = yaml.load(fin, Loader=yaml.FullLoader)
info['params'] = pp['params']
info['likelihood']['shcl'].update(pp['shcl'])
info['theory']['clccl'].update(pp.get('clccl', {}))

model = get_model(info)
loglikes, derived = model.loglikes({})
print(-2 * loglikes)
