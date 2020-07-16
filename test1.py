from shcl_like.shcl_like import ShClLike
from shcl_like.ccl import CCL
from cobaya.model import get_model
import yaml

                
info = {'params': {},
        'likelihood': {'shcl':
                       {'external': ShClLike}},
        'theory': {'ccl': {"external": CCL}},
        "debug": True}
with open('test_data/params_test1.yml', "r") as fin:
    pp = yaml.load(fin, Loader=yaml.FullLoader)
info['params'] = pp['params']
info['likelihood']['shcl'].update(pp['shcl'])
info['theory']['ccl'].update(pp.get('ccl', {}))

model = get_model(info)
loglikes, derived = model.loglikes({})
print(-2 * loglikes)
