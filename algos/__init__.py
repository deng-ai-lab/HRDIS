
from .ddim import DDIM
from .ddrm import DDRM
from .pgdm import PGDM
from .reddiff import REDDIFF
from .mcg import MCG
from .dps import DPS
from .HRDIS import HRDIS


def build_algo(cg_model, cfg):
    if cfg.algo.name == 'ddim':
        return DDIM(cg_model, cfg)
    elif cfg.algo.name == 'ddrm':
        return DDRM(cg_model, cfg)
    elif cfg.algo.name == 'pgdm':
        return PGDM(cg_model, cfg)
    elif cfg.algo.name == 'HRDIS':
        return HRDIS(cg_model, cfg)
    elif cfg.algo.name == 'reddiff':
        return REDDIFF(cg_model, cfg)
    elif cfg.algo.name == 'dps':
        return DPS(cg_model, cfg)
    else:
        raise ValueError(f'No algorithm named {cfg.algo.name}')
