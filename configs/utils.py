from yacs.config import CfgNode

from .default import get_cfg_defaults


def setup_cfg(cfg_file=(), set_cfgs=None) -> CfgNode:
    """defaults <- yaml file(s) <- command line (--set key value ...)."""
    cfg = get_cfg_defaults()
    for f in cfg_file:
        cfg.merge_from_file(f)

    if set_cfgs:
        if len(set_cfgs) % 2 != 0:
            raise ValueError(f"--set expects key/value pairs, got: {set_cfgs}")
        pairs = []
        for k, v in zip(set_cfgs[0::2], set_cfgs[1::2]):
            pairs.extend([k, _int2float(v, _get(cfg, k.split('.')))])
        cfg.merge_from_list(pairs)
    return cfg


def _get(c, ks):
    return c[ks[0]] if len(ks) == 1 else _get(c[ks[0]], ks[1:])


def _int2float(x, tgt):
    """'--set opti.l2 1' -> '1.0' when the default is a float (yacs refuses int -> float)."""
    if isinstance(tgt, float) and x.lstrip('-').isdigit():
        return x + '.0'
    return x


def update_from(cfg: CfgNode, ref: CfgNode, inplace=False) -> CfgNode:
    """Fill the None values of cfg with the values of ref (update block <- input block)."""
    if not inplace:
        cfg = cfg.clone()
    cfg.defrost()
    for k in cfg:
        if k in ref and cfg[k] is None and ref[k] is not None:
            cfg[k] = ref[k]
    return cfg


def cfg2flatdict(cfg: CfgNode, prefix: str = "") -> dict:
    """{'Bi.hid_dim': 256, 'Loss.sw': 5.0, ...} for the wandb config."""
    D = {}
    for k, v in cfg.items():
        if isinstance(v, CfgNode):
            D.update(cfg2flatdict(v, prefix=f"{prefix}{k}."))
        else:
            D[f"{prefix}{k}"] = v if isinstance(v, (int, float, bool, str)) else str(v)
    return D


def data_kwargs(cfg) -> dict:
    """Keyword arguments of SequenceDataModule."""
    return dict(
        dir_path=cfg.data.dir_path,
        data_name=cfg.data.data_name,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        version=cfg.data.version,
        cdt_annotation=cfg.data.cdt_annotation,
        blocked=cfg.data.blocked,
        max_frame=cfg.data.max_frame,
        n_tclasses=cfg.n_tclasses,
        n_actions=cfg.n_actions,
    )
