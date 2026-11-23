from yacs.config import CfgNode as CN

_C = CN()

_C.in_dim = 2048
_C.n_actions = 11
_C.n_tclasses = 5
_C.head_tclasses = None
_C.seed = 1405
_C.project_name = "MKLag"
_C.ckpt = None

_C.data = CN()
_C.data.embeddings_path = None
_C.data.dir_path = "DataSplit/Morphokinetic"
_C.data.data_name = "bovine.csv"
_C.data.batch_size = 32
_C.data.num_workers = 8
_C.data.max_frame = 301
_C.data.version = "v4"
_C.data.cdt_annotation = True
_C.data.blocked = True

_C.trainer = CN()
_C.trainer.max_epochs = 50
_C.trainer.gradient_clip_val = 10.0
_C.trainer.resume_from_checkpoint = None

_C.FACT = FACT = CN()
FACT.block = "iuu"
FACT.trans = False
FACT.ntoken = 60
FACT.fpos = False
FACT.cmr = 0.1
FACT.mwt = 0.1

_C.Bi = Bi = CN()
Bi.hid_dim = 256
Bi.dropout = 0.0

Bi.a = "sca"
Bi.a_dim = 128
Bi.a_ffdim = 256
Bi.a_layers = 6
Bi.a_nhead = 8

Bi.f = "m2"
Bi.f_dim = 128
Bi.f_layers = 10
Bi.f_ln = False
Bi.f_ngp = 1

Bi.s_head_num_heads = 8
Bi.s_head_ffdim = 256

_C.Bu = Bu = CN()
Bu.hid_dim = None
Bu.dropout = None

Bu.a = "sa"
Bu.a_dim = None
Bu.a_ffdim = None
Bu.a_layers = 1
Bu.a_nhead = 8

Bu.f = None
Bu.f_dim = None
Bu.f_layers = 10
Bu.f_ln = None
Bu.f_ngp = None

Bu.s_head_num_heads = 8
Bu.s_head_ffdim = 256

_C.Loss = Loss = CN()
Loss.frame = 2.0
Loss.action = 1.0
Loss.mk = 1.0
Loss.mk_type = "ce"
Loss.lag_type = "ce"
Loss.ca = 1.0
Loss.sw = 5.0
Loss.nullw = 1.0
Loss.match = "o2o"
Loss.pc = 0.2
Loss.a2fc = 1.0
Loss.lag_ord_w = 1.0
Loss.lag_reg_w = 1.0
Loss.lag_regu_w = 1.0

_C.TM = TM = CN()
TM.use = False
TM.t = 30
TM.p = 0.05
TM.m = 5

_C.opti = CN()
_C.opti.optimizer = "AdamW"
_C.opti.learning_rate = 0.0005
_C.opti.l2 = 0.05
_C.opti.schedule = "Warmup"
_C.opti.swa = True
_C.opti.swa_lrs = 1e-3
_C.opti.swa_epoch_start = 0.75


def get_cfg_defaults():
    return _C.clone()