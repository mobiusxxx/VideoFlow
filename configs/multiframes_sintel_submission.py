from yacs.config import CfgNode as CN
_CN = CN()

_CN.name = ''
_CN.suffix =''
_CN.gamma = 0.85
_CN.max_flow = 400
_CN.batch_size = 8
_CN.sum_freq = 100
_CN.val_freq = 100000000
_CN.image_size = [432, 960]
_CN.add_noise = False
_CN.use_smoothl1 = True
_CN.critical_params = []
_CN.infer_size = [1280, 720]
_CN.rois = [[517, 6], [650, 3], [669, 712], [517, 712]]  #[[514, 6], [674, 5], [670, 710], [508, 712]]   #[[522, 8], [672, 6], [670, 708], [520, 709]] #[[428, 3], [840, 2], [870, 715], [444, 713]]  #[[560, 200], [640, 200], [640, 520], [560, 520]]  # # [1070, 2133]] # [[511, 7], [672, 3], [674, 712], [510, 713]]   # [[558, 262], [626, 262], [626, 648], [558, 648]]  #[[1056, 9], [2913, 5], [2919, 2135], [976, 2125]]

_CN.network = 'MOFNetStack'

_CN.model = r'C:\dtt\code\VideoFlow-main\VideoFlow_ckpt_20240119T143605Z_001\VideoFlow_ckpt\MOF_sintel.pth'
_CN.input_frames = 10

# _CN.restore_ckpt = None

################################################
################################################
_CN.MOFNetStack = CN()
_CN.MOFNetStack.pretrain = True
_CN.MOFNetStack.Tfusion = 'stack'
_CN.MOFNetStack.cnet = 'twins'
_CN.MOFNetStack.fnet = 'twins'
_CN.MOFNetStack.down_ratio = 8
_CN.MOFNetStack.feat_dim = 256
_CN.MOFNetStack.corr_fn = 'default'
_CN.MOFNetStack.corr_levels = 4
_CN.MOFNetStack.mixed_precision = True
_CN.MOFNetStack.context_3D = False
_CN.MOFNetStack.position_only = False
_CN.MOFNetStack.position_and_content = True
_CN.MOFNetStack.decoder_depth = 16
_CN.MOFNetStack.critical_params = ["cnet", "fnet", "pretrain", 'corr_fn', "Tfusion", "corr_levels", "decoder_depth", "mixed_precision", "position_only", "position_and_content"]

### TRAINER
_CN.trainer = CN()
_CN.trainer.scheduler = 'OneCycleLR'
_CN.trainer.optimizer = 'adamw'
_CN.trainer.canonical_lr = 12.5e-5
_CN.trainer.adamw_decay = 1e-4
_CN.trainer.clip = 1.0
_CN.trainer.num_steps = 90000
_CN.trainer.epsilon = 1e-8
_CN.trainer.anneal_strategy = 'linear'


def get_cfg():
    return _CN.clone()
