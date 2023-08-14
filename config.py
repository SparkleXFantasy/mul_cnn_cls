from yacs.config import CfgNode as CN


class BaseConfig:
    def __init__(self, cfg_path=None):
        self.init_default_cfg()
        if cfg_path:
            self.__merge_from_file(cfg_path)

    def cfg(self):
        return self.__C.clone()
    
    def __merge_from_file(self, cfg_path):
        self.__C.merge_from_file(cfg_path)

    def init_default_cfg(self):
        self.__C = CN()
        self.__C.model = CN()
        self.__C.model.name = 'resnet50'
        self.__C.model.path = 'checkpoints'
        self.__C.model.load_from_checkpoint = False
        self.__C.model.checkpoint_name = 'checkpoint_latest.pth'
        self.__C.model.use_gpu = True
        self.__C.model.pretrained = True
        self.__C.model.feat = CN()
        self.__C.model.feat.backbone = 'resnet50'
        self.__C.model.feat.freq_branch_enabled = True
        self.__C.model.feat.freq_filter = [0.5]
        self.__C.model.feat.noise_branch_enabled = True
        self.__C.model.feat.fusion_out_channels = 256
        self.__C.model.head_cls = 6

        self.__C.train = CN()
        self.__C.train.dataset = CN()
        self.__C.train.dataset.data_root = 'data'
        self.__C.train.dataset.batch_size = 1
        self.__C.train.dataset.split = [0.7, 0.2, 0.1]
        self.__C.train.data_transforms = CN()
        self.__C.train.data_transforms.crop = 224
        self.__C.train.data_transforms.crop_enabled = True
        self.__C.train.data_transforms.flip_enabled = True
        self.__C.train.data_transforms.normalize = CN()
        self.__C.train.data_transforms.normalize.mean = [0.485, 0.456, 0.406]
        self.__C.train.data_transforms.normalize.std = [0.229, 0.224, 0.225]
        self.__C.train.data_transforms.normalize_enabled = True
        self.__C.train.data_transforms.post_processing = CN()
        self.__C.train.data_transforms.post_processing.gaussian = CN()
        self.__C.train.data_transforms.post_processing.gaussian.prob = 0.5
        self.__C.train.data_transforms.post_processing.gaussian.sigma = 0.25
        self.__C.train.data_transforms.post_processing.gaussian_enabled = True
        self.__C.train.data_transforms.post_processing.jpeg = CN()
        self.__C.train.data_transforms.post_processing.jpeg.prob = 0.5
        self.__C.train.data_transforms.post_processing.jpeg.quality = [90, 95]
        self.__C.train.data_transforms.post_processing.jpeg_enabled = True
        self.__C.train.data_transforms.post_processing_enabled = True
        self.__C.train.data_transforms.resize = 256
        self.__C.train.data_transforms.resize_enabled = True
        self.__C.train.epoch = 100
        self.__C.train.hyperparameter = CN()
        self.__C.train.hyperparameter.contrastive_loss = 0.01
        self.__C.train.hyperparameter.contrastive_loss_enabled = False
        self.__C.train.hyperparameter.early_stop = 5
        self.__C.train.hyperparameter.early_stop_metric = 'AUC'
        self.__C.train.hyperparameter.early_stop_enabled = False
        self.__C.train.hyperparameter.learning_rate = 0.001
        self.__C.train.hyperparameter.weight_decay = 0
        self.__C.train.metrics = ['AUC', 'F1']
        self.__C.train.save_step = 5
        self.__C.train.save_dir = 'output'
        self.__C.train.save_name = 'mul_resnet50'
        self.__C.train.seed = 1
        self.__C.train.log_step = 1

        self.__C.test = CN()
        self.__C.test.dataset = CN()
        self.__C.test.dataset.data_root = 'data'
        self.__C.test.dataset.batch_size = 1
        self.__C.test.data_transforms = CN()
        self.__C.test.data_transforms.crop = 224
        self.__C.test.data_transforms.crop_enabled = True
        self.__C.test.data_transforms.flip_enabled = True
        self.__C.test.data_transforms.normalize = CN()
        self.__C.test.data_transforms.normalize.mean = [0.485, 0.456, 0.406]
        self.__C.test.data_transforms.normalize.std = [0.229, 0.224, 0.225]
        self.__C.test.data_transforms.normalize_enabled = True
        self.__C.test.data_transforms.post_processing = CN()
        self.__C.test.data_transforms.post_processing.gaussian = CN()
        self.__C.test.data_transforms.post_processing.gaussian.prob = 0.5
        self.__C.test.data_transforms.post_processing.gaussian.sigma = 0.25
        self.__C.test.data_transforms.post_processing.gaussian_enabled = True
        self.__C.test.data_transforms.post_processing.jpeg = CN()
        self.__C.test.data_transforms.post_processing.jpeg.prob = 0.5
        self.__C.test.data_transforms.post_processing.jpeg.quality = [90, 95]
        self.__C.test.data_transforms.post_processing.jpeg_enabled = True
        self.__C.test.data_transforms.post_processing_enabled = True
        self.__C.test.data_transforms.resize = 256
        self.__C.test.data_transforms.resize_enabled = True
        self.__C.test.metrics = ['AUC', 'F1']
        self.__C.test.save_dir = 'output'
        self.__C.test.seed = 1
