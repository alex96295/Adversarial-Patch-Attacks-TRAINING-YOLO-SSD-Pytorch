from torch import optim


class BaseConfig(object):
    """
    Default parameters for all config files.
    """

    def __init__(self):
        """
        Set the defaults.
        """
        self.img_dir = "../inria/INRIAPerson/Train/pos"  # train images location
        self.lab_dir = "../inria/INRIAPerson/Train/pos/inria_anno_darknet_format"  # train labels location

        self.cfgfile_yolov2 = "./cfg/yolo.cfg"
        self.cfgfile_yolov3 = "./cfg/yolov3.cfg"
        self.cfgfile_yolov4 = "./cfg/yolov4.cfg"
        
        self.weightfile_yolov2 = "./weights/yolov2.weights"
        self.weightfile_yolov3 = "./weights/yolov3.weights"
        self.weightfile_yolov3pt_ultra = "./weights/yolov3_ultralytics.pt"
        self.weightfile_yolov4 = "./weights/yolov4.weights"

        #self.cfgfile_ssds = "C:/Users/Alessandro/PycharmProjects/YOLOv2_d2p_adv_COCO_thys2019/networks_cfgs/cfgs/ssd_lite_mobilenetv2_train_coco.yml"
        
        self.ssdmbntv1_model_path = "./mbntv1_ssd_voc.pth"
        self.ssdlitembntv2_model_path = "./mbnt2_ssd_lite_voc.pth"
        self.ssdvgg_model_path = "./vgg16_ssd_voc.pth"
        
        self.printfile = "./non_printability/30values.txt"
        self.patch_size = 300

        self.start_learning_rate = 0.03

        self.patch_name = 'base'

        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50)
        # reduce learning rate when a metric has stopped learning (keras??)
        # In min mode, lr will be reduced when the quantity monitored has stopped decreasing;
        # in max mode it will be reduced when the quantity monitored has stopped increasing. Default: ‘min’.

        self.max_tv = 0

        self.batch_size = 8

        self.loss_target = lambda obj, cls: obj*cls #for yolo only


class Experiment1(BaseConfig):
    """
    Model that uses a maximum total variation, tv cannot go below this point.
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.patch_name = 'Experiment1'
        self.max_tv = 0.165


class Experiment2HighRes(Experiment1):
    """
    Higher res
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.max_tv = 0.165
        self.patch_size = 400
        self.patch_name = 'Exp2HighRes'

class Experiment3LowRes(Experiment1):
    """
    Lower res
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.max_tv = 0.165
        self.patch_size = 100
        self.patch_name = "Exp3LowRes"

class Experiment4ClassOnly(Experiment1):
    """
    Only minimise class score.
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.patch_name = 'Experiment4ClassOnly'
        self.loss_target = lambda obj, cls: cls




class Experiment1Desktop(Experiment1):
    """
    """

    def __init__(self):
        """
        Change batch size.
        """
        super().__init__()

        self.batch_size = 8
        self.patch_size = 400


class ReproducePaperObj(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.batch_size = 8
        self.patch_size = 300

        self.patch_name = 'ObjectOnlyPaper'
        self.max_tv = 0.165

        self.loss_target = lambda obj, cls: obj #for yolo only


patch_configs = {
    "base": BaseConfig,
    "exp1": Experiment1,
    "exp1_des": Experiment1Desktop,
    "exp2_high_res": Experiment2HighRes,
    "exp3_low_res": Experiment3LowRes,
    "exp4_class_only": Experiment4ClassOnly,
    "paper_obj": ReproducePaperObj
}
