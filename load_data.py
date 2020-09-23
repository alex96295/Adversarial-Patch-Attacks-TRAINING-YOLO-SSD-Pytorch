import fnmatch
import math
import os
import sys
import time
from operator import itemgetter

import gc
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

#from darknet import Darknet

from median_pool import MedianPool2d  # see median_pool.py

# print('Test image loading on a random image:')
# im = Image.open('C:/Users/Alessandro/PycharmProjects/YOLOv2_d2p_adv_COCO_thys2019/data/horse.jpg').convert('RGB')
# print('Image has been read correctly!')


class yolov2_feature_output_manage(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, cls_id, num_cls, config):
        super(yolov2_feature_output_manage, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.config = config

    def forward(self, YOLOoutput, loss_type):
        # get values necessary for transformation

        if YOLOoutput.dim() == 3:
            YOLOoutput = YOLOoutput.unsqueeze(0)  # add one dimension of size 1
        batch = YOLOoutput.size(0)
        assert (YOLOoutput.size(1) == (5 + self.num_cls ) * 5)  # the last 5 is the anchor boxes number after k-means clustering
                                                                # the first 5 is the number of parameters of each box: x, y, w, h, objectness score
                                                                # self.num_cls indicates class probabilities, i.e. 20 values for VOC and 80 for COCO
                                                                # in total, there are 125 parameters per grid cell when VOC, 425 when COCO
        h = YOLOoutput.size(2)
        w = YOLOoutput.size(3)

        # print('dim0:' + str(YOLOoutput.size(0)))
        # print('dim1:' + str(YOLOoutput.size(1)))
        # print('h:' + str(h))
        # print('w:' + str(h))


        # transform the output tensor from [batch, 425, 13, 13] to [batch, 80, 1805]
        output = YOLOoutput.view(batch, 5, 5 + self.num_cls , h * w)  # [batch, 5, 85, 169]
        output = output.transpose(1, 2).contiguous()  # [batch, 85, 5, 169] swap 5 and 85, in position 1 and 2 respectively
        output = output.view(batch, 5 + self.num_cls , 5 * h * w)  # [batch, 85, 845]
        # todo first 5 numbers that make '85' are box xc, yc, w, h and objectness. Last 80 are class prob.

        # print(output[:, 4, :])
        # print(output[:, 5, :])
        # print(output[:, 6, :])
        # print(output[:, 34, :])

        output_objectness_not_norm = output[:, 4, :]
        output_objectness_norm = torch.sigmoid(output[:, 4, :])  # [batch, 1, 845]  # iou_truth*P(obj)
        # take the fifth value, i.e. object confidence score. There is one value for each box, in total 5 boxes

        output = output[:, 5:5 + self.num_cls , :]  # [batch, 80, 845]  # 845 = 5 * h * w
        # NB 80 means conditional class probabilities, one for each class related to a single box (there are 5 box for each grid cell)

        # perform softmax to normalize probabilities for object classes to [0,1] along the 1st dim of size 80 (no. classes in COCO)
        not_normal_confs = output
        normal_confs = torch.nn.Softmax(dim=1)(output)
        # NB Softmax is applied to all slices along dim, and will re-scale them so that the elements lie in the range [0, 1] and sum to 1

        # we only care for conditional probabilities of the class of interest (person, i.e. cls_id = 0 in COCO)
        confs_for_class_not_normal = not_normal_confs[:, self.cls_id, :]
        confs_for_class_normal = normal_confs[:, self.cls_id, :] # take class number 0, so just one kind of cond. prob out of 80. This is for 1 box, there are 5 boxes

        confs_if_object_not_normal = self.config.loss_target(output_objectness_not_norm, confs_for_class_not_normal)
        confs_if_object_normal = self.config.loss_target(output_objectness_norm, confs_for_class_normal)  # loss_target in patch_config

        if loss_type == 'max_approach':
            max_conf, max_conf_idx = torch.max(confs_if_object_normal, dim=1) # take the maximum value among your 5 priors
            return max_conf

        elif loss_type == 'threshold_approach':
            threshold = 0.3
            batch_stack = torch.unbind(confs_if_object_normal, dim=0)
            print('yolo batch stack: \n')
            print(batch_stack)
            penalized_tensor_batch = []
            for img_tensor in batch_stack:
                size = img_tensor.size()
                zero_tensor = torch.cuda.FloatTensor(size).fill_(0)
                penalized_tensor = torch.max(img_tensor - threshold, zero_tensor) ** 2
                penalized_tensor_batch.append(penalized_tensor)

            penalized_tensor_batch = torch.stack(penalized_tensor_batch, dim=0)
            thresholded_conf = torch.sum(penalized_tensor_batch, dim=1)
            return thresholded_conf
            

class ssd_feature_output_manage(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, cls_id, num_cls, config):
        super(ssd_feature_output_manage, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.config = config
        #self.num_priors = num_priors

    def forward(self, ssd_output, loss_type):
        # get values necessary for transformation
        conf_normal, conf_not_normal, loc = ssd_output

        obj_scores = torch.sum(conf_normal[:,:,1:], dim=2)                                       
                                                                                         
        confs_if_object_normal = conf_normal[:, :, self.cls_id] # softmaxed #obj*cls             
                                                                                         
        person_cls_score = confs_if_object_normal / obj_scores                                   
                                                                                         
        loss_target = confs_if_object_normal                                                                

        if loss_type == 'max_approach':
            max_conf, max_conf_idx = torch.max(loss_target, dim=1) # take the maximum value among your 5 priors
            return max_conf
        elif loss_type == 'threshold_approach':
            threshold = 0.35
            batch_stack = torch.unbind(confs_if_object_normal, dim=0)
            #print('ssd batch stack: \n')
            #print(batch_stack)
            penalized_tensor_batch = []
            for img_tensor in batch_stack:
                size = img_tensor.size()
                zero_tensor = torch.cuda.FloatTensor(size).fill_(0)
                penalized_tensor = torch.max(img_tensor - threshold, zero_tensor)**2
                #penalized_tensor = penalized_tensor**2
                penalized_tensor_batch.append(penalized_tensor)

            penalized_tensor_batch = torch.stack(penalized_tensor_batch, dim=0)
            thresholded_conf = torch.sum(penalized_tensor_batch, dim=1)
            return thresholded_conf

        # code below good for coco training
        # loc_data = loc.data
        # conf_data = conf.data
        # print(conf_data.size())
        # num_priors = self.num_priors.data.size(0)
        #
        # batch = loc_data.size(0)
        #
        # output = conf_data.view(batch, num_priors, self.num_cls)
        # output = output[:, :, self.cls_id]
        # max_conf, max_conf_idx = torch.max(output, dim=1)


class yolov3_feature_output_manage(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, cls_id, num_cls, config):
        super(yolov3_feature_output_manage, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.config = config
        #self.num_priors = num_priors

    def forward(self, yv3_output, loss_type):
        # get values necessary for transformation

        #print(len(yv3_output))

        #print(yv3_output[0].size(), len(yv3_output[1]))

        #print(yv3_output[1][0].size(), yv3_output[1][1].size(), yv3_output[1][2].size())
        yolo_output = yv3_output[0]

        #print(yolo_output.size())

        

        loc = yolo_output[:, :, :4]
        objectness = yolo_output[:, :, 4]
        cond_prob = yolo_output[:, :, 5:]

        cond_prob_targeted_class = cond_prob[:, :, self.cls_id]

        confs_if_object_normal = self.config.loss_target(objectness, cond_prob_targeted_class)

        if loss_type == 'max_approach':
            max_conf, max_conf_idx = torch.max(confs_if_object_normal, dim=1)
            return max_conf

        elif loss_type == 'threshold_approach':
            threshold = 0.35
            batch_stack = torch.unbind(confs_if_object_normal, dim=0)
            penalized_tensor_batch = []
            for img_tensor in batch_stack:
                size = img_tensor.size()
                zero_tensor = torch.cuda.FloatTensor(size).fill_(0)
                penalized_tensor = torch.max(img_tensor - threshold, zero_tensor) ** 2
                penalized_tensor_batch.append(penalized_tensor)

            penalized_tensor_batch = torch.stack(penalized_tensor_batch, dim=0)
            thresholded_conf = torch.sum(penalized_tensor_batch, dim=1)
            return thresholded_conf
            
class yolov4_feature_output_manage(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, cls_id, num_cls, config):
        super(yolov4_feature_output_manage, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.config = config

    def forward(self, YOLOoutput, loss_type):
        # get values necessary for transformation
        output_list = []
        h_list = []
        w_list = []
        for out_layer in YOLOoutput:
            if out_layer.dim() == 3:
                out_layer = out_layer.unsqueeze(0)  # add one dimension of size 1 if there is not
            batch = out_layer.size(0)
            assert (out_layer.size(1) == (5 + self.num_cls) * 3)  # the last 3 is the anchor boxes number after k-means clustering
                                                                    # the first 5 is the number of parameters of each box: x, y, w, h, objectness score
                                                                    # self.num_cls indicates class probabilities, i.e. 20 values for VOC and 80 for COCO
                                                                    # in total, there are 125 parameters per grid cell when VOC, 425 when COCO
            h = out_layer.size(2)
            w = out_layer.size(3)

            # print('dim0:' + str(YOLOoutput.size(0)))
            # print('dim1:' + str(YOLOoutput.size(1)))
            # print('h:' + str(h))
            # print('w:' + str(h))

            output_layer = out_layer.view(batch, 3, 5 + self.num_cls, h * w)
            output_layer = output_layer.transpose(1, 2).contiguous()
            output_layer = output_layer.view(batch, 5 + self.num_cls, 3 * h * w)

            # out_layer = out_layer.view(batch, h*w*3, (5 + self.num_cls))
            output_list.append(output_layer)

        total_out = torch.cat([output_list[0], output_list[1], output_list[2]], dim=2)
        #print(total_out.size())

        #print(total_out[:,4,:])
        objectness_score = torch.sigmoid(total_out[:,4,:])
        #print(objectness_score)

        class_cond_prob = torch.nn.Softmax(dim=1)(total_out[:, 5:5+self.num_cls,:])
        person_cond_prob = class_cond_prob[:, self.cls_id,:]
        confs_if_object_normal = self.config.loss_target(objectness_score, person_cond_prob)  # loss_target in patch_config

        if loss_type == 'max_approach':
            max_conf, max_conf_idx = torch.max(confs_if_object_normal, dim=1) # take the maximum value among your 5 priors
            print(max_conf)
            return max_conf

        elif loss_type == 'threshold_approach':
            threshold = 0.3
            batch_stack = torch.unbind(confs_if_object_normal, dim=0)
            print('yolo batch stack: \n')
            print(batch_stack)
            penalized_tensor_batch = []
            for img_tensor in batch_stack:
                size = img_tensor.size()
                zero_tensor = torch.cuda.FloatTensor(size).fill_(0)
                penalized_tensor = torch.max(img_tensor - threshold, zero_tensor) ** 2
                penalized_tensor_batch.append(penalized_tensor)

            penalized_tensor_batch = torch.stack(penalized_tensor_batch, dim=0)
            thresholded_conf = torch.sum(penalized_tensor_batch, dim=1)
            return thresholded_conf
        
        elif loss_type == 'carlini_wagner':
            loss = (objectness_score*person_cond_prob - (1 - objectness_score))
            batch_stack = torch.unbind(loss, dim=0)
            loss_total = []
            for img_loss in batch_stack:
                size = img_loss.size()
                zero_tensor = torch.cuda.FloatTensor(size).fill_(0)
                batch_loss = torch.max((img_loss), zero_tensor)
                loss_total.append(batch_loss)
            loss_total = torch.stack(loss_total, 0)
            c_w_conf = torch.sum(loss_total, dim=1)
            print(c_w_conf)
            return c_w_conf

class NPSCalculator(nn.Module):
    """NMSCalculator: calculates the non-printability score of a patch.

    Module providing the functionality necessary to calculate the non-printability score (NMS) of an adversarial patch.

    """

    def __init__(self, printability_file, patch_side):
        super(NPSCalculator, self).__init__()
        self.printability_array = nn.Parameter(self.get_printability_array(printability_file, patch_side),requires_grad=False)

    def forward(self, adv_patch):
        # calculate euclidian distance between colors in patch and colors in printability_array 
        # square root of sum of squared difference

        color_dist = (adv_patch - self.printability_array+0.000001)
        #print(color_dist.size())
        color_dist = color_dist ** 2  # squared difference
        color_dist = torch.sum(color_dist, 1)+0.000001
        #print(color_dist.size())
        color_dist = torch.sqrt(color_dist)

        # only work with the min distance
        color_dist_prod = torch.min(color_dist, 0)[0] #test: change prod for min (find distance to closest color)
        #print(type(color_dist_prod))
        #print('size ' + str(color_dist_prod.size()))

        # calculate the nps by summing over all pixels
        nps_score = torch.sum(color_dist_prod,0)
        nps_score = torch.sum(nps_score,0)
        return nps_score/torch.numel(adv_patch)  # divide by the total number of elements in the input tensor

    def get_printability_array(self, printability_file, side):
        #  side = patch_size in adv_examples.py
        printability_list = []

        # read in printability triplets and put them in a list
        with open(printability_file) as f:
            for line in f:
                printability_list.append(line.split(","))

        # see notes for a better graphical representation
        printability_array = []
        for printability_triplet in printability_list:
            printability_imgs = []
            red, green, blue = printability_triplet
            printability_imgs.append(np.full((side, side), red))
            printability_imgs.append(np.full((side, side), green))
            printability_imgs.append(np.full((side, side), blue))

            printability_array.append(printability_imgs)

        printability_array = np.asarray(printability_array)  # convert input lists, tuples etc. to array
        printability_array = np.float32(printability_array)
        pa = torch.from_numpy(printability_array)  # Creates a Tensor from a numpy array.
        return pa


class TotalVariation(nn.Module):
    """TotalVariation: calculates the total variation of a patch.

    Module providing the functionality necessary to calculate the total variation (TV) of an adversarial patch.

    """

    def __init__(self):
        super(TotalVariation, self).__init__()

    def forward(self, adv_patch):
        # compute total variation of the adv_patch
        tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1]+0.000001),0)  # NB -1 indicates the last element!
        tvcomp1 = torch.sum(torch.sum(tvcomp1,0),0)

        tvcomp2 = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :]+0.000001),0)
        tvcomp2 = torch.sum(torch.sum(tvcomp2,0),0)
        tv = tvcomp1 + tvcomp2
        return tv/torch.numel(adv_patch)


class PatchTransformer(nn.Module):
    """PatchTransformer: transforms batch of patches

    Module providing the functionality necessary to transform a batch of patches, randomly adjusting brightness and
    contrast, adding random amount of noise, and rotating randomly. Resizes patches according to as size based on the
    batch of labels, and pads them to the dimension of an image.

    """

    def __init__(self):
        super(PatchTransformer, self).__init__()
        
        # os.environ['CUDA_VISIBLE_DEVICES'] = '3'
        # self.device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

        
        self.min_contrast = 0.8
        self.max_contrast = 1.2
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.10
        self.minangle = -20 / 180 * math.pi
        self.maxangle = 20 / 180 * math.pi
        self.medianpooler = MedianPool2d(7, same=True)  # kernel_size = 7? see again
        '''
        kernel = torch.cuda.FloatTensor([[0.003765, 0.015019, 0.023792, 0.015019, 0.003765],
                                         [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
                                         [0.023792, 0.094907, 0.150342, 0.094907, 0.023792],
                                         [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
                                         [0.003765, 0.015019, 0.023792, 0.015019, 0.003765]])
        self.kernel = kernel.unsqueeze(0).unsqueeze(0).expand(3,3,-1,-1)
        '''
    def forward(self, adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=True):

        use_cuda = 1

        #adv_patch = F.conv2d(adv_patch.unsqueeze(0),self.kernel,padding=(2,2))
        adv_patch = self.medianpooler(adv_patch.unsqueeze(0))  # pre-processing on the image with 1 more dimension: 1 x 3 x 300 x 300, see median_pool.py

        # Determine size of padding
        pad = (img_size - adv_patch.size(-1)) / 2  # img_size = 416, adv_patch size = patch_size in adv_examples.py, = 300
        # print('pad =' + str(pad)) # pad = 0.5*(416 - 300) = 58

        # Make a batch of patches
        adv_patch = adv_patch.unsqueeze(0)
        # print('adv_patch in load_data.py, PatchTransforme, size =' + str(adv_patch.size()))
        # adv_patch in load_data.py, PatchTransforme, size =torch.Size([1, 1, 3, 300, 300]), tot 5 dimensions

        adv_batch = adv_patch.expand(lab_batch.size(0), lab_batch.size(1), -1, -1, -1)
        # print('adv_batch in load_data.py, PatchTransforme, size =' + str(adv_batch.size()))
        # adv_batch in load_data.py, PatchTransforme, size =torch.Size([6, 14, 3, 300, 300])

        batch_size = torch.Size((lab_batch.size(0), lab_batch.size(1)))
        # print('batch_size in load_data.py, PatchTransforme, size =' + str(batch_size))
        # batch_size in load_data.py, PatchTransforme, size =torch.Size([6, 14])

        # Contrast, brightness and noise transforms

        # Create random contrast tensor

        if use_cuda:
            contrast = torch.cuda.FloatTensor(batch_size).uniform_(self.min_contrast, self.max_contrast)
        else:
            contrast = torch.FloatTensor(batch_size).uniform_(self.min_contrast, self.max_contrast)
            # Fills self tensor (here 6 x 14) with numbers sampled from the continuous uniform distribution: 1/(max_contrast - min_contrast)

        # print('contrast1 in load_data.py, PatchTransforme, size =' + str(contrast.size()))
        # contrast1 in load_data.py, PatchTransforme, size =torch.Size([6, 14])

        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # print('contrast2 in load_data.py, PatchTransforme, size =' + str(contrast.size()))
        # contrast2 in load_data.py, PatchTransforme, size =torch.Size([6, 14, 1, 1, 1])

        contrast = contrast.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        # print('contrast3 in load_data.py, PatchTransforme, size =' + str(contrast.size()))
        # contrast3 in load_data.py, PatchTransforme, size =torch.Size([6, 14, 3, 300, 300])

        # lines 206-221 could be replaced by:
        # contrast = torch.FloatTensor(adv_batch).uniform_(self.min_contrast, self.max_contrast)
        # print('contrast4 in load_data.py, PatchTransforme, size =' + str(contrast.size()))

        if use_cuda:
            contrast = contrast.cuda()
        else:
            contrast = contrast
#_________________________________________________________________________________________________________________________________________________
        # Create random brightness tensor
        if use_cuda:
            brightness = torch.cuda.FloatTensor(batch_size).uniform_(self.min_brightness, self.max_brightness)
        else:
            brightness = torch.FloatTensor(batch_size).uniform_(self.min_brightness, self.max_brightness)

        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        brightness = brightness.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))

        # lines 227-239 could be replaced by:
        # brightness = torch.FloatTensor(adv_batch).uniform_(self.min_brightness, self.max_brightness)
        # print('brightness in load_data.py, PatchTransforme, size =' + str(brightness.size()))

        if use_cuda:
            brightness = brightness.cuda()
        else:
            brightness = brightness

# _____________________________________________________________________________________________________________________________________________
        # Create random noise tensor
        if use_cuda:
            noise = torch.cuda.FloatTensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor
        else:
            noise = torch.FloatTensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor
        # dim: 6 x 14 x 3 x 300 x 300
#______________________________________________________________________________________________________________________________________________
        # Apply contrast/brightness/noise, clamp
        adv_batch = adv_batch * contrast + brightness + noise

        adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)  # keep all elements in the range 0.000001-0.99999 (real numbers since FLoatTensor)
        # dim: 6 x 14 x 3 x 300 x 300

#______________________________________________________________________________________________________________________________________________
        # Where the label class_ids is 1 we don't want a patch (padding) --> fill mask with zero's

        cls_ids = torch.narrow(lab_batch, 2, 0, 1)  # Consider just the first 'column' of lab_batch, where we can
                                                    # discriminate between detected person (or 'yes person') and 'no person')
                                                    # in this way, sensible data about x, y, w and h of the rectangles are not used for building the mask

        # NB torch.narrow returns a new tensor that is a narrowed version of input tensor. The dimension dim is input from start to start + length.
        # The returned tensor and input tensor share the same underlying storage.

        cls_mask = cls_ids.expand(-1, -1, 3)
        cls_mask = cls_mask.unsqueeze(-1)
        cls_mask = cls_mask.expand(-1, -1, -1, adv_batch.size(3))
        cls_mask = cls_mask.unsqueeze(-1)
        cls_mask = cls_mask.expand(-1, -1, -1, -1, adv_batch.size(4))  # 6 x 14 x 3 x 300 x 300

        if use_cuda:
            msk_batch = torch.cuda.FloatTensor(cls_mask.size()).fill_(1) - cls_mask
        else:
            msk_batch = torch.FloatTensor(cls_mask.size()).fill_(1) - cls_mask  # take a matrix of 1s, subtract that of the labels so that
                                                                                # we can have 0s where there is no person detected,
                                                                                # obtained by doing 1-1=0

        # NB! Now the mask has 1s 'above', where the labels data are sensible since they represent detected persons, and 0s where there are no detections
        # In this way, multiplying the adv_batch to this mask, built from the lab_batch tensor, allows to target only detected persons and nothing else,
        # i.e. pad with zeros the rest
#_______________________________________________________________________________________________________________________________________________
        # Pad patch and mask to image dimensions with zeros
        mypad = nn.ConstantPad2d((int(pad + 0.5), int(pad), int(pad + 0.5), int(pad)), 0)
        adv_batch = mypad(adv_batch)  # dim 6 x 14 x 3 x 416 x 416
        msk_batch = mypad(msk_batch)  # dim 6 x 14 x 3 x 416 x 416

        # NB you see only zeros when you print it because they are all surrounding the patch to pad it to image dimensions (3 x 416 x 416)

#_______________________________________________________________________________________________________________________________________________
        # Rotation and rescaling transforms
        anglesize = (lab_batch.size(0) * lab_batch.size(1))  # dim = 6*14 = 84
        if do_rotate:
            if use_cuda:
                angle = torch.cuda.FloatTensor(anglesize).uniform_(self.minangle, self.maxangle)
            else:
                angle = torch.FloatTensor(anglesize).uniform_(self.minangle, self.maxangle)

        else:
            if use_cuda:
                angle = torch.cuda.FloatTensor(anglesize).fill_(0)
            else:
                angle = torch.FloatTensor(anglesize).fill_(0)
#_______________________________________________________________________________________________________________________________________________
        # Resizes and rotates
        current_patch_size = adv_patch.size(-1)  # 300

        if use_cuda:
            lab_batch_scaled = torch.cuda.FloatTensor(lab_batch.size()).fill_(0)
        else:
            lab_batch_scaled = torch.FloatTensor(lab_batch.size()).fill_(0)  # dim 6 x 14 x 5

        lab_batch_scaled[:, :, 1] = lab_batch[:, :, 1] * img_size
        lab_batch_scaled[:, :, 2] = lab_batch[:, :, 2] * img_size
        lab_batch_scaled[:, :, 3] = lab_batch[:, :, 3] * img_size
        lab_batch_scaled[:, :, 4] = lab_batch[:, :, 4] * img_size

        target_size = torch.sqrt(((lab_batch_scaled[:, :, 3].mul(0.2)) ** 2) + ((lab_batch_scaled[:, :, 4].mul(0.2)) ** 2))
        target_x = lab_batch[:, :, 1].view(np.prod(batch_size))  # np.prod(batch_size) = 4*16 = 84
        target_y = lab_batch[:, :, 2].view(np.prod(batch_size))
        targetoff_x = lab_batch[:, :, 3].view(np.prod(batch_size))  # used to get off_x
        targetoff_y = lab_batch[:, :, 4].view(np.prod(batch_size))  # used to get off_y

        if(rand_loc):
            if use_cuda:
                off_x = targetoff_x*(torch.cuda.FloatTensor(targetoff_x.size()).uniform_(-0.4,0.4))
                off_y = targetoff_y*(torch.cuda.FloatTensor(targetoff_y.size()).uniform_(-0.4,0.4))
            else:
                off_x = targetoff_x * (torch.FloatTensor(targetoff_x.size()).uniform_(-0.4, 0.4))
                off_y = targetoff_y * (torch.FloatTensor(targetoff_y.size()).uniform_(-0.4, 0.4))

            target_x = target_x + off_x
            target_y = target_y + off_y

        target_y = target_y - 0.05

        scale = target_size / current_patch_size
        scale = scale.view(anglesize)

        s = adv_batch.size() # 6 x 14 x 3 x 416 x 416
        adv_batch = adv_batch.view(s[0] * s[1], s[2], s[3], s[4])  # 84 x 3 x 416 x 16
        msk_batch = msk_batch.view(s[0] * s[1], s[2], s[3], s[4])  # 84 x 3 x 416 x 16

        tx = (-target_x+0.5)*2
        ty = (-target_y+0.5)*2

        sin = torch.sin(angle)
        cos = torch.cos(angle)

        # Theta = rotation, rescale matrix
        if use_cuda:
            theta = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
        else:
            theta = torch.FloatTensor(anglesize, 2, 3).fill_(0) # dim 84 x 2 x 3 (N x 2 x 3) required by F.affine_grid

        theta[:, 0, 0] = cos/scale
        theta[:, 0, 1] = sin/scale
        theta[:, 0, 2] = tx*cos/scale+ty*sin/scale
        theta[:, 1, 0] = -sin/scale
        theta[:, 1, 1] = cos/scale
        theta[:, 1, 2] = -tx*sin/scale+ty*cos/scale

        grid = F.affine_grid(theta, adv_batch.shape)  # adv_batch should be of type N x C x Hin x Win. Output is N x Hg x Wg x 2

        adv_batch_t = F.grid_sample(adv_batch, grid)  # computes the output using input values and pixel locations from grid.
        msk_batch_t = F.grid_sample(msk_batch, grid)  # Output has dim N x C x Hg x Wg
        # print(adv_batch_t.size()) dim 84 x 3 x 416 x 416
        # print(msk_batch_t.size()) dim 84 x 3 x 416 x 416


        '''
        # Theta2 = translation matrix
        theta2 = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
        theta2[:, 0, 0] = 1
        theta2[:, 0, 1] = 0
        theta2[:, 0, 2] = (-target_x + 0.5) * 2
        theta2[:, 1, 0] = 0
        theta2[:, 1, 1] = 1
        theta2[:, 1, 2] = (-target_y + 0.5) * 2

        grid2 = F.affine_grid(theta2, adv_batch.shape)
        adv_batch_t = F.grid_sample(adv_batch_t, grid2)
        msk_batch_t = F.grid_sample(msk_batch_t, grid2)

        '''
        adv_batch_t = adv_batch_t.view(s[0], s[1], s[2], s[3], s[4]) # 4 x 16 x 3 x 416 x 416
        msk_batch_t = msk_batch_t.view(s[0], s[1], s[2], s[3], s[4])

        adv_batch_t = torch.clamp(adv_batch_t, 0.000001, 0.999999)
        #img = msk_batch_t[0, 0, :, :, :].detach().cpu()
        #img = transforms.ToPILImage()(img)
        #img.show()
        #exit()

        # print((adv_batch_t * msk_batch_t).size()) dim = 6 x 14 x 3 x 416 x 416

        return adv_batch_t * msk_batch_t  # It is as if I have passed adv_batch_t "filtered" by the mask itself

# NB output of PatchTransformer is the input of PatchApplier

class PatchApplier(nn.Module):
    """PatchApplier: applies adversarial patches to images.

    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.

    """

    def __init__(self):
        super(PatchApplier, self).__init__()

    def forward(self, img_batch, adv_batch):

        advs = torch.unbind(adv_batch, 1)  # Returns a tuple of all slices along a given dimension, already without it.
        # print(np.shape(advs)) # dim = (14,) --> it indicates TODO 14 copies of the adv patch: one for each detected person (random number)
        # plus the remaining to get a total = max_lab (i.e. 14)

        for adv in advs:
            img_batch = torch.where((adv == 0), img_batch, adv)  # the output tensor has elements belonging to img_batch if adv == 0, else belonging to adv
            # dim img_batch = 6 x 3 x 416 x 416

            # you put one after the other your 14 adv_patches on the image. When you meet those which are totally 0, i.e. those that do not
            # correspond to a detected object in the image, you keep your image as it is (do nothing). Otherwise, you will have your scaled, rotated and
            # well-positioned patch corresponding to one of the detected objects of the image. I think its pixels are 0s where there is not the object, and =/= 0
            # where there is the object, with appropriate affine properties. Here, you substitute imgage pixels with adv pixels.
            # At the end of the 14th cycle you have attached your patches to all detected regions of the image 'layer by layer', for all images in the batch (6).
        return img_batch


#TODO ____________________________________________________________________________________________________________________________________________________________
    # TODO Summary of PatchTransformer + PatchApplier:
    # take a batch of 6 images, consider one. For it, I have 14 ready adv patches, of which a number that varies for each image is non-zero (remember:
    # the mask is done starting from 0 and 1 labels in lab_batch. Suppose that 5 are non zero. It means that they correspond to 5 detected object in that image.
    # They are already transformed according to correct positions and scales of the 5 detected rectangles. Now, we consider the image of the six composing the batch,
    # and substitute the patches in their positions where they are not zero (so 5 out of 14 in this example)
#TODO ____________________________________________________________________________________________________________________________________________________________


class InriaDataset(Dataset):
    """InriaDataset: representation of the INRIA person dataset.

    Internal representation of the commonly used INRIA person dataset.
    Available at: http://pascal.inrialpes.fr/data/human/

    Attributes:
        len: An integer number of elements in the
        img_dir: Directory containing the images of the INRIA dataset.
        lab_dir: Directory containing the labels of the INRIA dataset.
        img_names: List of all image file names in img_dir.
        shuffle: Whether or not to shuffle the dataset.

    """

    def __init__(self, img_dir, lab_dir, max_lab, imgsize, shuffle=True):
        #imgsize = 416 from yolo
        n_png_images = len(fnmatch.filter(os.listdir(img_dir), '*.png'))
        n_jpg_images = len(fnmatch.filter(os.listdir(img_dir), '*.jpg'))
        n_images = n_png_images + n_jpg_images
        n_labels = len(fnmatch.filter(os.listdir(lab_dir), '*.txt'))
        assert n_images == n_labels, "Number of images and number of labels don't match"
        self.len = n_images
        self.img_dir = img_dir
        self.lab_dir = lab_dir
        self.imgsize = imgsize
        self.img_names = fnmatch.filter(os.listdir(img_dir), '*.png') + fnmatch.filter(os.listdir(img_dir), '*.jpg')
        self.shuffle = shuffle
        self.img_paths = []
        for img_name in self.img_names:
            self.img_paths.append(os.path.join(self.img_dir, img_name))
        self.lab_paths = []
        for img_name in self.img_names:
            lab_path = os.path.join(self.lab_dir, img_name).replace('.jpg', '.txt').replace('.png', '.txt')
            self.lab_paths.append(lab_path)
        self.max_n_labels = max_lab

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        lab_path = os.path.join(self.lab_dir, self.img_names[idx]).replace('.jpg', '.txt').replace('.png', '.txt')
        image = Image.open(img_path).convert('RGB')
        if os.path.getsize(lab_path):       #check to see if label file contains data. 
            label = np.loadtxt(lab_path)
        else:
            label = np.ones([5])

        label = torch.from_numpy(label).float()
        if label.dim() == 1:
            label = label.unsqueeze(0)

        image, label = self.pad_and_scale(image, label)
        transform = transforms.ToTensor()
        image = transform(image)
        label = self.pad_lab(label)  # to make it agrees with max_lab dimensions. We choose a max_lab to say: no more than 14 persons could stand in one picture
        return image, label

    def pad_and_scale(self, img, lab): # this method for taking a non-square image and make it square by filling the difference in w and h with gray
                                       # needed to keep proportions
        """

        Args:
            img:

        Returns:

        """
        w,h = img.size
        if w==h:
            padded_img = img
        else:
            dim_to_pad = 1 if w<h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h,h), color=(127,127,127))
                padded_img.paste(img, (int(padding), 0))
                lab[:, [1]] = (lab[:, [1]] * w + padding) / h
                lab[:, [3]] = (lab[:, [3]] * w / h)
            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(127,127,127))
                padded_img.paste(img, (0, int(padding)))
                lab[:, [2]] = (lab[:, [2]] * h + padding) / w
                lab[:, [4]] = (lab[:, [4]] * h  / w)
        resize = transforms.Resize((self.imgsize,self.imgsize)) # make a square image of dim 416 x 416
        padded_img = resize(padded_img)     #choose here
        return padded_img, lab

    def pad_lab(self, lab):
        pad_size = self.max_n_labels - lab.shape[0]
        if(pad_size>0):
            padded_lab = F.pad(lab, (0, 0, 0, pad_size), value=1)  # padding of the labels to have a pad_size = max_lab (14 here).
                                                                   # add 1s to make dimensions = max_lab x batch_size (14 x 6) after the images lines,
                                                                   # whose number is not known a priori
        else:
            padded_lab = lab
        return padded_lab
