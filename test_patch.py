import sys
import time
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from utils import *
from darknet import *
from load_data import PatchTransformer, PatchApplier, InriaDataset
import json


if __name__ == '__main__':
    print("Setting everything up")
    use_cuda = 0
    imgdir = "C:/Users/Alessandro/PycharmProjects/YOLOv2_d2p_adv_COCO_thys2019/inria/INRIAPerson/Test/pos"
    cfgfile = "C:/Users/Alessandro/PycharmProjects/YOLOv2_d2p_adv_COCO_thys2019/cfg/yolo.cfg"
    weightfile = "C:/Users/Alessandro/PycharmProjects/YOLOv2_d2p_adv_COCO_thys2019/weights/yolov2.weights"
    patchfile = "C:/Users/Alessandro/PycharmProjects/YOLOv2_d2p_adv_COCO_thys2019/saved_patches_my_trial/ale_cls.jpg"
    # patchfile = "/home/wvr/Pictures/individualImage_upper_body.png"
    # patchfile = "/home/wvr/Pictures/class_only.png"
    # patchfile = "/home/wvr/Pictures/class_transfer.png"
    savedir = "C:/Users/Alessandro/PycharmProjects/YOLOv2_d2p_adv_COCO_thys2019/test_results_mytrial/"

    darknet_model = Darknet(cfgfile)
    darknet_model.load_weights(weightfile)

    if use_cuda:
        darknet_model = darknet_model.eval().cuda()
        patch_applier = PatchApplier().cuda()
        patch_transformer = PatchTransformer().cuda()
    else:
        darknet_model = darknet_model.eval()
        patch_applier = PatchApplier()
        patch_transformer = PatchTransformer()

    batch_size = 1
    max_lab = 14
    img_size = darknet_model.height # 416

    patch_size = 300

    # open the learned patch image file
    patch_img = Image.open(patchfile).convert('RGB')
    tf = transforms.Resize((patch_size,patch_size))
    patch_img = tf(patch_img)
    tf = transforms.ToTensor()
    adv_patch_cpu = tf(patch_img)

    if use_cuda:
        adv_patch = adv_patch_cpu.cuda()
    else:
        adv_patch = adv_patch_cpu

    clean_results = []
    noise_results = []
    patch_results = []
    
    print("Done")
#TODO _______________________________________________________________________________________________________________________________
#TODO Cleaned images

    #Loop over cleaned images
    for imgfile in os.listdir(imgdir):
        print("new image")
        if imgfile.endswith('.jpg') or imgfile.endswith('.png'):
            name = os.path.splitext(imgfile)[0]    #image name w/o extension
            txtname = name + '.txt'
            #txtpath = os.path.abspath(os.path.join(savedir, "clean/", "yolo-labels", txtname))
            txtpath = os.path.join(savedir, "clean/", "yolo-labels/", txtname)

            # open image and adjust to yolo input size
            imgfile = os.path.abspath(os.path.join(imgdir, imgfile))
            img = Image.open(imgfile).convert('RGB')
            w,h = img.size
            if w==h:
                padded_img = img
            else:
                dim_to_pad = 1 if w<h else 2
                if dim_to_pad == 1:
                    padding = (h - w) / 2
                    padded_img = Image.new('RGB', (h,h), color=(127,127,127))
                    padded_img.paste(img, (int(padding), 0))
                else:
                    padding = (w - h) / 2
                    padded_img = Image.new('RGB', (w, w), color=(127,127,127))
                    padded_img.paste(img, (0, int(padding)))

            resize = transforms.Resize((img_size,img_size)) # resize image for yolo: 416 x 416
            padded_img = resize(padded_img)
            cleanname = name + ".png"

            #TODO save cleaned image
            padded_img.save(os.path.join(savedir, "clean/bef_detection", cleanname))
            #plt.imshow(padded_img)
            #plt.show()
            
            # generate a label file for the clean padded image after detection
            boxes = do_detect(darknet_model, padded_img, 0.4, 0.4, False)
            boxes = nms(boxes, 0.4)

            # # plot boxes on cleaned images
            #class_names = load_class_names('C:/Users/Alessandro/PycharmProjects/YOLOv2_d2p_adv_COCO_thys2019/data/coco.names')  # load_class_names in utils
            #plotted_image = plot_boxes(padded_img, boxes, class_names=class_names)  # plot_boxes in utils
            #plt.imshow(plotted_image)
            # plt.show()
            # plotted_image.save(os.path.join(savedir, 'clean/after_detection', cleanname))
            #print(boxes)
            textfile = open(txtpath,'w+')
            for box in boxes:
                #print(box)
                # print(box[0])
                # print(box[1])
                # print(box[2])
                # print(box[3])
                # print(box[4])
                # print(box[5])
                # print(box[6])
                cls_id = box[6]
                if(cls_id == 0):   #if person
                    x_center = box[0]
                    y_center = box[1]
                    width = box[2]
                    height = box[3]
                    textfile.write(f'{cls_id} {x_center} {y_center} {width} {height}\n')
                    clean_results.append({'image_id': name, 'bbox': [x_center.item() - width.item() / 2,
                                                                     y_center.item() - height.item() / 2,
                                                                     width.item(),
                                                                     height.item()],
                                          'score': box[4].item(),
                                          'category_id': 1})

            textfile.close()

            # read this label file back as a tensor
            if os.path.getsize(txtpath):       #check to see if label file contains data. 
                label = np.loadtxt(txtpath)
            else:
                label = np.ones([5])
            label = torch.from_numpy(label).float()
            if label.dim() == 1:
                label = label.unsqueeze(0)

            transform = transforms.ToTensor()
            if use_cuda:
                padded_img = transform(padded_img).cuda()
                img_fake_batch = padded_img.unsqueeze(0).cuda() # added by ale
                lab_fake_batch = label.unsqueeze(0).cuda()
            else:
                padded_img = transform(padded_img)
                img_fake_batch = padded_img.unsqueeze(0)  # added by ale
                lab_fake_batch = label.unsqueeze(0)
                #print('1:' + str(img_fake_batch.size()))
                #print('2:' + str(lab_fake_batch.size()))

#TODO Proper patch insertion

            #Transform the learned patch and apply it to images

            #print(adv_patch.size())
            #print(lab_fake_batch.size())

            adv_batch_t = patch_transformer(adv_patch, lab_fake_batch, img_size, do_rotate=True, rand_loc=False)
            p_img_batch = patch_applier(img_fake_batch, adv_batch_t)
            #print(p_img_batch)
            p_img = F.interpolate(p_img_batch, (darknet_model.height, darknet_model.width))
            #print(p_img)
            p_img = p_img.squeeze(0)
            p_img_pil = transforms.ToPILImage('RGB')(p_img.cpu())
            properpatchedname = name + "_p.png"
            p_img_pil.save(os.path.join(savedir, 'proper_patched/bef_detection', properpatchedname))
            # plt.imshow(p_img_pil)
            # plt.show()
            
            # generate a label file for the image with sticker
            txtname = properpatchedname.replace('.png', '.txt')
            #txtpath = os.path.abspath(os.path.join(savedir, 'proper_patched/', 'yolo-labels/', txtname))
            txtpath = os.path.join(savedir, "proper_patched/", "yolo-labels/", txtname)

            boxes = do_detect(darknet_model, p_img_pil, 0.8, 0.4, False)
            # NB do_detect (utils.py) takes the model as input, and calculates output(img) + get_region_boxes(output)
            # In the file adversarial_example.py, these tasks are performed separately

            # Perform non maximal suppression
            boxes = nms(boxes, 0.4)

            # plot boxes on proper patched images
            class_names = load_class_names('C:/Users/Alessandro/PycharmProjects/YOLOv2_d2p_adv_COCO_thys2019/data/coco.names')  # load_class_names in utils
            plotted_image_patch = plot_boxes(p_img_pil, boxes, class_names=class_names)  # plot_boxes in utils
            plotted_image_patch.save(os.path.join(savedir, 'proper_patched/after_detection', properpatchedname))
            #plt.imshow(plotted_image_patch)
            #plt.show()

            textfile = open(txtpath,'w+')
            for box in boxes:
                cls_id = box[6]
                if(cls_id == 0):   #if person
                    x_center = box[0]
                    y_center = box[1]
                    width = box[2]
                    height = box[3]
                    textfile.write(f'{cls_id} {x_center} {y_center} {width} {height}\n')
                    patch_results.append({'image_id': name, 'bbox': [x_center.item() - width.item() / 2, y_center.item() - height.item() / 2, width.item(), height.item()], 'score': box[4].item(), 'category_id': 1})
            textfile.close()
#TODO _________________________________________________________________________________________________________________________________________________________________________________________________________________________
# # TODO Random patch
#
            # Make a random patch, transform it and add it to the image
            if use_cuda:
                random_patch = torch.rand(adv_patch_cpu.size()).cuda()
            else:
                random_patch = torch.rand(adv_patch_cpu.size())

            adv_batch_t = patch_transformer(random_patch, lab_fake_batch, img_size, do_rotate=True, rand_loc=False)
            p_img_batch_rnd = patch_applier(img_fake_batch, adv_batch_t)
            p_img_rnd = p_img_batch_rnd.squeeze(0)
            p_img_pil_rnd = transforms.ToPILImage('RGB')(p_img_rnd.cpu())
            randompatchedname = name + "_rdp.png"
            p_img_pil_rnd.save(os.path.join(savedir, 'random_patched/bef_detection', randompatchedname))
            #plt.imshow(p_img_pil_rnd)
            #plt.show()

            # generate a label file for the random patch image
            txtname = randompatchedname.replace('.png', '.txt')
            #txtpath = os.path.abspath(os.path.join(savedir, 'random_patched/', 'yolo-labels/', txtname))
            txtpath = os.path.join(savedir, "random_patched/", "yolo-labels/", txtname)

            boxes = do_detect(darknet_model, p_img_pil_rnd, 0.8, 0.4, False)
            boxes = nms(boxes, 0.4)

            # plot boxes on proper patched images
            class_names = load_class_names('C:/Users/Alessandro/PycharmProjects/YOLOv2_d2p_adv_COCO_thys2019/data/coco.names')  # load_class_names in utils
            plotted_image_rnd = plot_boxes(p_img_pil_rnd, boxes, class_names=class_names)  # plot_boxes in utils
            plotted_image_rnd.save(os.path.join(savedir, 'random_patched/after_detection', randompatchedname))
            #plt.imshow(plotted_image_rnd)
            #plt.show()

            textfile = open(txtpath,'w+')
            for box in boxes:
                cls_id = box[6]
                if(cls_id == 0):   #if person
                    x_center = box[0]
                    y_center = box[1]
                    width = box[2]
                    height = box[3]
                    textfile.write(f'{cls_id} {x_center} {y_center} {width} {height}\n')
                    noise_results.append({'image_id': name, 'bbox': [x_center.item() - width.item() / 2, y_center.item() - height.item() / 2, width.item(), height.item()], 'score': box[4].item(), 'category_id': 1})
            textfile.close()

    with open('clean_results.json', 'w') as fp:
        json.dump(clean_results, fp)
    with open('noise_results.json', 'w') as fp:
        json.dump(noise_results, fp)
    with open('patch_results.json', 'w') as fp:
        json.dump(patch_results, fp)
            

