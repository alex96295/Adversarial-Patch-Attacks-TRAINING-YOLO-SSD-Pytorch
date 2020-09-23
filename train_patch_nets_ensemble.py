"""
Training code for Adversarial patch training


"""

import PIL
import load_data
from tqdm import tqdm
from darknet_yolov3 import *
from darknet import *
from load_data import *
import gc
import matplotlib.pyplot as plt
from torch import autograd
from torchvision import transforms
from torch.autograd import Variable
# from tensorboardX import SummaryWriter
import subprocess

from utilsv4.utils import *
from toolv4.darknet2pytorch import YOLOv4


import patch_config
import sys
import time
import os
#from lib_ssd.modeling.model_builder import create_model
#from lib_ssd.utils.config_parse import cfg, cfg_from_file
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.ssd.vgg_ssd import create_vgg_ssd

from vision.ssd.config import mobilenetv1_ssd_config, vgg_ssd_config

if __name__ == '__main__':

    class PatchTrainer(object):

        #os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
        torch.cuda.set_device(0)

        def __init__(self, mode):

            self.config = patch_config.patch_configs[mode]()  # select the mode for the patch

            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            print(torch.cuda.device_count())

            #self.mbntv2_ssdlite_model, self.priorbox = create_model(self.cfgfile_ssds.MODEL) # COCO
            #self.priors = Variable(self.priorbox.forward(), volatile=True) # num_priors = grid x grid x num_anchors

            # yolov2
            #self.darknet_model_yolov2 = Darknet(self.config.cfgfile_yolov2)
            #self.darknet_model_yolov2.load_weights(self.config.weightfile_yolov2)

            # yolov3
            #self.darknet_model_yolov3 = yolov3(self.config.cfgfile_yolov3)
            #load_darknet_weights(self.darknet_model_yolov3, self.config.weightfile_yolov3)
            
            # yolov4
            self.darknet_model_yolov4 = YOLOv4(self.config.cfgfile_yolov4)
            self.darknet_model_yolov4.load_weights(self.config.weightfile_yolov4)

            #mobilenetv1 + ssd
            #self.mbntv1_ssd_model = create_mobilenetv1_ssd(21, is_test=True) # VOC
            #self.mbntv1_ssd_model.load(self.config.ssdmbntv1_model_path)

            #mobilenetv2 + ssdlite
            self.mbntv2_ssdlite_model = create_mobilenetv2_ssd_lite(21, is_test=True) # VOC
            self.mbntv2_ssdlite_model.load(self.config.ssdlitembntv2_model_path)

            # vgg + ssd
            #self.vgg_ssd_model = create_vgg_ssd(21, is_test=True)  # VOC
            #self.vgg_ssd_model.load(self.config.ssdvgg_model_path)

            if use_cuda:
                #self.darknet_model_yolov2 = self.darknet_model_yolov2.eval().to(self.device)  # Why eval? test!
                #self.darknet_model_yolov3 = self.darknet_model_yolov3.eval().to(self.device)
                self.darknet_model_yolov4 = self.darknet_model_yolov4.eval().to(self.device)
                #self.mbntv1_ssd_model = self.mbntv1_ssd_model.eval().to(self.device)
                self.mbntv2_ssdlite_model = self.mbntv2_ssdlite_model.eval().to(self.device)
                #self.vgg_ssd_model = self.vgg_ssd_model.eval().to(self.device)

                self.patch_applier = PatchApplier().to(self.device)
                self.patch_transformer = PatchTransformer().to(self.device)

                #self.score_extractor_yolov2 = yolov2_feature_output_manage(0, 80, self.config).to(self.device)
                #self.score_extractor_yolov3 = yolov3_feature_output_manage(0, 80, self.config).to(self.device)
                self.score_extractor_yolov4 = yolov4_feature_output_manage(0, 80, self.config).to(self.device)
                self.score_extractor_ssd = ssd_feature_output_manage(15, 21, self.config).to(self.device) # 15 is person class in VOC (with 21 elements)

                self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_size).to(self.device)
                self.total_variation = TotalVariation().to(self.device)
            else:
                #self.darknet_model_yolov2 = self.darknet_model_yolov2.eval()  # Why eval? test!
                #self.darknet_model_yolov3 = self.darknet_model_yolov3.eval()
                self.darknet_model_yolov4 = self.darknet_model_yolov4.eval().to(self.device)
                #self.mbntv1_ssd_model = self.mbntv1_ssd_model.eval()
                self.mbntv2_ssdlite_model = self.mbntv2_ssdlite_model.eval()
                #self.vgg_ssd_model = self.vgg_ssd_model.eval()

                self.patch_applier = PatchApplier()
                self.patch_transformer = PatchTransformer()

                #self.score_extractor_yolov2 = yolov2_feature_output_manage(0, 80, self.config)
                #self.score_extractor_yolov3 = yolov3_feature_output_manage(0, 80, self.config)
                self.score_extractor_yolov4 = yolov4_feature_output_manage(0, 80, self.config)
                self.score_extractor_ssd = ssd_feature_output_manage(15, 21, self.config) # 15 is person class in VOC (with 21 elements)

                self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_size)
                self.total_variation = TotalVariation()

        # __________________________________________________________________________________________________________________________________-
        #     self.writer = self.init_tensorboard(mode)

        # def init_tensorboard(self, name=None):
        #     subprocess.Popen(['tensorboard', '--logdir=runs'])
        #     if name is not None:
        #         time_str = time.strftime("%Y%m%d-%H%M%S")
        #         return SummaryWriter(f'runs/{time_str}_{name}')
        #     else:
        #         return SummaryWriter()
        # ___________________________________________________________________________________________________________________________________
        def train(self):

            """
            Optimize a patch to generate an adversarial example.
            :return: Nothing
            """
            destination_path = "./"
            destination_name = 'loss_tracking_ens_yv4mbntv2lite_obj_max_mean.txt'
            destination_name2 = 'loss_tracking_ens_yv4mbntv2lite_obj_max_mean_compact_batch.txt'
            destination_name3 = 'loss_tracking_ens_yv4mbntv2lite_obj_max_mean_compact_epochs.txt'
            destination = os.path.join(destination_path, destination_name)
            destination2 = os.path.join(destination_path, destination_name2)
            destination3 = os.path.join(destination_path, destination_name3)
            textfile = open(destination, 'w+')
            textfile2 = open(destination2, 'w+')
            textfile3 = open(destination3, 'w+')
            
            img_size_init = 500
            img_size_yolo = 416  # 416 for yolo family
            img_size_ssd = mobilenetv1_ssd_config.image_size # default 300, changed to 416 for ensemble training!

            batch_size = self.config.batch_size
            n_epochs = 600
            max_lab = 14

            # Generate starting point
            adv_patch_cpu = self.generate_patch("gray")
            # adv_patch_cpu = self.read_image("saved_patches/patchnew0.jpg")

            adv_patch_cpu.requires_grad_(True)

            train_loader = torch.utils.data.DataLoader(
                InriaDataset(self.config.img_dir, self.config.lab_dir, max_lab, img_size_init,
                             shuffle=True),
                batch_size=batch_size,
                shuffle=True,
                num_workers=10)

            # NB: now the dataset has images correctly padded for patch application and of size initialized to yolo suitable one: 416

            self.epoch_length = len(train_loader)
            print(f'One epoch is {len(train_loader)}')

            optimizer = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate,
                                   amsgrad=True)  # starting lr = 0.03
            scheduler = self.config.scheduler_factory(optimizer)
            # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50) # write it directly

            et0 = time.time()  # epoch start
            for epoch in range(n_epochs):
                ep_det_loss = 0
                ep_nps_loss = 0
                ep_tv_loss = 0
                ep_loss = 0
                bt0 = time.time()  # batch start
                for i_batch, (img_batch, lab_batch) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}',
                                                            total=self.epoch_length):
                    with autograd.detect_anomaly():

                        if use_cuda:
                            img_batch = img_batch.to(self.device)
                            lab_batch = lab_batch.to(self.device)
                            adv_patch = adv_patch_cpu.to(self.device)
                        else:
                            img_batch = img_batch
                            lab_batch = lab_batch
                            adv_patch = adv_patch_cpu

                        adv_batch_t = self.patch_transformer(adv_patch, lab_batch, img_size_init, do_rotate=True, rand_loc=False)
                        p_img_batch = self.patch_applier(img_batch, adv_batch_t)

                        # resize to correct dimensions in order to feed the detectors:
                        p_img_batch_yolo = F.interpolate(p_img_batch, (img_size_yolo, img_size_yolo)) # yolo_family
                        p_img_batch_ssd = F.interpolate(p_img_batch, (img_size_ssd, img_size_ssd)) # ssd_family

                        # calculate the output
                        #output_yolov2 = self.darknet_model_yolov2(p_img_batch_yolo) # yolo_family
                        #output_yolov3 = self.darknet_model_yolov3(p_img_batch_yolo)
                        output_yolov4 = self.darknet_model_yolov4(p_img_batch_yolo)

                        #output_ssd_mbntv1 = self.mbntv1_ssd_model(p_img_batch_ssd) # ssd family
                        output_ssdlite_mbntv2 = self.mbntv2_ssdlite_model(p_img_batch_ssd) # ssd family
                        #output_ssd_vgg = self.vgg_ssd_model(p_img_batch_ssd) # ssd family

                        # loss calculation, three contributions: detection loss, tv loss, nps loss

                        # detection loss, single networks:
                        # METHOD 1) maximum score extraction approach
                        # METHOD 2) threshold approach

                        loss_type = 'max_approach'

                        #score_yolov2 = self.score_extractor_yolov2(output_yolov2, loss_type)
                        #score_yolov3 = self.score_extractor_yolov3(output_yolov3, loss_type)
                        score_yolov4 = self.score_extractor_yolov4(output_yolov4, loss_type)
                        #score_ssd_mbntv1 = self.score_extractor_ssd(output_ssd_mbntv1, loss_type)
                        score_ssdlite_mbntv2 = self.score_extractor_ssd(output_ssdlite_mbntv2, loss_type)
                        #score_ssd_vgg = self.prob_extractor_ssd(output_ssd_vgg)

                        scores_ensemble = torch.stack([score_yolov4, score_ssdlite_mbntv2], dim=0)
                        #print(scores_ensemble)

                        # detection loss, networks interface:
                        # METHOD 1) sum over ensemble scores
                        # METHOD 2) mean over ensemble scores
                        # METHOD 3) max over ensemble scores
                        
                        ensemble_op = 'ensemble_mean'
                        det_loss_ens = self.ensemble_op(scores_ensemble, ensemble_op)
                        #print(det_loss_ens)

                        # manage the batch of images: mean, max?
                        det_loss = torch.mean(det_loss_ens)
                        #print(det_loss)

                        nps = self.nps_calculator(adv_patch)
                        tv = self.total_variation(adv_patch)
                        nps_loss = nps * 0.01
                        tv_loss = tv * 2.5

                        if use_cuda:
                            loss = det_loss + nps_loss + torch.max(tv_loss, torch.tensor(0.1).to(self.device))
                        else:
                            loss = det_loss + nps_loss + torch.max(tv_loss, torch.tensor(0.1))

                        ep_det_loss += det_loss.detach().cpu().numpy() / len(train_loader)
                        ep_nps_loss += nps_loss.detach().cpu().numpy()
                        ep_tv_loss += tv_loss.detach().cpu().numpy()
                        ep_loss += loss

                        # Optimization step + backward
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        adv_patch_cpu.data.clamp_(0, 1)  # keep patch in image range

                        bt1 = time.time()  # batch end
                        if i_batch % 1 == 0:
                            # Plot the adversarial patch in learning phase during one epoch for each batch (remember one batch = 6 images, around 100 batches in tot)
                            im = transforms.ToPILImage('RGB')(adv_patch_cpu)
                            # plt.imshow(im)
                            # plt.show()

                            # Plot the adv patch in learning phase during one epoch applied on one image of the six composing a single batch.
                            # In total there are 100 batches, i.e. 6 images are picked for 100 times, and this is one epoch. In total, there are 10000 epochs.
                            # img = p_img_batch[1, :, :, ]
                            # img = transforms.ToPILImage()(img.detach().cpu())
                            # img.show()

                            #iteration = self.epoch_length * epoch + i_batch

                            print('  BATCH NR: ', i_batch)
                            print('BATCH LOSS: ', loss)  # .detach().cpu().numpy())
                            print('  DET LOSS: ', det_loss)  # .detach().cpu().numpy())
                            print('  NPS LOSS: ', nps_loss)  # .detach().cpu().numpy())
                            print('   TV LOSS: ', tv_loss)  # .detach().cpu().numpy())
                            print('BATCH TIME: ', bt1 - bt0)

                            # self.writer.add_scalar('total_loss', loss.detach().cpu().numpy(), iteration)
                            # self.writer.add_scalar('loss/det_loss', det_loss.detach().cpu().numpy(), iteration)
                            # self.writer.add_scalar('loss/nps_loss', nps_loss.detach().cpu().numpy(), iteration)
                            # self.writer.add_scalar('loss/tv_loss', tv_loss.detach().cpu().numpy(), iteration)
                            # self.writer.add_scalar('misc/epoch', epoch, iteration)
                            # self.writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)
                            # self.writer.add_scalar('batch_time', bt1-bt0, iteration)

                            # self.writer.add_image('patch', adv_patch_cpu, iteration)
                            
                            textfile.write(f'i_batch: {i_batch}\nb_tot_loss:{loss}\nb_det_loss: {det_loss}\nb_nps_loss: {nps_loss}\nb_TV_loss: {tv_loss}\n\n')
                            textfile2.write(f'{i_batch} {loss} {det_loss} {nps_loss} {tv_loss}\n')

                        if i_batch + 1 >= len(train_loader):
                            print('\n')
                        else:
                            del adv_batch_t, output_yolov4, output_ssdlite_mbntv2, score_yolov4, score_ssdlite_mbntv2, scores_ensemble, det_loss, p_img_batch, nps_loss, tv_loss, loss

                            if use_cuda:
                                torch.cuda.empty_cache()

                        bt0 = time.time()

                et1 = time.time()  # epoch end

                ep_det_loss = ep_det_loss / len(train_loader)
                ep_nps_loss = ep_nps_loss / len(train_loader)
                ep_tv_loss = ep_tv_loss / len(train_loader)
                ep_loss = ep_loss / len(train_loader)

                scheduler.step(ep_loss)

                if True:
                    print('  EPOCH NR: ', epoch),
                    print('EPOCH LOSS: ', ep_loss)
                    print('  DET LOSS: ', ep_det_loss)
                    print('  NPS LOSS: ', ep_nps_loss)
                    print('   TV LOSS: ', ep_tv_loss)
                    print('EPOCH TIME: ', et1 - et0)

                    # Plot the final adv_patch (learned) and save it
                    im = transforms.ToPILImage('RGB')(adv_patch_cpu)
                    # plt.imshow(im)
                    # plt.show()
                    im.save("./saved_patches_mytrial/net_ensemble_yv4_ssdlitembntv2_objobjcls_max_mean.jpg")
                    
                    textfile.write(f'\ni_epoch: {epoch}\ne_total_loss:{ep_loss}\ne_det_loss: {ep_det_loss}\ne_nps_loss: {ep_nps_loss}\ne_TV_loss: {ep_tv_loss}\n\n')
                    textfile3.write(f'{epoch} {ep_loss} {ep_det_loss} {ep_nps_loss} {ep_tv_loss}\n')

                    del adv_batch_t, output_yolov4, score_yolov4, score_ssdlite_mbntv2, output_ssdlite_mbntv2, scores_ensemble, det_loss, p_img_batch, nps_loss, tv_loss, loss

                    if use_cuda:
                        torch.cuda.empty_cache()

                et0 = time.time()

        # TODO __________________________________________________________________________________________________________________________________________________

        def generate_patch(self, type):
            """
            Generate a random patch as a starting point for optimization.

            :param type: Can be 'gray' or 'random'. Whether or not generate a gray or a random patch.
            :return:
            """
            if type == 'gray':
                adv_patch_cpu = torch.full((3, self.config.patch_size, self.config.patch_size), 0.5)
            elif type == 'random':
                adv_patch_cpu = torch.rand((3, self.config.patch_size, self.config.patch_size))

            return adv_patch_cpu

        def read_image(self, path):
            """
            Read an input image to be used as a patch

            :param path: Path to the image to be read.
            :return: Returns the transformed patch as a pytorch Tensor.
            """
            patch_img = Image.open(path).convert('RGB')
            tf = transforms.Resize((self.config.patch_size, self.config.patch_size))
            patch_img = tf(patch_img)
            tf = transforms.ToTensor()

            adv_patch_cpu = tf(patch_img)
            return adv_patch_cpu

        def ensemble_op(self, ensemble_score, ensemble_op):

            if ensemble_op == 'ensemble_sum':
                ensemble_res = torch.sum(ensemble_score, axis=0)

            elif ensemble_op == 'ensemble_mean':
                ensemble_res = ensemble_score.mean(dim=0)

            elif ensemble_op == 'ensemble_max':
                ensemble_res, ensemble_res_idx = torch.max(ensemble_score, dim=0)

            return ensemble_res





    # def main():
    #     if len(sys.argv) != 2:
    #         print('You need to supply (only) a configuration mode.')
    #         print('Possible modes are:')
    #         print(patch_config.patch_configs)
    #
    #

    use_cuda = 1
    trainer = PatchTrainer('paper_obj')
    trainer.train()





