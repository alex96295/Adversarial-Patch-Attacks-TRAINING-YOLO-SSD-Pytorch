"""
Training code for Adversarial patch training


"""

import PIL
#import load_data
from tqdm import tqdm

from load_data import *
#import gc
import matplotlib.pyplot as plt
from torch import autograd
from torchvision import transforms
# from tensorboardX import SummaryWriter
#import subprocess
from darknet_yolov3 import *
import patch_config
#import sys
import time
import yaml
import os

if __name__ == '__main__':

    class PatchTrainer(object):

        os.environ['CUDA_VISIBLE_DEVICES'] = '1'

        def __init__(self, mode):

            self.config = patch_config.patch_configs[mode]()  # select the mode for the patch

            #self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            self.device = 'cpu'
            print(torch.cuda.device_count())

            # cfg = self.config.cfgfile_yolov3
            # with open(cfg, 'r') as f:
            #     self.cfg = yaml.load(f)

            self.darknet_model = yolov3(self.config.cfgfile_yolov3, 416)
            #load_darknet_weights(self.darknet_model, self.config.weightfile_yolov3)
            self.darknet_model.load_state_dict(torch.load(self.config.weightfile_yolov3pt_ultra)['model'])

            if use_cuda:
                self.darknet_model = self.darknet_model.eval().to(self.device)  # TODO: Why eval? test!
                self.patch_applier = PatchApplier().to(self.device)
                self.patch_transformer = PatchTransformer().to(self.device)
                self.prob_extractor = yolov3_feature_output_manage(0, 80, self.config).to(self.device)
                self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_size).to(self.device)
                self.total_variation = TotalVariation().to(self.device)
            else:
                self.darknet_model = self.darknet_model.eval()  # TODO: Why eval?
                self.patch_applier = PatchApplier()
                self.patch_transformer = PatchTransformer()
                self.prob_extractor = yolov3_feature_output_manage(0, 80, self.config)
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

            img_size = 416  # 416 for this yolov3

            # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # self.darknet_model = torch.nn.DataParallel(self.darknet_model)
            # self.darknet_model.to(self.device)

            destination_path = "./"
            destination_name = 'loss_tracking_yv3_ultra_obj_noepoch.txt'
            destination_name2 = 'loss_tracking_compact_batch_yv3_ultra_obj_noepoch.txt'
            destination_name3 = 'loss_tracking_compatc_epochs_yv3_ultra_obj_noepoch.txt'
            destination = os.path.join(destination_path, destination_name)
            destination2 = os.path.join(destination_path, destination_name2)
            destination3 = os.path.join(destination_path, destination_name3)
            textfile = open(destination, 'w+')
            textfile2 = open(destination2, 'w+')
            textfile3 = open(destination3, 'w+')
            

            batch_size = self.config.batch_size
            n_epochs = 1000
            max_lab = 14

            #time_str = time.strftime("%Y%m%d-%H%M%S")

            # Generate starting point
            adv_patch_cpu = self.generate_patch("gray")
            # adv_patch_cpu = self.read_image("saved_patches/patchnew0.jpg")

            #adv_patch_cpu.requires_grad_(True)
            adv_patch_cpu = autograd.Variable(adv_patch_cpu, requires_grad=True)

            train_loader = torch.utils.data.DataLoader(
                InriaDataset(self.config.img_dir, self.config.lab_dir, max_lab, img_size,
                             shuffle=True),
                batch_size=batch_size,
                shuffle=True,
                num_workers=10)
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

                            adv_batch_t = self.patch_transformer(adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=False)
                            p_img_batch = self.patch_applier(img_batch, adv_batch_t)
                            p_img_batch = F.interpolate(p_img_batch, (img_size, img_size))
                            #p_img_batch = torch.autograd.Variable(p_img_batch, requires_grad=True)
                            # im = transforms.ToPILImage('RGB')(p_img_batch[0])
                            # plt.imshow(im)
                            # plt.show()

                            #CUDA = torch.cuda.is_available()

                            output = self.darknet_model.forward(p_img_batch)  # TODO apply YOLOv2 to all (patched) images in the batch (6)

                            loss_type = 'max_approach'

                            score_yolov3 = self.prob_extractor(output, loss_type)

                            nps = self.nps_calculator(adv_patch)
                            tv = self.total_variation(adv_patch)

                            nps_loss = nps * 0.01
                            tv_loss = tv * 2.5
                            det_loss = torch.mean(score_yolov3)

                            if use_cuda:
                                loss = det_loss + nps_loss + torch.max(tv_loss, torch.tensor(0.1).to(self.device))
                            else:
                                loss = det_loss + nps_loss + torch.max(tv_loss, torch.tensor(0.1))

                            ep_det_loss += det_loss.detach().cpu().numpy() / len(train_loader)
                            ep_nps_loss += nps_loss.detach().cpu().numpy()
                            ep_tv_loss += tv_loss.detach().cpu().numpy()
                            ep_loss += loss

                            # Optimization step + backward
                            #loss.retain_grad()
                            loss.backward()
                            #print("Gradients:\n" + str(torch.autograd.grad(loss, p_img_batch, retain_graph=True, allow_unused=True)))
                            #print("Gradients \n"+str(loss.grad))
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

                                textfile2.write(f'{i_batch} {loss} {det_loss} {nps_loss} {tv_loss}\n')

                                # self.writer.add_scalar('total_loss', loss.detach().cpu().numpy(), iteration)
                                # self.writer.add_scalar('loss/det_loss', det_loss.detach().cpu().numpy(), iteration)
                                # self.writer.add_scalar('loss/nps_loss', nps_loss.detach().cpu().numpy(), iteration)
                                # self.writer.add_scalar('loss/tv_loss', tv_loss.detach().cpu().numpy(), iteration)
                                # self.writer.add_scalar('misc/epoch', epoch, iteration)
                                # self.writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)
                                # self.writer.add_scalar('batch_time', bt1-bt0, iteration)

                                # self.writer.add_image('patch', adv_patch_cpu, iteration)

                            if i_batch + 1 >= len(train_loader):
                                print('\n')
                            else:
                                del adv_batch_t, output, score_yolov3, det_loss, p_img_batch, nps_loss, tv_loss, loss

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

                    textfile.write(f'\ni_epoch: {epoch}\ne_total_loss:{ep_loss}\ne_det_loss: {ep_det_loss}\ne_nps_loss: {ep_nps_loss}\ne_TV_loss: {ep_tv_loss}\n\n')
                    textfile3.write(f'{ep_loss} {ep_det_loss} {ep_nps_loss} {ep_tv_loss}\n')

                    # Plot the final adv_patch (learned) and save it
                    im = transforms.ToPILImage('RGB')(adv_patch_cpu)
                    #plt.imshow(im)
                    #plt.show()
                    im.save("./saved_patches_mytrial/yolov3_ultralytics_obj.png")

                    del adv_batch_t, output, score_yolov3, det_loss, p_img_batch, nps_loss, tv_loss, loss

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





