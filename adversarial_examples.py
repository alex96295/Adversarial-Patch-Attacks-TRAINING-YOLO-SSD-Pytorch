import PIL
import matplotlib.pyplot as plt
from utils import *
from load_data import *
from torch import autograd
plt.rcParams["axes.grid"] = False
plt.axis('off')
from patch_config import *

if __name__ == "__main__":  # to avoid multiprocessing on Windows

    # print(PIL.PILLOW_VERSION)

    # img_dir = "C:/Users/Alessandro/PycharmProjects/YOLOv2_d2p_adv_COCO_thys2019/inria/INRIAPerson/Train/pos"
    # lab_dir = "C:/Users/Alessandro/PycharmProjects/YOLOv2_d2p_adv_COCO_thys2019/inria/INRIAPerson/Train/pos/yolo-labels"
    # cfgfile = "C:/Users/Alessandro/PycharmProjects/YOLOv2_d2p_adv_COCO_thys2019/cfg/yolo.cfg"
    # weightfile = "C:/Users/Alessandro/PycharmProjects/YOLOv2_d2p_adv_COCO_thys2019/weights/yolov2.weights"
    # printfile = "C:/Users/Alessandro/PycharmProjects/YOLOv2_d2p_adv_COCO_thys2019/non_printability/30values.txt"
    
    img_dir = "/scratch/msc20f3/master_thesis/inria/INRIAPerson/Train/pos"
    lab_dir = "/scratch/msc20f3/master_thesis/inria/INRIAPerson/Train/pos/yolo-labels"
    cfgfile = "/scratch/msc20f3/master_thesis/cfg/yolo.cfg"
    weightfile = "/scratch/msc20f3/master_thesis/weights/yolov2.weights"
    printfile = "/scratch/msc20f3/master_thesis/non_printability/30values.txt"

    patch_size = 300  # set the patch size dimension(s)

    # Load the models (Darknet, weights, patch configuration file)

    print('LOADING MODELS')
    darknet_model = Darknet(cfgfile)
    darknet_model.load_weights(weightfile)
    base_config = BaseConfig()

    use_cuda = 1
    if use_cuda:
        darknet_model = darknet_model.eval().cuda()
        patch_applier = PatchApplier().cuda()
        patch_transformer = PatchTransformer().cuda()
        prob_extractor = MaxProbExtractor(0, 80, base_config).cuda()
        nps_calculator = NPSCalculator(printfile, patch_size).cuda()
        total_variation = TotalVariation().cuda()
    else:

        # Also as a rule of thumb for programming in general, try to explicitly state your intent and set model.train() and model.eval() when necessary.
        darknet_model = darknet_model.eval() #  evaluate the model

        patch_transformer = PatchTransformer()  # apply transformations to the patch
        patch_applier = PatchApplier()  # apply the patch to the testing image preserving its original patch_sizeXpatch_size dimensions

        # Module providing the functionality necessary to extract the max class probability for one class from YOLO output.
        prob_extractor = MaxProbExtractor(0, 80, base_config)

        # Module providing the functionality necessary to calculate the non-printability score (NMS) of an adversarial patch.
        nps_calculator = NPSCalculator(printfile, patch_size)

        # Module providing the functionality necessary to calculate the total vatiation (TV) of an adversarial patch.
        total_variation = TotalVariation()

    print('MODELS LOADED')

    # Image patch loading and initialisation settings

    img_size = darknet_model.height  # set the input image (not patch image) dimensions to those indicated in the cfg/yolo.cfg file (416)
    batch_size = 6#6#10#18#20
    n_epochs = 1000
    max_lab = 14

    # Patch initialisation, two possibilities:
        # 1. Random patch initialised at grey or random colors
        # 2. Specific image from path

    # 1. Choose between initializing with gray or random
    adv_patch_cpu = torch.full((3,patch_size,patch_size),0.5)  # initialize a (random) tensor of size 3 X patch_size X patch_size filled with 0.5 (gray)
    # im = transforms.ToPILImage('RGB')(adv_patch_cpu)
    # plt.imshow(im)
    # plt.show()


    # adv_patch_cpu = torch.rand((3,patch_size,patch_size)) # initialize a (random) tensor of size 3 X patch_size X patch_size filled with random colors

    # 2. Load the un-learned candidate patch image
    # patch_img = Image.open("C:/Users/Alessandro/PycharmProjects/YOLOv2_d2p_adv_COCO_thys2019/saved_patches/patchnew0.jpg").convert('RGB')    # convert to RGB format
    # tf = transforms.Resize((patch_size, patch_size))
    # patch_img = tf(patch_img)  # resize the candidate patch_image to have dimensions patch_size X patch_size
    # tf = transforms.ToTensor()  # from (H x W x C) with pixels in the range 0-255 to (C x H x W) in the range 0-1
    # adv_patch_cpu = tf(patch_img)  # Convert the PIL patch image (resized) to tensor
    #print(adv_patch_cpu)

    print('adv_patch_cpu size:' + str(adv_patch_cpu.size()))
    # adv_patch_cpu size:torch.Size([3, 300, 300])

#TODO ____________________________________________________________________________________________________________________________________________________________

# TRAIN: train the patch to fool the network

    adv_patch_cpu.requires_grad_(True)  # let the parameters of the patch to be optimized by the adv backward pass

    # Dataloader initialization: training image set, training label set, batch_size etc...
    print('INITIALIZING DATALOADER')
    train_loader = torch.utils.data.DataLoader(InriaDataset(img_dir, lab_dir, max_lab, img_size, shuffle=True),
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=10)

    # train_loader2 = InriaDataset(img_dir, lab_dir, max_lab, img_size, shuffle=True)
    #
    # patched_im_store = torch.full((len(train_loader2), 3, img_size, img_size),0)
    # adv_train_lab_store = torch.full((len(train_loader2), max_lab, 5),0)
    #
    # for i, (immagine, label) in enumerate(train_loader2):
    #     print(i)
    #     adv_transformed = patch_transformer(adv_patch_cpu, label, img_size, do_rotate=True, rand_loc=False)
    #     #print(adv_transformed.size())
    #     patched_im_store[0] = patch_applier(immagine, adv_transformed)
    #     adv_train_lab_store[0] = label
    #     i = i + 1
    #
    # patched_im_store = F.interpolate(patched_im_store, (darknet_model.height, darknet_model.width))
    # print(patched_im_store.size())
    # print(adv_train_lab_store.size())

    # DataLoader guide: https://pytorch.org/docs/1.1.0/_modules/torch/utils/data/dataloader.html
    print('DATALOADER INITIALIZED')
    # print(type(adv_patch_cpu))
    # print(type([adv_patch_cpu]))
    # Set the optimizer for the backward pass (Adam)
    #print(type([adv_patch_cpu]))
    #print([adv_patch_cpu])
    optimizer = optim.Adam([adv_patch_cpu], lr=.03, amsgrad=True)

    et0 = time.time()
    for epoch in range(n_epochs):
        ep_det_loss = 0
        ep_nps_loss = 0
        ep_tv_loss = 0
        ep_loss = 0
        bt0 = time.time()
        for i_batch, (img_batch, lab_batch) in enumerate(train_loader):

            print('img_batch size:' + str(img_batch.size()))
            print('lab_batch size:' + str(lab_batch.size()))

            im = transforms.ToPILImage('RGB')(img_batch[0])
            # plt.imshow(im)
            # plt.show()

            # img_batch size:torch.Size([6, 3, 416, 416])  --> batch_size = 6
            # lab_batch size:torch.Size([6, 14, 5])  --> batch_size = 6, max_label = 14

            # enumerate each (image, label) tuple in train_loader, which contains images/labels directories.
            # i_batch counts for the tuple number in the enumerated list, starting from 0

            # Running the forward pass with detection enabled will allow the backward pass to print
            # the traceback of the forward operation that created the failing backward function.
            # Any backward computation that generate “nan” value will raise an error.

            with autograd.detect_anomaly():

                if use_cuda:
                    img_batch = img_batch.cuda()
                    lab_batch = lab_batch.cuda()
                    #print('TRAINING EPOCH %i, BATCH %i'%(epoch, i_batch))
                    adv_patch = adv_patch_cpu.cuda()
                else:
                    img_batch = img_batch
                    lab_batch = lab_batch
                    adv_patch = adv_patch_cpu

                adv_batch_t = patch_transformer(adv_patch, lab_batch, img_size, do_rotate=True)  # take the patch (3 x 300 x 300) and apply transformations
                # im = transforms.ToPILImage('RGB')(adv_batch_t[0,0])
                # plt.imshow(im)
                # plt.show()

                p_img_batch = patch_applier(img_batch, adv_batch_t)  # take the transformed patch and apply it to all the detected objects in each image of the batch (without scaling I think)
                im = transforms.ToPILImage('RGB')(p_img_batch[0].cpu())
                # plt.imshow(im)
                # plt.show()

                # Down/up samples according to the given size of the input image, or its scale
                p_img_batch = F.interpolate(p_img_batch,(darknet_model.height, darknet_model.width))

                # Take the batch of images (6) that have the patch applied on every detected object and fill it into the yolov2 model to be processed again
                output = darknet_model(p_img_batch)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! input dim batch x 3 x 416 x 416
                # print(output.size())
                # print(type(output))
                #  Calculate max_prob, nps and tv (used for the loss) after yolov2 detection with the patch_img_batch analyzed:
                #  have the object been detected as well after patch insertion? The answer is yes at the beginning with un-learned patch and increasingly
                #  no after patch is learned to minimize its associated loss

                max_prob = prob_extractor(output)
                nps = nps_calculator(adv_patch)
                tv = total_variation(adv_patch)
                # print('max ' + str(max_prob.size()))
                # print('nps ' + str(nps.size()))
                # print('tv ' + str(tv.size()))

                det_loss = torch.mean(max_prob)
                ep_det_loss += det_loss.detach().cpu().numpy()

                #  .detach(): In order to enable automatic differentiation, PyTorch keeps track of all operations involving tensors
                #  for which the gradient may need to be computed (i.e., require_grad is True). The operations are recorded as a directed graph.
                #  The detach() method constructs a new view on a tensor which is declared not to need gradients, i.e., it is to be excluded from
                #  further tracking of operations, and therefore the subgraph involving this view is not recorded.

                '''
                nps_loss = nps
                tv_loss = tv*8
                loss = nps_loss + (det_loss**3/tv_loss + tv_loss**3/det_loss)**(1/3)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                adv_patch_cpu.data.clamp_(0,1)       #keep patch in image range
    
                '''

                # scale nps_loss
                alpha = 0.01
                nps_loss = nps*alpha

                # scale tv_loss
                beta = 2.5
                tv_loss = tv*beta

                # calculate the total loss
                loss = det_loss + nps_loss + tv_loss
                # print('loss ' + str(loss.size()))
                # print(loss)

                # Backward pass to minimize the loss using Adam
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # NB:
                # zero_grad clears old gradients from the last step (otherwise you’d just accumulate the gradients from all loss.backward() calls).
                # loss.backward() computes the derivative of the loss w.r.t. the parameters (or anything requiring gradients) using backpropagation.
                # opt.step() causes the optimizer to take a step based on the gradients of the parameters.
                #print(adv_patch_cpu)
                adv_patch_cpu.data.clamp_(0,1)       # keep patch in image range

                bt1 = time.time()

                # show the learning process of the adv_patch

                if i_batch%1 == 0:
                    print('BATCH', i_batch, end='...\n')
                    #print(adv_patch_cpu)
                    im = transforms.ToPILImage('RGB')(adv_patch_cpu)  # transform the adv_patch tensor at the current training state into an image
                    # plt.imshow(im)  # show the adv_patch at the current training state, every 5 images from the batch
                    # plt.show()

                print('  BATCH NR: ', i_batch)
                print('BATCH LOSS: ', loss.detach().cpu().numpy())
                print('  DET LOSS: ', det_loss.detach().cpu().numpy())
                print('  NPS LOSS: ', nps_loss.detach().cpu().numpy())
                print('   TV LOSS: ', tv_loss.detach().cpu().numpy())
                print('BATCH TIME: ', bt1-bt0)

                if i_batch + 1 >= len(train_loader):
                    print('\n') #  at the last batch (100 I think) make newline
                else:
                    del adv_batch_t, output, max_prob, det_loss, p_img_batch, nps_loss, tv_loss, loss
                    if use_cuda:
                        torch.cuda.empty_cache()
                bt0 = time.time()

        # when the batch is completely explored, evaluate your results upon the single epoch, which is finished as well
        et1 = time.time()
        ep_det_loss = ep_det_loss/len(train_loader)
        ep_nps_loss += nps_loss.detach().cpu().numpy()
        ep_tv_loss += tv_loss.detach().cpu().numpy()
        ep_loss += loss

        if True:
            print('  EPOCH NR: ', epoch),
            print('EPOCH LOSS: ', ep_loss)
            print('  DET LOSS: ', ep_det_loss)
            print('  NPS LOSS: ', ep_nps_loss)
            print('   TV LOSS: ', ep_tv_loss)
            print('EPOCH TIME: ', et1-et0)  # print the duration of an epoch

            # take the last optimized image of the current epoch and save it in your folder.
            # the very last image saved of the entire process will be that at the last epoch, at batch 100 (the last) of that epoch
            im = transforms.ToPILImage('RGB')(adv_patch_cpu)
            # plt.imshow(im)
            # plt.show()
            im.save("/scratch/msc20f3/master_thesis/saved_patches_mytrial/test260320.jpg")  # save learned patch into a folder
            del adv_batch_t, output, max_prob, det_loss, p_img_batch, nps_loss, tv_loss, loss
            if use_cuda:
                torch.cuda.empty_cache()
        et0 = time.time()

#TODO _________________________________________________________________________________________________________________________________________________________________
    # TEST: Check if our patch fools the detector (patch test)

    patch_size = 300
    img_size = darknet_model.height

    img_dir_v = "/scratch/msc20f3/master_thesis/test/img"  # images on which the patch can be tested
    lab_dir_v = "/scratch/msc20f3/master_thesis/test/lab"  # labels of the aforementioned image

    adv_patch = Image.open("/scratch/msc20f3/master_thesis/saved_patches_mytrial/test260320.jpg").convert('RGB')  # open the learned patch

    if use_cuda:
        adv_patch = transforms.ToTensor(adv_patch).cuda()
    else:
        adv_patch = transforms.ToTensor(adv_patch)  # Convert a PIL Image or numpy.ndarray to tensor


    test_loader = torch.utils.data.DataLoader(InriaDataset(img_dir_v, lab_dir_v, 14, img_size, shuffle=True),
                                                   batch_size=1,
                                                   shuffle=True,
                                                   num_workers=10)

    for i_batch_test, (img_batch_test, lab_batch_test) in enumerate(test_loader):
        img_size = img_batch_test.size(-1) # take the size of the last dimension of the input image

        if use_cuda:
            adv_batch_t = patch_transformer(adv_patch, lab_batch.cuda(), img_size, do_rotate=True, rand_loc=False)
            p_img = patch_applier(img_batch.cuda(), adv_batch_t)
        else:
            adv_batch_t = patch_transformer(adv_patch, lab_batch_test, img_size, do_rotate=True, rand_loc=False)
            p_img = patch_applier(img_batch_test, adv_batch_t)

        p_img = F.interpolate(p_img,(darknet_model.height, darknet_model.width))

        output = darknet_model(p_img)  # see detector response when a test image with the patch applied is analysed

        boxes = get_region_boxes(output,0.5,darknet_model.num_classes, darknet_model.anchors, darknet_model.num_anchors)[0]  # get_region_boxes in utils
        boxes = nms(boxes,0.4)  # nms in utils

        class_names = load_class_names('/scratch/msc20f3/master_thesis/data/coco.names')  # load_class_names in utils
        squeezed = p_img.squeeze(0)  # remove dimensions of size 1 in the tensor
        print(squeezed.shape)

        img = transforms.ToPILImage('RGB')(squeezed.detach().cpu())  # transform the tensor into image
        plotted_image = plot_boxes(img, boxes, class_names=class_names)  # plot_boxes in utils
        # plt.imshow(plotted_image)
        # plt.show()
