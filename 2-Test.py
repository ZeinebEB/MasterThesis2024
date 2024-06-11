import argparse
import json
import os

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from FFTRadNet.model.Resnet18pEnc_FftDec import resnetEnc
from FFTRadNet.model.Unet_FFt import Unet_fft
from dataset.dataset import RADIal
from dataset.encoder import ra_encoder
from utils.util import DisplayHMI, DisplayHMI_Seg, DisplayHMI_Det


def main(config, checkpoint_filename, difficult,code):
    code="11"
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    output_dir = "Testzzzz/"
    os.makedirs(output_dir, exist_ok=True)
    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load the dataset
    enc = ra_encoder(geometry=config['dataset']['geometry'],
                     statistics=config['dataset']['statistics'],
                     regression_layer=2)

    dataset = RADIal(root_dir=config['dataset']['root_dir'],
                     statistics=config['dataset']['statistics'],
                     encoder=enc.encode,
                     difficult=difficult)


    # Create the model
    # net = FFTRadNet(blocks=config['model']['backbone_block'],
    #                 mimo_layer=config['model']['MIMO_output'],
    #                 channels=config['model']['channels'],
    #                 regression_layer=2,
    #                 detection_head=config['model']['DetectionHead'],
    #                 segmentation_head=config['model']['SegmentationHead'])
    # #



    # net = DeepLabV3PlusWithMIMO(n_channels=32, n_classes=1)
    net=resnetEnc(n_channels=32, n_classes=1,detection_head=False,segmentation_head=True)
    # net=SegnetEnc(n_channels=32, BN_momentum=0.5, n_classes=3, detection_head=True, segmentation_head=True)
    # net=Unet_fft(segmentation_head=True,detection_head=False)


    # net= UNetWithMIMOSD(n_channels=32, n_classes=1, bilinear=False,segmentation_head=True,detection_head=True)

    net.to('cuda')

    # Load the model
    dict = torch.load(checkpoint_filename)

    net.load_state_dict(dict['net_state_dict'])
    net.eval()


    for idx, data in enumerate(dataset):

    # for data in dataset:

        # Convert data to tensor and move to GPU
        inputs = torch.tensor(data[0]).permute(2, 0, 1).to('cuda').float().unsqueeze(0)

        # Perform inference
        with torch.set_grad_enabled(False):
            outputs = net(inputs)
        # Generate HMI visualization

        # data is composed of [radar_FFT 0, segmap 1,out_label 2,box_labels 3,image 4]

        if code=='11':
            hmi = DisplayHMI(data[4], data[0], outputs, enc, data[3])

        elif code=='10':
            hmi = DisplayHMI_Det(data[4], data[0], outputs, enc,data[3])

        elif code == '01':
            hmi = DisplayHMI_Seg(data[4], data[0], outputs, enc)


        # Save the HMI image
        image_name = f'output_{str(idx).zfill(6)}.png'  # Using index with leading zeros
        image_path = os.path.join(output_dir, image_name)
        cv2.imwrite(image_path, hmi)

        # cv2.imshow('FFTRadNet', hmi)
        # cv2.imwrite('FFTRadNet.jpg', hmi)
        #
        # # Press Q on keyboard to  exit
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     break


    cv2.destroyAllWindows()


if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='FFTRadNet test')
    parser.add_argument('-c', '--config', default='weights/Resnet18Enc/Resnet18_D&S_rand/config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--checkpoint', default="weights/Resnet18Enc/Resnet18_D&S_rand/FFTRadNet_RA_192_56_epoch99_loss_759006.2871_AP_0.9255_AR_0.6944_IOU_0.6683.pth", type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('--difficult', action='store_true')
    parser.add_argument('--DET/SEG',default='11',type=str,help='11=D+S 10=D 01=S')
    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config, args.checkpoint,args.difficult,"11")