# import required packages
from config import Config
import pickle as pkl
from generate_images import ImgGenerator
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import cv2
import argparse


# utils
def read_image(img_path, img_h=32, char_w=16):
    valid_img = True
    img_h=32
    char_w=16
    
    try:
        img = cv2.imread(img_path, 0)
        y,x=img.shape
        img=img[10:y-10,10:x-10]
        iy,iw=img.shape
        thresh= cv2.threshold(img, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        blur=cv2.GaussianBlur(img,(13,13),100)
        thresh_inv=cv2.threshold(blur,128,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
        cnts=cv2.findContours(thresh_inv,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts=cnts[0] if len(cnts)==2 else cnts[1]
        cnts=sorted(cnts,key=lambda x:cv2.boundingRect(x)[1])
        xl,yl,xh,yh=0,0,0,0
        for c in cnts:
            x,y,w,h=cv2.boundingRect(c)
            if xh==0:
                xl,yl,xh,yh=x,y,x+w,y+h
            else:
                xl=min(xl,x)
                yl=min(yl,y)
                xh=max(xh,x+w)
                yh=max(yh,y+h)
        img=thresh[yl:yh,xl:xh]
        curr_h, curr_w = img.shape
        modified_w = int(curr_w * (img_h / curr_h))
        img = cv2.resize(img, (modified_w, img_h))

    except AttributeError:
        print('Error Loading the Image')

    return img

def main(args):
    #  loading the model
    dataset='hindi'
    config = Config
    # config.dataset = 'hindi'
    # config.num_chars = 109

    with open(config.data_file, 'rb') as f:  # data file path
        char_map = pkl.load(f)
    char_map=char_map['char_map']
    MainModel = ImgGenerator(checkpt_path=args.model_path,  # model path
                            config=config, char_map=char_map)
    transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
    ])
    recogniser = MainModel.model.R


    # Load image
    img=read_image(args.input_file)  # path to the image
    transformed_img = transforms(img / 255.)
    transformed_img = transformed_img.float()
    transformed_img = pad_sequence([transformed_img.squeeze().permute(1, 0)],
                    batch_first=True,
                    padding_value=1.)
    transformed_img = transformed_img.permute(0, 2, 1).unsqueeze(1)

    # run recogniser model
    recogniser_output=recogniser(transformed_img).permute(1, 0, 2).max(2)[1].permute(1, 0)
    preds = MainModel.word_map.recognizer_decode(recogniser_output.cpu().numpy())
    print('PREDICTED OUTPUT :',preds)
    return preds



def parse_args():
    parser = argparse.ArgumentParser(description="Preprocessing image", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--input_file", type=str, default=None, help="path to the input img file")
    parser.add_argument("-m", "--model_path", type=str, default='./weights/model_checkpoint_epoch_100.pth.tar', help="path to the model")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)