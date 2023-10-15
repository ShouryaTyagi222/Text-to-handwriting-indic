# import required packages
from config import Config
import pickle as pkl
from generate_images import ImgGenerator
import matplotlib.pyplot as plt
import torch
import cv2
import argparse
import numpy as np
import os

# produce images 
def produce_image(generator,words,images_folder):
  for i in words:
    generated_imgs, _, word_labels = generator.generate(word_list=i)
    sentence_img = []
    for label, img in zip(word_labels, generated_imgs):
        img = img[:, img.sum(0) < 31.5]
        img*=255
        cv2.imwrite(images_folder+'/'+label+'.jpg',img)


def main(args):
    images_folder = args.output_folder_name

    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
    
    # loading the model
    dataset='hindi'
    config = Config
    # config.dataset = 'hindi'
    # config.num_chars = 109

    print('LOADING THE MODEL')
    with open(config.data_file, 'rb') as f:  # data file path
        char_map = pkl.load(f)
    char_map=char_map['char_map']
    generator = ImgGenerator(checkpt_path=args.model_path,
                            config=config, char_map=char_map)
    z_dist = torch.distributions.Normal(loc=0, scale=1.)


    print('LOADING THE INPUT WORDS')
    with open(args.input_file, 'r') as f:
        ids = f.read()
        words = [i.split()[-1] for i in ids.split('\n') if len(i) > 1]
    word_list=[]
    twords=[]
    for index,word in enumerate(words):
        if index!=0 and index%50==0:
            word_list.append(twords)
            twords=[]
        twords.append(word)
    
    print('STARTING TO GENERATE THE IMAGES')
    produce_image(generator,word_list,images_folder)

    print('ALL THE IMAGES HAVE BEEN PRODUCED AND SAVED IN THE OUPUT FOLDER')



def parse_args():
    parser = argparse.ArgumentParser(description="Preprocessing image", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--input_file", type=str, default=None, help="path to the input img file")
    parser.add_argument("-o", "--output_folder_name", type=str, default="OUTPUT", help="path to the output img directory")
    parser.add_argument("-m", "--model_path", type=str, default='./weights/model_checkpoint_epoch_100.pth.tar', help="path to the model")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)