import sys

sys.path.extend(['..'])
import numpy as np
import pickle as pkl
import cv2


def read_image(img_path, label_len, img_h=32, char_w=16):
    valid_img = True
    img_h=32
    char_w=16
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
                
            # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),5)
    img=thresh[yl:yh,xl:xh]
    try:
        curr_h, curr_w = img.shape
        modified_w = int(curr_w * (img_h / curr_h))

        # Remove outliers
        if ((modified_w / label_len) < (char_w / 3)) | ((modified_w / label_len) > (3 * char_w)):
            valid_img = False
        else:
            # Resize image so height = img_h and width = char_w * label_len
            img_w = label_len * char_w
            img = cv2.resize(img, (img_w, img_h))

    except AttributeError:
        valid_img = False

    return img, valid_img


def read_data(config):
    """
    Saves dictionary of preprocessed images and labels for the required partition
    """
    img_h = config.img_h
    char_w = config.char_w
    partition = config.partition
    out_name = config.data_file
    data_folder_path = config.data_folder_path
    dataset = config.dataset

    if dataset == 'hindi':

        if partition == 'tr':
            partition_name = 'train'
        elif partition == 'vl':
            partition_name = 'val'
        else:
            partition_name = 'test'

        # create char_map using training labels
        with open(f'{data_folder_path}/train.txt', 'r') as f:
            ids = f.read()
            partition_ids = [i.split()[0] for i in ids.splitlines() if len(i) > 1]
            words_raw = [i.split()[1] for i in ids.splitlines() if len(i) > 1]

        # Get list of unique characters and create dictionary for mapping them to integer
        chars = np.unique(np.concatenate([[char for char in w_i.split()[-1]] for w_i in words_raw]))
        char_map = {value: idx + 1 for (idx, value) in enumerate(chars)}
        char_map['<BLANK>'] = 0
        num_chars = len(char_map.keys())

        # Extract IDs for required set
        with open(f'{data_folder_path}/{partition_name}.txt', 'r') as f:
            ids = f.read()
            partition_ids = [i.split()[0] for i in ids.splitlines() if len(i) > 1]
            words_raw = [i.split()[1] for i in ids.splitlines() if len(i) > 1]

        word_data = {}
        for img_path, label in zip(partition_ids, words_raw):
            img_path = f'{data_folder_path}/{partition_name}/{img_path}'
            img, valid_img = read_image(img_path, len(label), img_h, char_w)
            img_id = img_path.split('/')[-1].split('.')[0]
            if valid_img:
                try:
                    word_data[img_id] = [[char_map[char] for char in label], img]
                except KeyError:
                    pass

    print(f'Number of images = {len(word_data)}')
    print(f'Number of unique characters = {num_chars}')

    # Save the data
    with open(f'{out_name[out_name.rfind("/")+1:]}', 'wb') as f:
        pkl.dump({'word_data': word_data,
                  'char_map': char_map,
                  'num_chars': num_chars}, f, protocol=pkl.HIGHEST_PROTOCOL)

from config import Config
if __name__ == '__main__':
    config = Config
    print('Processing Data:\n')
    read_data(config)
    print('\nData processing completed')