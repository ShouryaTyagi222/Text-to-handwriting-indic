# import required packages
from config import Config
import pickle as pkl
from generate_images import ImgGenerator
import matplotlib.pyplot as plt
import torch
import numpy as np

dataset='hindi'
config = Config
config.num_chars = 109

with open('./data/hindi_tr_data.pkl', 'rb') as f:
   char_map = pkl.load(f)
char_map=char_map['char_map']
generator = ImgGenerator(checkpt_path=f'./weights/model_checkpoint_epoch_45.pth.tar',  # path to the saved model
                         config=config, char_map=char_map)
z_dist = torch.distributions.Normal(loc=0, scale=1.)

noise_seed = 0 # min:0, max:100, step:1

torch.manual_seed(noise_seed)
z = z_dist.sample([128])

# specific words, same style
sentences = ["सुनकर"]  # input image
for word_list in sentences:
    word_list = word_list.split(' ')
    generated_imgs, _, word_labels = generator.generate(word_list=word_list)
    sentence_img = []
    for label, img in zip(word_labels, generated_imgs):
        img = img[:, img.sum(0) < 31.5]
        sentence_img.append(img)
        sentence_img.append(np.ones((img.shape[0], 15)))
    sentence_img = np.hstack(sentence_img)
    plt.imshow(sentence_img, cmap='gray')
    plt.axis('off')
    plt.show()