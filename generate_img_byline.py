# import required packages
from config import Config
import pickle as pkl
from generate_images import ImgGenerator
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
import argparse

def load_model(model_path):
    dataset='hindi'
    config = Config
    # config.dataset = 'hindi'
    # config.lexicon_file = '/content/drive/MyDrive/Text-to-handwriting-indic/data/Lexicon/words.txt'
    # config.num_chars = 109

    with open(config.data_file, 'rb') as f:
        char_map = pkl.load(f)
    char_map=char_map['char_map']
    generator = ImgGenerator(checkpt_path=model_path,
                            config=config, char_map=char_map)
    return generator
    
def prepare_text(text):
    para=[]
    line=[]
    for index,word in enumerate(text.split()):
        if index!=0 and index % 10 == 0:
            para.append(line)
            line=[]
        line.append(word)
    return para

def gen_images(para,generator):
    final_image=[]
    for line in para:
        generated_imgs, _, word_labels = generator.generate(word_list=line)
        sentence_img = []
        for label, img in zip(word_labels, generated_imgs):
            img = img[:, img.sum(0) < 31.5]
            sentence_img.append(img)
            sentence_img.append(np.ones((img.shape[0], 15)))
        sentence_img = np.hstack(sentence_img)
        page_w=800
        # print(sentence_img.shape)
        if sentence_img.shape[1]<page_w:
            residual=page_w-sentence_img.shape[1]
        else:
            sentence_img=cv2.resize(sentence_img,(page_w-20,32))
            residual=page_w-sentence_img.shape[1]
        sentence_img = np.hstack([np.ones((img.shape[0], 15)),sentence_img,np.ones((32,residual))])
        final_image.append(sentence_img*255)
    return final_image

def save_image(final_image,output_path):
    min_w=min([i.shape[1] for i in final_image])
    image=[]
    for i in final_image:
        y,w=i.shape
        image.append(cv2.resize(i,(min_w,y)))
    i=np.concatenate(image)
    cv2.imwrite(output_path+'/image.jpg',i)

def main(args):
    generator=load_model(args.model_path)
    print('PREPARING THE TEXT ...')
    paras=prepare_text(args.input_text)
    print('GENERATING IMAGES ...')
    img=gen_images(paras,generator)
    print('SAVING THE FINAL IMAGE ...')
    save_image(img,args.output_folder_name)
    print('IMAGES GENERATED')

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocessing image", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--input_text", type=str, default=None, help="input text")
    parser.add_argument("-o", "--output_folder_name", type=str, default="OUTPUT", help="path to the output img directory")
    parser.add_argument("-m", "--model_path", type=str, default='./weights/model_checkpoint_epoch_100.pth.tar', help="path to the model")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
# if __name__=='__main__':
#     main(text='थी फिर आज जाता होने उसे लोग सकता उनके सरकार देश उन्होंने उसके इन वो सभी होती इसके अगर थी। बार जाने हर ऐसा गए सब करें ऐसे जाती मेरे हुई जीवन साल वाली  हमारेदे लेकर होगा गया। इसी ज्यादा अभी आपको उनका उसकी भारतीय एवं समाज सकते क्योंकि यही तुम जाते फिल्म कहीं मन देने लगता हूँ बना हमें आगे करती चाहिए ऐसी उसका बीच सकती आदमी हूं दिल्ली रह श्री होना अन्य दूसरे वहां आया मे केवल कंपनी जाएगा। डॉ माध्यम राम उसमें तमाम बाजार महिलाओं रख रखा लगातार स्तर कार्यक्रम जवाब देशों पसंद बाबा मत मानव व्यवस्था अमेरिका कहने जाए। जिले डर देखकर बता शर्मा अधिकारी करें। छोटे तय नया पाया पुराने बिहार मतलब मुताबिक लगी  विज्ञान सूचना अंदर घटना ज्ञान बनाए मौत संसद हमारा आरोप चाहता जाकर दें रहा। लाल विवाह संख्या सामाजिक हाल अध्यक्ष आवाज कानून चले जरूरी तुम्हारे नेताओं पहला बातों मुख्यमंत्री लेखक होगी। जगत परन्तु बाल मित्र लिया। अन्ना किन्तु गंभीर दर्ज प्रभावित बेहद मार मौजूद रोक वर्ग अचानक आत्मा उनसे उम्मीद गये। जाएगा टिप्पणी देखता पूछा बेटे मामला राज विरोध सारा स्थापित हिस्सा अंतिम उम्र कितने चाहती जन दिये बताते बैठक राजस्थान लगा। साथ-साथ उनको उसको कहता कुल खो चली जल्दी जून दस निर्णय पेश प्रवेश भविष्य महसूस लिखने सहित स्वीकार आनंद ऑफ कदम जाति जिसकी प्रस्तुत भोजन मुंह लेख विदेशी संघर्ष सत्ता सहायता स्पष्ट आखिर उदाहरण छोड़ जुलाई दल नंबर निजी परीक्षा पाने प्रक्रिया बड़ा मान मैदान लगते शादी संसार हत्या हेतु आंदोलन आयोग इसीलिए करनी खान गैस छोटी जितना डाल प्रमुख बड़े मंत्रालय मालूम राशि शासन शेयर सार्वजनिक स्वामी होंगे आँखों आदेश ऊर्जा कब कहती चीन तेजी दर्द देवी पत्थर पूजा प्रभाव बचपन बल बोले मध्य मुझसे युवा रोज वर्षों सदा अगले अप्रैल आपसे ईश्वर किनारे खाली घोषणा तर्क दृष्टि पक्ष पृथ्वी प्रश्न बचा मानना मुक्ति लगे। लोकतंत्र  विधायक शुक्रवार समिति सामान्य सूची आयोजित एम  किताब केजरीवाल खत्म खर्च छोड़ जेल तुम्हारी निकाल पड़ बनाई भरोसा मैने रखी रहना लाने शिकार सजा सफल सालों सुनकर')