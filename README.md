# ScrabbleGAN - Handwritten Text Generation

## Steps for training the ScrabbleGAN model from scratch
1. Download the dataset

and keep them in the data `/data/` directory as shown below:
```
Repo Directory
├── data
|   ├── hindi                       # Name According to the name of the dataset in config.py eg. hindi,tamil, etc.
|       └──train/                   # Training Images
|       └──test/
|       └──val/
|       └──train.txt                # Text file for training data
|       └──test.txt
|       └──val.txt                  
|       └──vocab.txt                # Lexicon
|   └──Here will come the generated char_map for the hindi by running the prepare_data.py follow the steps.
```

2. Modify the `/config.py` file to change :
`dataset`, `num_chars`, `start_epoch`, `train_gen_steps`,  `num_epochs`, `batch_size`, `resume_training` , `partition` and File_Structure


3. To Prepare Data for Training, run:
    ```bash
    python prepare_data.py
    ```
    This will process the char_map , word_data and num_chars and create a pickle file to be used for training. 

4. Start model training by running the below command from the main directory:
    ```bash
    python train.py
    ```
   This will start training the model. A sample generated image will be saved in the `output` directory
   after every epoch. Tensorboard logging has also been enabled.  
   And The model checkpoint will be saved in every 5 epochs.

## Generating new image
To generate new image run `run_inference.py` , where you can change the 

1. path to the model.
2. path to the data file.
3. word for which you want to generate image.
4. noise in the generated image.

## Use the Recogniser model
To use the recogniser model run `run_recogniser.py `. In it you have to provide the path of the image and the path of the model

## To produce images of many words Using the Generator
To Produce images of many words run `produce_images.py ` in this you have to provide :
1. txt file which consists of all the words separated by lines.
2. path to the output folder.
3. path to the generator model.