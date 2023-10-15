import torch


class Config:
    dataset = 'hindi'  # hindi, tamil, etc.
    data_folder_path = f'./data/{dataset}' 
    img_h = 32
    char_w = 16
    partition = 'all'  # 'tr = train' / 'vl = val' / 'te = test' / all = train + test + val

    batch_size = 16
    num_epochs = 100   # number of epochs to train for
    epochs_lr_decay = 100  # learning rate decay will be applied for last these many steps (should be <= num_epochs)
    resume_training = False
    start_epoch = 1

    train_gen_steps = 4  # generator weights to be updated after every specified number of steps
    grad_alpha = 1
    grad_balance = True

    data_file = f'./data/{dataset}_{partition}_data.pkl'
    lexicon_file_name = 'vocab.txt'
    lexicon_file = f'./data/{dataset}/{lexicon_file_name}'
    lmdb_output = f'./data/{dataset}_{partition}_data'

    architecture = 'ScrabbleGAN'
    # Recognizer network
    r_ks = [3, 3, 3, 3, 3, 3, 2]
    r_pads = [1, 1, 1, 1, 1, 1, 0]
    r_fs = [64, 128, 256, 256, 512, 512, 512]

    # Generator and Discriminator networks
    # arch[g_resolution] defines the architecture to be selected
    # arch[16] has been added in BigGAN.py with parameters as specified in the paper
    resolution = 16
    bn_linear = 'SN'
    g_shared = False

    g_lr = 2e-4
    d_lr = 2e-4
    r_lr = 2e-4
    g_betas = [0., 0.999]
    d_betas = [0., 0.999]
    r_betas = [0., 0.999]
    g_loss_fn = 'HingeLoss'
    d_loss_fn = 'HingeLoss'
    r_loss_fn = 'CTCLoss'

    # Noise vector
    z_dim = 128
    num_chars = 109  # change the num chars on the basis of the dataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
