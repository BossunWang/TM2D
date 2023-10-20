import os

from os.path import join as pjoin

import utils.paramUtil as paramUtil
from options.train_options import TrainVQTokenizerOptions
from utils.plot_script import *

from networks.modules import *
from networks.quantizer import *
from data.dataset import DanceTokenizeDataset
# from data.dataset import DanceTokenizeDatasetV2  # 1018
from scripts.motion_process import *
from torch.utils.data import DataLoader
import codecs as cs

"""
Compared to tokenize_script.py, this one doesn't have random shift. 
It tokenizes directly according to the original motion so that it can align with the audio.
"""


def loadVQModel(opt):
    vq_encoder = VQEncoderV3(dim_pose - 4, enc_channels, opt.n_down)
    # vq_decoder = VQDecoderV3(opt.dim_vq_latent, dec_channels, opt.n_resblk, opt.n_down)
    quantizer = None
    if opt.q_mode == 'ema':
        quantizer = EMAVectorQuantizer(opt.codebook_size, opt.dim_vq_latent, opt.lambda_beta)
    elif opt.q_mode == 'cmt':
        quantizer = Quantizer(opt.codebook_size, opt.dim_vq_latent, opt.lambda_beta)
    # checkpoint = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'model', 'finest.tar'),
    #                         map_location=opt.device)
    checkpoint = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'model',
                                  '%s.tar' % (opt.which_vqvae)), map_location=opt.device)
    vq_encoder.load_state_dict(checkpoint['vq_encoder'])
    quantizer.load_state_dict(checkpoint['quantizer'])
    return vq_encoder, quantizer


if __name__ == '__main__':
    parser = TrainVQTokenizerOptions()
    opt = parser.parse()

    opt.is_train = False

    opt.device = torch.device("cpu" if opt.gpu_id==-1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)
    if opt.gpu_id != -1:
        torch.cuda.set_device(opt.gpu_id)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.meta_dir, exist_ok=True)

    if opt.dataset_name == 't2m':
        assert False, 'for mix'
    elif opt.dataset_name == 'aistpp':
        opt.data_root = './dataset/aistpp/'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        # opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 24
        # opt.max_motion_length = 196   # this is for
        dim_pose = 287
        radius = 4
        fps = 60
        kinematic_chain = paramUtil.smpl24_kinematic_chain
    elif opt.dataset_name == 'aistppml3d':
        opt.data_root = './dataset/aistppml3d/'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 24  #22
        # opt.max_motion_length = 196   # this is for
        dim_pose = 287  #263
        radius = 4
        fps = 60
        # kinematic_chain = paramUtil.t2m_kinematic_chain
        kinematic_chain = paramUtil.smpl24_kinematic_chain
    elif opt.dataset_name == 'kit':
        assert False, 'for mix'
    else:
        raise KeyError('Dataset Does Not Exist')

    mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
    std = np.load(pjoin(opt.meta_dir, 'std.npy'))

    all_split_file = pjoin(opt.data_root, 'all.txt')

    # enc_channels = [1024, opt.dim_vq_latent]
    # dec_channels = [opt.dim_vq_latent, 1024, dim_pose]
    enc_channels = [1024,] + [opt.dim_vq_latent,] * (opt.n_down - 1)
    dec_channels = [opt.dim_vq_latent,] * (opt.n_down - 1) +[1024, dim_pose]

    vq_encoder, quantizer = loadVQModel(opt)

    all_params = 0
    pc_vq_enc = sum(param.numel() for param in vq_encoder.parameters())
    print(vq_encoder)
    print("Total parameters of encoder net: {}".format(pc_vq_enc))
    all_params += pc_vq_enc

    pc_quan = sum(param.numel() for param in quantizer.parameters())
    print(quantizer)
    print("Total parameters of codebook: {}".format(pc_quan))
    all_params += pc_quan

    print('Total parameters of all models: {}'.format(all_params))

    all_dataset = DanceTokenizeDataset(opt, mean, std, all_split_file)

    all_loader = DataLoader(all_dataset, batch_size=1, num_workers=1, pin_memory=True)

    # token_data_dir = pjoin(opt.data_root, opt.name)
    token_data_dir = pjoin(opt.data_root, '{}_{}'.format(opt.name, opt.which_vqvae))  # try other epoch of VAE
    os.makedirs(token_data_dir, exist_ok=True)

    start_token = opt.codebook_size
    end_token = opt.codebook_size + 1
    pad_token = opt.codebook_size + 2

    # max_length = 55
    # num_replics = 5
    opt.unit_length = 8  # 8 for fps 60, 4 for fps 20

    vq_encoder.to(opt.device)
    quantizer.to(opt.device)
    vq_encoder.eval()
    quantizer.eval()
    with torch.no_grad():
        # Since our dataset loader introduces some randomness (not much), we could generate multiple token sequences
        # to increase the robustness.
        # not suitable for dance one.
        # for e in range(num_replics):
        for i, data in enumerate(tqdm(all_loader)):
            motion, name = data
            motion = motion.detach().to(opt.device).float()
            pre_latents = vq_encoder(motion[..., :-4])
            indices = quantizer.map2index(pre_latents)
            indices = list(indices.cpu().numpy())
            # indices = [start_token] + indices + [end_token] + [pad_token] * (max_length - len(indices) - 2)
            indices = [str(token) for token in indices]
            with cs.open(pjoin(token_data_dir, '%s.txt'%name[0]), 'a+') as f:
                # if e!= 0:
                #     f.write('\n')
                f.write(' '.join(indices))