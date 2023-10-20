import os

from os.path import join as pjoin

import utils.paramUtil as paramUtil
from options.train_options import TrainT2MOptions
from utils.plot_script import *

from networks.transformer_x import TransformerV1, TransformerV2, TransformerX3
from networks.quantizer import *
from networks.modules import *
from networks.trainers_x5 import TransformerATMTrainerV2
from data.dataset import AudioMotionTokenDatasetV2, TextMotionTokenDatasetV2, MixDataset
from scripts.motion_process import *
from torch.utils.data import DataLoader
from utils.word_vectorizer import WordVectorizerV2


"""
This file is to train the TM2D
"""


if __name__ == '__main__':
    parser = TrainT2MOptions()
    opt = parser.parse()

    opt.device = torch.device("cpu" if opt.gpu_id==-1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)
    if opt.gpu_id != -1:
        torch.cuda.set_device(opt.gpu_id)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.eval_dir = pjoin(opt.save_root, 'animation')
    opt.log_dir = pjoin('./log', opt.dataset_name, opt.name)

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    if opt.dataset_name == 't2m':
        assert False, 'for mix'
        opt.data_root = './dataset/HumanML3D/'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 24  # 22
        opt.max_motion_len = 84 # 84 = 55x3/8x4# 55
        opt.max_text_len = 84 # 84 = 55x3/8x4# 55
        dim_pose = 287  # 263
        radius = 4
        fps = 60
        # kinematic_chain = paramUtil.t2m_kinematic_chain
        kinematic_chain = paramUtil.smpl24_kinematic_chain
    elif opt.dataset_name == 'aistpp':
        assert False, 'for mix'
        opt.data_root = './dataset/aistpp/'
        # opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        # opt.audio_dir = pjoin(opt.data_root, 'audio_vecs')
        opt.joints_num = 24
        opt.max_motion_len = None
        opt.window_size = 30  # because min-len of dance is 130 (20fps) -> 30x4. 425 (60fps) -> 50x8
        # dim_pose = 263
        # dim_audio = 438
        # radius = 4
        # fps = 20
        kinematic_chain = paramUtil.smpl24_kinematic_chain
    elif opt.dataset_name == 'aistppml3d':
        opt.data_root = './dataset/aistppml3d/'

        opt.joints_num = 24
        opt.max_motion_len = None
        # opt.window_size = 30  # because min-len of dance is 130 (20fps) -> 30x4. 425 (60fps) -> 50x8
        opt.window_size = 50  # because min-len of dance is 130 (20fps) -> 30x4. 425 (60fps) -> 50x8

        # opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.audio_dir = pjoin(opt.data_root, 'audio_vecs_7.5fps')
        # opt.joints_num = 24  # 22
        opt.max_motion_len = 84  # 84 = 55x3/8x4# 55
        opt.max_text_len = 84 # 84 = 55x3/8x4# 55

        kinematic_chain = paramUtil.smpl24_kinematic_chain
    elif opt.dataset_name == 'kit':
        assert False, 'for mix'
    else:
        raise KeyError('Dataset Does Not Exist')

    # if opt.text_aug:
    #     opt.text_dir = pjoin(opt.data_root, '%s_AUG_texts'%(opt.tokenizer_name))
    #
    # else:
    #     opt.text_dir = pjoin(opt.data_root, 'texts')


    # mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.tokenizer_name, 'meta', 'mean.npy'))
    # std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.tokenizer_name, 'meta', 'std.npy'))

    train_split_file_a2d = pjoin(opt.data_root, 'aistpp_train.txt')
    val_split_file_a2d = pjoin(opt.data_root, 'aistpp_val.txt')
    train_split_file_t2m = pjoin(opt.data_root, 'ml3d_train.txt')
    val_split_file_t2m = pjoin(opt.data_root, 'ml3d_val.txt')

    w_vectorizer = WordVectorizerV2('./glove', 'our_vab')

    n_mot_vocab = opt.codebook_size + 3
    opt.mot_start_idx = opt.codebook_size
    opt.mot_end_idx = opt.codebook_size + 1
    opt.mot_pad_idx = opt.codebook_size + 2

    n_aud_vocab = opt.codebook_size + 3
    opt.aud_start_idx = opt.codebook_size
    opt.aud_end_idx = opt.codebook_size + 1
    opt.aud_pad_idx = opt.codebook_size + 2

    n_txt_vocab = len(w_vectorizer) + 2
    _, _, opt.txt_start_idx = w_vectorizer['sos/OTHER']
    _, _, opt.txt_end_idx = w_vectorizer['eos/OTHER']
    opt.txt_mlen_idx = len(w_vectorizer)
    opt.txt_pad_idx = len(w_vectorizer) + 1


    a2d_transformer = TransformerV1(n_mot_vocab, opt.mot_pad_idx, d_src_word_vec=438, d_trg_word_vec=512,
                                    d_model=opt.d_model, d_inner=opt.d_inner_hid, n_enc_layers=opt.n_enc_layers,
                                    n_dec_layers=opt.n_dec_layers, n_head=opt.n_head, d_k=opt.d_k, d_v=opt.d_v,
                                    dropout=0.1,
                                    n_src_position=100, n_trg_position=100,
                                    trg_emb_prj_weight_sharing=opt.proj_share_weight)

    t2m_transformer = TransformerV2(n_txt_vocab, opt.txt_pad_idx, n_mot_vocab, opt.mot_pad_idx,
                                    d_src_word_vec=opt.d_model, d_trg_word_vec=opt.d_model,
                                    d_model=opt.d_model, d_inner=opt.d_inner_hid, n_enc_layers=opt.n_enc_layers,
                                    n_dec_layers=opt.n_dec_layers, n_head=opt.n_head, d_k=opt.d_k, d_v=opt.d_v,
                                    dropout=0.1,
                                    n_src_position=100, n_trg_position=100,
                                    trg_emb_prj_weight_sharing=opt.proj_share_weight
                                    )

    m2m_transformer = TransformerV2(n_mot_vocab, opt.mot_pad_idx, n_mot_vocab, opt.mot_pad_idx,
                                    d_src_word_vec=opt.d_model, d_trg_word_vec=opt.d_model,
                                    d_model=opt.d_model, d_inner=opt.d_inner_hid, n_enc_layers=opt.n_enc_layers,
                                    n_dec_layers=opt.n_dec_layers, n_head=opt.n_head, d_k=opt.d_k, d_v=opt.d_v,
                                    dropout=0.1,
                                    n_src_position=100, n_trg_position=100,
                                    trg_emb_prj_weight_sharing=opt.proj_share_weight
                                    )

    atm_transformer = TransformerX3(a2d_transformer, t2m_transformer, m2m_transformer)

    all_params = 0
    pc_transformer = sum(param.numel() for param in atm_transformer.parameters())
    print(atm_transformer)
    print("Total parameters of a2d_transformer net: {}".format(pc_transformer))
    all_params += pc_transformer

    print('Total parameters of all models: {}'.format(all_params))

    trainer = TransformerATMTrainerV2(opt, atm_transformer)

    train_dataset_a2d = AudioMotionTokenDatasetV2(opt, train_split_file_a2d)
    val_dataset_a2d = AudioMotionTokenDatasetV2(opt, val_split_file_a2d)

    train_dataset_t2m = TextMotionTokenDatasetV2(opt, train_split_file_t2m, w_vectorizer)
    val_dataset_t2m = TextMotionTokenDatasetV2(opt, val_split_file_t2m, w_vectorizer)

    train_dataset = MixDataset(train_dataset_a2d, train_dataset_t2m)
    val_dataset = MixDataset(val_dataset_a2d, val_dataset_t2m)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
                              shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, drop_last=True, num_workers=4,
                            shuffle=True, pin_memory=True)

    trainer.train(train_loader, val_loader, None)