from data.dataset import Text2MotionDatasetV2, Motion2TextEvalDataset, collate_fn, Text2MotionDatasetV2_60fps20j, TextMotionTokenDatasetV2_forT2MfinalEval
from utils.word_vectorizer import WordVectorizer, WordVectorizerV2
import numpy as np
from os.path import join as pjoin
from torch.utils.data import DataLoader
from utils.get_opt import get_opt

def get_dataset_motion_loader(opt_path, batch_size, device):
    opt = get_opt(opt_path, device)

    # Configurations of T2M dataset and KIT dataset is almost the same
    if opt.dataset_name == 't2m' or opt.dataset_name == 'kit':
        print('Loading dataset %s ...' % opt.dataset_name)

        mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
        std = np.load(pjoin(opt.meta_dir, 'std.npy'))

        w_vectorizer = WordVectorizer('./glove', 'our_vab')
        split_file = pjoin(opt.data_root, 'test.txt')
        dataset = Text2MotionDatasetV2(opt, mean, std, split_file, w_vectorizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, drop_last=True,
                                collate_fn=collate_fn, shuffle=True)

    elif opt.dataset_name == 'aistppml3d':
        print('Loading dataset %s ...' % opt.dataset_name)

        # 1007: cp the meta from vqvae checkpoint folder to here.
        mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
        std = np.load(pjoin(opt.meta_dir, 'std.npy'))

        w_vectorizer = WordVectorizer('./glove', 'our_vab')
        split_file = pjoin(opt.data_root, 'test.txt')
        dataset = Text2MotionDatasetV2_60fps20j(opt, mean, std, split_file, w_vectorizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, drop_last=True,
                                collate_fn=collate_fn, shuffle=True)
    else:
        raise KeyError('Dataset not Recognized !!')

    print('Ground Truth Dataset Loading Completed!!!')
    return dataloader, dataset

def get_TXM_dataset_motion_loader(opt_path, batch_size, device):
    """
    :param opt_path:
    :param batch_size:
    :param device:
    :return:
    专门准备一份给atm txm用的dataloader, 因为他们都是token based的.
    """
    opt = get_opt(opt_path, device)

    # Configurations of T2M dataset and KIT dataset is almost the same
    if opt.dataset_name == 'aistppml3d':
        print('Loading dataset %s ...' % opt.dataset_name)

        # 1007: cp the meta from vqvae checkpoint folder to here.
        mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
        std = np.load(pjoin(opt.meta_dir, 'std.npy'))

        w_vectorizer = WordVectorizerV2('./glove', 'our_vab')
        split_file = pjoin(opt.data_root, 'test.txt')


        opt.n_mot_vocab = opt.codebook_size + 3
        opt.mot_start_idx = opt.codebook_size
        opt.mot_end_idx = opt.codebook_size + 1
        opt.mot_pad_idx = opt.codebook_size + 2

        opt.n_txt_vocab = len(w_vectorizer) + 2
        _, _, opt.txt_start_idx = w_vectorizer['sos/OTHER']
        _, _, opt.txt_end_idx = w_vectorizer['eos/OTHER']
        opt.txt_mlen_idx = len(w_vectorizer)
        opt.txt_pad_idx = len(w_vectorizer) + 1


        opt.joints_num = 24
        # opt.window_size = 30  # because min-len of dance is 130 (20fps) -> 30x4. 425 (60fps) -> 50x8
        opt.window_size = 50  # because min-len of dance is 130 (20fps) -> 30x4. 425 (60fps) -> 50x8
        # 0924: window size可以考虑加长一下.

        # opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        # opt.joints_num = 24  # 22
        opt.max_motion_len = 84  # 84 = 55x3/8x4# 55
        opt.max_text_len = 84 # 84 = 55x3/8x4# 55

        # dataset = Text2MotionDatasetV2_60fps20j(opt, mean, std, split_file, w_vectorizer)
        dataset = TextMotionTokenDatasetV2_forT2MfinalEval(opt, split_file, w_vectorizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, drop_last=True,
                                collate_fn=collate_fn, shuffle=True)
    else:
        raise KeyError('Dataset not Recognized !!')

    print('Ground Truth Dataset Loading Completed!!!')
    return dataloader, dataset

