import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import spacy

from torch.utils.data._utils.collate import default_collate

# import spacy

'''
nlp = spacy.load('en_core_web_sm')
def process_text(sentence):
    sentence = sentence.replace('-', '')
    doc = nlp(sentence)
    word_list = []
    pos_list = []
    for token in doc:
        word = token.text
        if not word.isalpha():
            continue
        if token.pos_ == 'NOUN' or token.pos_ == 'VERB':
            word_list.append(token.lemma_)
        else:
            word_list.append(word)
        pos_list.append(token.pos_)
    return word_list, pos_list

'''


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


class TextMotionTokenDataset(data.Dataset):
    def __init__(self, opt, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer

        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        data_dict = {}
        for name in tqdm(id_list):
            try:
                m_token_list = []
                # Read tokens
                with cs.open(pjoin(opt.data_root, opt.tokenizer_name, '%s.txt' % name), 'r') as f:
                    for line in f.readlines():
                        m_token_list.append(line.strip().split(' '))

                # Read text
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    text_data = []
                    flag = False
                    lines = f.readlines()
                    # if 'train' in split_file:
                    #     lines = lines

                    for line in lines:
                        try:
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            t_tokens = line_split[1].split(' ')
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict['caption'] = caption
                            text_dict['tokens'] = t_tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                m_token_list = [tokens[int(f_tag * 5): int(to_tag * 5)] for tokens in m_token_list]
                                #
                                # if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                #     continue
                                new_name = '%s_%f_%f' % (name, f_tag, to_tag)
                                # new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                # while new_name in data_dict:
                                #     new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'m_token_list': m_token_list,
                                                       'text': [text_dict]}
                                new_name_list.append(new_name)
                        except:
                            # print(line_split)
                            # print(line_split[2], line_split[3], f_tag, to_tag, name)
                            pass

                if flag:
                    data_dict[name] = {'m_token_list': m_token_list,
                                       'text': text_data}
                    new_name_list.append(name)
            except:
                pass
        self.data_dict = data_dict
        self.name_list = new_name_list

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[self.name_list[item]]
        m_token_list, text_list = data['m_token_list'], data['text']
        m_tokens = random.choice(m_token_list)
        m_tokens = [int(token) for token in m_tokens]
        text_data = random.choice(text_list)
        caption, t_tokens = text_data['caption'], text_data['tokens']

        if len(t_tokens) < self.opt.max_text_len:
            t_tokens = ['sos/OTHER'] + t_tokens + ['eos/OTHER']
            sent_len = len(t_tokens)
            t_tokens += ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            t_tokens = t_tokens[:self.opt.max_text_len]
            t_tokens = ['sos/OTHER'] + t_tokens + ['eos/OTHER']
            sent_len = len(t_tokens)
        word_embeddings = []
        word_ids = []
        for i, t_token in enumerate(t_tokens):
            word_emb, _, word_id = self.w_vectorizer[t_token]
            word_embeddings.append(word_emb[None, :])
            if i >= sent_len:
                word_ids.append(self.opt.txt_pad_idx)
            else:
                word_ids.append(word_id)
        word_embeddings = np.concatenate(word_embeddings, axis=0)
        word_ids = np.array(word_ids, dtype=int)
        coin = np.random.choice([False, False, True])
        # print(len(m_tokens))
        if coin:
            # drop one token at the head or tail
            coin2 = np.random.choice([True, False])
            if coin2:
                m_tokens = m_tokens[:-1]
            else:
                m_tokens = m_tokens[1:]
        m_tokens_len = len(m_tokens)

        assert m_tokens_len <= self.opt.max_motion_len, ('m_tokens_len:', m_tokens_len)

        m_tokens = [self.opt.mot_start_idx] + \
                   m_tokens + \
                   [self.opt.mot_end_idx] + \
                   [self.opt.mot_pad_idx] * (self.opt.max_motion_len - len(m_tokens) - 2)
        # print(len(word_embeddings), sent_len, len(m_tokens))
        m_tokens = np.array(m_tokens, dtype=int)

        # max_m_tokens = max(m_tokens)
        # print('dataloader, max m_tokens:', max_m_tokens)
        # if  max_m_tokens> 1027:
        #     print('debug')
        # max_word_ids = max(word_ids)
        # print('dataloader, max max_word_ids:', max_word_ids)
        # if  max_m_tokens> 1027:
        #     print('debug')

        return word_embeddings, word_ids, caption, sent_len, m_tokens, m_tokens_len


class MotionTokenDataset(TextMotionTokenDataset):
    def __getitem__(self, item):
        data = self.data_dict[self.name_list[item]]
        name = self.name_list[item]
        m_token_list, text_list = data['m_token_list'], data['text']
        m_tokens = random.choice(m_token_list)
        m_tokens = [int(token) for token in m_tokens]

        # coin = np.random.choice([False, False, True])
        # print(len(m_tokens))
        # if coin:
        #     # drop one token at the head or tail
        #     coin2 = np.random.choice([True, False])
        #     if coin2:
        #         m_tokens = m_tokens[:-1]
        #     else:
        #         m_tokens = m_tokens[1:]
        m_tokens_len = len(m_tokens)

        m_tokens = [self.opt.mot_start_idx] + \
                   m_tokens + \
                   [self.opt.mot_end_idx] + \
                   [self.opt.mot_pad_idx] * (self.opt.max_motion_len - len(m_tokens) - 2)
        # print(len(word_embeddings), sent_len, len(m_tokens))
        m_tokens = np.array(m_tokens, dtype=int)
        return m_tokens, m_tokens_len, name


class TextMotionTokenDatasetV2(data.Dataset):
    '''
    0924: 尝试把text和motion等长的输出
    '''
    def __init__(self, opt, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer

        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        data_dict = {}
        for name in tqdm(id_list):
            try:
                m_token_list = []
                # Read tokens
                # with cs.open(pjoin(opt.data_root, opt.tokenizer_name, '%s.txt' % name), 'r') as f:
                with cs.open(pjoin(opt.data_root, opt.tokenizer_name_motion, '%s.txt' % name), 'r') as f:
                    for line in f.readlines():
                        m_token_list.append(line.strip().split(' '))

                # Read text
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    text_data = []
                    flag = False
                    lines = f.readlines()
                    # if 'train' in split_file:
                    #     lines = lines

                    for line in lines:
                        try:
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            t_tokens = line_split[1].split(' ')
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict['caption'] = caption
                            text_dict['tokens'] = t_tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                m_token_list = [tokens[int(f_tag * 5): int(to_tag * 5)] for tokens in m_token_list]
                                #
                                # if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                #     continue
                                new_name = '%s_%f_%f' % (name, f_tag, to_tag)
                                # new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                # while new_name in data_dict:
                                #     new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'m_token_list': m_token_list,
                                                       'text': [text_dict]}
                                new_name_list.append(new_name)
                        except:
                            # print(line_split)
                            # print(line_split[2], line_split[3], f_tag, to_tag, name)
                            pass

                if flag:
                    data_dict[name] = {'m_token_list': m_token_list,
                                       'text': text_data}
                    new_name_list.append(name)
            except:
                pass
        self.data_dict = data_dict
        self.name_list = new_name_list

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[self.name_list[item]]
        m_token_list, text_list = data['m_token_list'], data['text']
        m_tokens = random.choice(m_token_list)
        m_tokens = [int(token) for token in m_tokens]
        text_data = random.choice(text_list)
        caption, t_tokens = text_data['caption'], text_data['tokens']

        if len(t_tokens) < self.opt.max_text_len:
            t_tokens = ['sos/OTHER'] + t_tokens + ['eos/OTHER']
            sent_len = len(t_tokens)
            t_tokens += ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            t_tokens = t_tokens[:self.opt.max_text_len]
            t_tokens = ['sos/OTHER'] + t_tokens + ['eos/OTHER']
            sent_len = len(t_tokens)
        word_embeddings = []
        word_ids = []
        for i, t_token in enumerate(t_tokens):
            word_emb, _, word_id = self.w_vectorizer[t_token]
            word_embeddings.append(word_emb[None, :])
            if i >= sent_len:
                if i >= len(m_tokens):
                    word_ids.append(self.opt.txt_pad_idx)
                else:  # 这里把 >len(text) 和 <len(motion)部分, 用长度indicator代替.
                    word_ids.append(self.opt.txt_mlen_idx)
            else:
                word_ids.append(word_id)
        word_embeddings = np.concatenate(word_embeddings, axis=0).astype('float32')
        word_ids = np.array(word_ids, dtype=int)
        coin = np.random.choice([False, False, True])
        # print(len(m_tokens))
        if coin:
            # drop one token at the head or tail
            coin2 = np.random.choice([True, False])
            if coin2:
                m_tokens = m_tokens[:-1]
            else:
                m_tokens = m_tokens[1:]
        m_tokens_len = len(m_tokens)

        assert m_tokens_len <= self.opt.max_motion_len, ('m_tokens_len:', m_tokens_len)

        m_tokens = [self.opt.mot_start_idx] + \
                   m_tokens + \
                   [self.opt.mot_end_idx] + \
                   [self.opt.mot_pad_idx] * (self.opt.max_motion_len - len(m_tokens))  # 保持和text的token长度一致.
                    # [self.opt.mot_pad_idx] * (self.opt.max_motion_len - len(m_tokens) - 2)

        # print(len(word_embeddings), sent_len, len(m_tokens))
        m_tokens = np.array(m_tokens, dtype=int)

        # max_m_tokens = max(m_tokens)
        # print('dataloader, max m_tokens:', max_m_tokens)
        # if  max_m_tokens> 1027:
        #     print('debug')
        # max_word_ids = max(word_ids)
        # print('dataloader, max max_word_ids:', max_word_ids)
        # if  max_m_tokens> 1027:
        #     print('debug')

        return word_embeddings, word_ids, caption, sent_len, m_tokens, m_tokens_len



'''For use of audio-2-motion generative model'''


# class AudioMotionTokenDataset(data.Dataset):
#     def __init__(self, opt, split_file):
#         self.opt = opt
#
#         id_list = []
#         with cs.open(split_file, 'r') as f:
#             for line in f.readlines():
#                 id_list.append(line.strip())
#
#         new_name_list = []
#         data_dict = {}
#         for name in tqdm(id_list):
#             try:
#                 m_token_list = []
#                 a_token_list = []
#                 # Read motion tokens
#                 with cs.open(pjoin(opt.data_root, opt.tokenizer_name_motion, '%s.txt' % name), 'r') as f:
#                     for line in f.readlines():
#                         m_token_list.append(line.strip().split(' '))
#                 # Read audio tokens
#                 with cs.open(pjoin(opt.data_root, opt.tokenizer_name_audio, '%s.txt' % name), 'r') as f:
#                     for line in f.readlines():
#                         a_token_list.append(line.strip().split(' '))
#
#                 data_dict[name] = {'m_token_list': m_token_list,
#                                    'a_token_list': a_token_list}
#                 new_name_list.append(name)
#             except:
#                 pass
#         self.data_dict = data_dict
#         self.name_list = new_name_list
#
#     def __len__(self):
#         return len(self.data_dict)
#
#     def __getitem__(self, item):
#         data = self.data_dict[self.name_list[item]]
#         # m_token_list, text_list = data['m_token_list'], data['text']
#         m_token_list, a_token_list = data['m_token_list'], data['a_token_list']
#         m_tokens = random.choice(m_token_list)
#         m_tokens = [int(token) for token in m_tokens]
#         a_tokens = random.choice(a_token_list)
#         a_tokens = [int(token) for token in a_tokens]
#         # text_data = random.choice(text_list)
#         # caption, t_tokens = text_data['caption'], text_data['tokens']
#
#         m_tokens_len = len(m_tokens)
#         m_tokens = [self.opt.mot_start_idx] + \
#                    m_tokens + \
#                    [self.opt.mot_end_idx] + \
#                    [self.opt.mot_pad_idx] * (self.opt.max_motion_len - len(m_tokens) - 2)
#         m_tokens = np.array(m_tokens, dtype=int)
#
#         a_tokens_len = len(a_tokens)
#         a_tokens = [self.opt.mot_start_idx] + \
#                    a_tokens + \
#                    [self.opt.mot_end_idx] + \
#                    [self.opt.mot_pad_idx] * (self.opt.max_motion_len - len(a_tokens) - 2)
#         a_tokens = np.array(a_tokens, dtype=int)
#         # return word_embeddings, word_ids, caption, sent_len, m_tokens, m_tokens_len
#         return 0, a_tokens, 0, a_tokens_len, m_tokens, m_tokens_len


class AudioMotionTokenDatasetV2(data.Dataset):
    def __init__(self, opt, split_file):
        """
        chunk dataloader for audio and dance translation.
        """
        self.opt = opt
        self.data_motion = []
        self.data_audio = []
        self.lengths = []
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        for name in tqdm(id_list):
            try:
                # Read motion tokens
                with cs.open(pjoin(opt.data_root, opt.tokenizer_name_motion, '%s.txt' % name), 'r') as f:
                    for line in f.readlines():
                        m_token = line.strip().split(' ')
                        m_token = np.array(m_token, dtype=int)  # change it to array and meet the XXX
                # # Read audio tokens
                # with cs.open(pjoin(opt.data_root, opt.tokenizer_name_audio, '%s.txt' % name), 'r') as f:
                #     for line in f.readlines():
                #         a_token = line.strip().split(' ')
                #         a_token = np.array(a_token, dtype=int)  # change it to array and meet the XXX
                # read audio feature 7.5fps
                audio_feature = np.load(pjoin(opt.audio_dir, name + '.npy')).astype('float32')
                # assert audio_feature.shape[0] == a_token.shape[0] # 看看audio的feature长度和这个token差距大不大.

                min_token_lenth = min(m_token.shape[0], audio_feature.shape[0])
                m_token = m_token[:min_token_lenth]
                audio_feature = audio_feature[:min_token_lenth]
                if min_token_lenth < opt.window_size:
                    continue
                self.lengths.append(m_token.shape[0] - opt.window_size)
                self.data_motion.append(m_token)
                self.data_audio.append(audio_feature)
            except:
                # Some motion may not exist in KIT dataset
                pass

        self.cumsum = np.cumsum([0] + self.lengths)
        print("Total number of motions {}, snippets {}".format(len(self.data_motion), self.cumsum[-1]))

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        m_token = self.data_motion[motion_id][idx:idx + self.opt.window_size]
        # m_token = [self.opt.mot_start_idx] + m_token.tolist()
        m_token = np.array(m_token, dtype=int)

        audio_feature = self.data_audio[motion_id][idx:idx + self.opt.window_size]
        # audio_feature = [self.opt.aud_start_idx] + audio_feature.tolist()
        audio_feature = np.array(audio_feature)
        # return 0, a_token, 0, 0,             m_token, 0
        # return 0, a_tokens, 0, a_tokens_len, m_tokens, m_tokens_len
        a_feature_len = audio_feature.shape[0]
        m_tokens_len = m_token.shape[0]
        return audio_feature, 0, 0, a_feature_len, m_token, m_tokens_len
        # return word_embeddings, word_ids, caption, sent_len, m_tokens, m_tokens_len


class MixDataset(data.Dataset):
    def __init__(self, a2d_dataset, t2m_dataset):
        self.a2d_dataset = a2d_dataset
        self.t2m_dataset = t2m_dataset
        # assert len(t2m_dataset) < len(a2d_dataset)

    def __getitem__(self, index):
        a2d = self.a2d_dataset.__getitem__(index % len(self.a2d_dataset))
        # _, a_token, _, _, m_token, _ = a2d
        # _, a_token, _, _, d_token, _ = a2d
        audio_feature, _, _, a_feature_len, d_token, d_token_len = a2d

        t2m = self.t2m_dataset.__getitem__(index)
        # word_embeddings, word_ids, caption, sent_len, m_tokens, m_tokens_len = t2m
        # _, word_ids, _, _, m_tokens, _ = t2m
        word_embeddings, word_ids, caption, sent_len, m_tokens, m_tokens_len = t2m

        # audio_tokens, d_tokens, word_tokens, m_tokens = batch_data # from trains_x
        # return a_token, d_token, word_ids, m_tokens
        return audio_feature, a_feature_len, d_token, word_embeddings, word_ids, sent_len, m_tokens

    def __len__(self):
        return len(self.t2m_dataset)


'''For use of training text-2-motion generative model'''


class Text2MotionDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        min_motion_len = 40 if self.opt.dataset_name == 't2m' else 24

        joints_num = opt.joints_num

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 600):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * 20): int(to_tag * 20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text': [text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                # Some motion may not exist in KIT dataset
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3: 4 + (
                    joints_num - 1) * 9] / 1.0
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = std[
                                                                                       4 + (joints_num - 1) * 9: 4 + (
                                                                                               joints_num - 1) * 9 + joints_num * 3] / 1.0
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = std[
                                                              4 + (
                                                                          joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias

            assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(opt.meta_dir, 'std.npy'), std)

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.opt.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        len_gap = (m_length - self.max_length) // self.opt.unit_length

        if self.opt.is_train:
            if m_length != self.max_length:
                # print("Motion original length:%d_%d"%(m_length, len(motion)))
                if self.opt.unit_length < 10:
                    coin2 = np.random.choice(['single', 'single', 'double'])
                else:
                    coin2 = 'single'
                if len_gap == 0 or (len_gap == 1 and coin2 == 'double'):
                    m_length = self.max_length
                    idx = random.randint(0, m_length - self.max_length)
                    motion = motion[idx:idx + self.max_length]
                else:
                    if coin2 == 'single':
                        n_m_length = self.max_length + self.opt.unit_length * len_gap
                    else:
                        n_m_length = self.max_length + self.opt.unit_length * (len_gap - 1)
                    idx = random.randint(0, m_length - n_m_length)
                    motion = motion[idx:idx + self.max_length]
                    m_length = n_m_length
                # print(len_gap, idx, coin2)
        else:
            if self.opt.unit_length < 10:
                coin2 = np.random.choice(['single', 'single', 'double'])
            else:
                coin2 = 'single'

            if coin2 == 'double':
                m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
            elif coin2 == 'single':
                m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
            idx = random.randint(0, len(motion) - m_length)
            motion = motion[idx:idx + m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length


'''For use of training text motion matching model, and evaluations'''


class Text2MotionDatasetV2(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name == 't2m' else 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        # id_list = id_list[:250]  # debug

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 600):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * 20): int(to_tag * 20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text': [text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)


from tools.pose2feature_converter import Pose2FeatureConverter
class Text2MotionDatasetV2_60fps20j(data.Dataset):
    """
    尝试用60fps, 24joint的去测试
    这里写一个转换方程, 把60fps24j的, 和 20fps22j的都输出出去.
    """
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        self.MotionConvert = Pose2FeatureConverter()
        min_motion_len = 40 if self.opt.dataset_name == 't2m' else 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        # id_list = id_list[:250] # debug 1007

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                # if (len(motion)) < min_motion_len or (len(motion) >= 600):
                if (len(motion)) < min_motion_len or (len(motion) >= 1800):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * 20): int(to_tag * 20)]
                                # if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 600):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text': [text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        # convert 60fps24j to 20fps22j
        motion_20fps22j = self.MotionConvert.normed_f24j_to_normed_f22j(motion)
        m_length_20fps22j = motion_20fps22j.shape[0]

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        if m_length_20fps22j < 200:
            motion_20fps22j = np.concatenate([motion_20fps22j,
                                     np.zeros((200 - m_length_20fps22j, motion_20fps22j.shape[1]))
                                     ], axis=0)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        # return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)

        return word_embeddings, pos_one_hots, caption, sent_len, motion_20fps22j, m_length_20fps22j, '_'.join(tokens), motion, m_length



'''For use of training baseline motion-2-text generative model'''


class Motion2TextBaselineDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_motion_frame = opt.max_motion_length
        self.pointer = 0
        min_motion_len = 40 if self.opt.dataset_name == 't2m' else 24

        joints_num = opt.joints_num

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 600):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * 20): int(to_tag * 20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text': [text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                # Some motion may not exist in KIT dataset
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3: 4 + (
                    joints_num - 1) * 9] / 1.0
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = std[
                                                                                       4 + (joints_num - 1) * 9: 4 + (
                                                                                               joints_num - 1) * 9 + joints_num * 3] / 1.0
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = std[
                                                              4 + (
                                                                          joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias

            assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(opt.meta_dir, 'std.npy'), std)

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        word_embeddings = []

        word_ids = []
        for i, t_token in enumerate(tokens):
            word_emb, _, word_id = self.w_vectorizer[t_token]
            word_embeddings.append(word_emb[None, :])
            if i >= sent_len:
                word_ids.append(self.opt.txt_pad_idx)
            else:
                word_ids.append(word_id)
        word_embeddings = np.concatenate(word_embeddings, axis=0)
        word_ids = np.array(word_ids, dtype=int)

        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]

        if m_length < self.max_motion_frame:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_frame - m_length, motion.shape[1]))
                                     ], axis=0)
        else:
            m_length = self.max_motion_frame
            motion = motion[:self.max_motion_frame]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return word_embeddings, word_ids, caption, sent_len, motion, m_length


'''For use of training baseline models'''


class Text2MotionDatasetBaseline(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name == 't2m' else 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:200]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 600):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * 20): int(to_tag * 20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text': [text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        len_gap = (m_length - self.max_length) // self.opt.unit_length

        if m_length != self.max_length:
            # print("Motion original length:%d_%d"%(m_length, len(motion)))
            if self.opt.unit_length < 10:
                coin2 = np.random.choice(['single', 'single', 'double'])
            else:
                coin2 = 'single'
            if len_gap == 0 or (len_gap == 1 and coin2 == 'double'):
                m_length = self.max_length
                s_idx = random.randint(0, m_length - self.max_length)
            else:
                if coin2 == 'single':
                    n_m_length = self.max_length + self.opt.unit_length * len_gap
                else:
                    n_m_length = self.max_length + self.opt.unit_length * (len_gap - 1)
                s_idx = random.randint(0, m_length - n_m_length)
                m_length = n_m_length
        else:
            s_idx = 0

        src_motion = motion[s_idx: s_idx + m_length]
        tgt_motion = motion[s_idx: s_idx + self.max_length]

        "Z Normalization"
        src_motion = (src_motion - self.mean) / self.std
        tgt_motion = (tgt_motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            src_motion = np.concatenate([src_motion,
                                         np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                         ], axis=0)
        # print(m_length, src_motion.shape, tgt_motion.shape)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return word_embeddings, caption, sent_len, src_motion, tgt_motion, m_length


class MotionDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        joints_num = opt.joints_num

        self.data = []
        self.lengths = []
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if motion.shape[0] < opt.window_size:  # remove length less than 64.
                    continue
                self.lengths.append(motion.shape[0] - opt.window_size)
                self.data.append(motion)
            except:
                # Some motion may not exist in KIT dataset
                pass

        self.cumsum = np.cumsum([0] + self.lengths)

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3: 4 + (
                    joints_num - 1) * 9] / 1.0
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = std[
                                                                                       4 + (joints_num - 1) * 9: 4 + (
                                                                                               joints_num - 1) * 9 + joints_num * 3] / 1.0
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = std[
                                                              4 + (
                                                                          joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias

            assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(opt.meta_dir, 'std.npy'), std)

        self.mean = mean
        self.std = std
        print("Total number of motions {}, snippets {}".format(len(self.data), self.cumsum[-1]))

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        motion = self.data[motion_id][idx:idx + self.opt.window_size]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion


class AudioDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        # joints_num = opt.joints_num

        self.data = []
        self.lengths = []
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        for name in tqdm(id_list):
            try:
                audio = np.load(pjoin(opt.audio_dir, name + '.npy'))
                if audio.shape[0] < opt.window_size:
                    continue
                self.lengths.append(audio.shape[0] - opt.window_size)
                self.data.append(audio)
            except:
                # Some motion may not exist in KIT dataset
                pass

        self.cumsum = np.cumsum([0] + self.lengths)

        np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
        np.save(pjoin(opt.meta_dir, 'std.npy'), std)
        self.mean = mean
        self.std = std
        print("Total number of audios {}, snippets {}".format(len(self.data), self.cumsum[-1]))

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            audio_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[audio_id] - 1
        else:
            audio_id = 0
            idx = 0
        audio = self.data[audio_id][idx:idx + self.opt.window_size]
        "Z Normalization"
        audio = (audio - self.mean) / self.std

        return audio


class MotionTokenizeDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        min_motion_len = 40 if self.opt.dataset_name == 't2m' else 24

        joints_num = opt.joints_num

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len: # or (len(motion) >= 600):  # 0907: dance data is longer.
                    continue

                data_dict[name] = {'motion': motion,
                                   'length': len(motion),
                                   'name': name}
                new_name_list.append(name)
                length_list.append(len(motion))
            except:
                # Some motion may not exist in KIT dataset
                pass

        # name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3: 4 + (
                    joints_num - 1) * 9] / 1.0
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = std[
                                                                                       4 + (joints_num - 1) * 9: 4 + (
                                                                                               joints_num - 1) * 9 + joints_num * 3] / 1.0
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = std[
                                                              4 + (
                                                                          joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias

            assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(opt.meta_dir, 'std.npy'), std)

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = new_name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        name = self.name_list[item]
        data = self.data_dict[name]
        motion, m_length = data['motion'], data['length']

        m_length = (m_length // self.opt.unit_length) * self.opt.unit_length

        idx = random.randint(0, len(motion) - m_length)  # not suitable for dance-music task.
        motion = motion[idx:idx + m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion, name


class DanceTokenizeDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        # min_motion_len = 40 if self.opt.dataset_name == 't2m' else 24
        min_motion_len = 40

        joints_num = opt.joints_num

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                # if (len(motion)) < min_motion_len or (len(motion) >= 600):  # 0907: dance data is longer.
                if len(motion) < min_motion_len:  # 0907: dance data is longer.
                    continue

                data_dict[name] = {'motion': motion,
                                   'length': len(motion),
                                   'name': name}
                new_name_list.append(name)
                length_list.append(len(motion))
            except:
                # Some motion may not exist in KIT dataset
                pass

        # name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3: 4 + (
                    joints_num - 1) * 9] / 1.0
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = std[
                                                                                       4 + (joints_num - 1) * 9: 4 + (
                                                                                               joints_num - 1) * 9 + joints_num * 3] / 1.0
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = std[
                                                              4 + (
                                                                          joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias

            assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(opt.meta_dir, 'std.npy'), std)

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = new_name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        name = self.name_list[item]
        data = self.data_dict[name]
        motion, m_length = data['motion'], data['length']

        m_length = (m_length // self.opt.unit_length) * self.opt.unit_length

        # idx = random.randint(0, len(motion) - m_length)  # not suitable for dance-music task.
        idx = 0
        motion = motion[idx:idx + m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion, name


# from tools.ml3d_preprocess_crop import get_clip_idx
# class DanceTokenizeDatasetV2(data.Dataset):
#     def __init__(self, opt, mean, std, split_file):
#         self.opt = opt
#         # min_motion_len = 40 if self.opt.dataset_name == 't2m' else 24
#         min_motion_len = 40
#
#         joints_num = opt.joints_num
#
#         data_dict = {}
#         id_list = []
#         with cs.open(split_file, 'r') as f:
#             for line in f.readlines():
#                 id_list.append(line.strip())
#
#         ml3d_id_list = []
#         ml3d_split_file = pjoin(opt.data_root, 'ml3d_all.txt')
#         with cs.open(ml3d_split_file, 'r') as f:
#             for line in f.readlines():
#                 ml3d_id_list.append(line.strip())
#
#         new_name_list = []
#         length_list = []
#         for name in tqdm(id_list):
#             try:
#                 motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
#                 # if (len(motion)) < min_motion_len or (len(motion) >= 600):  # 0907: dance data is longer.
#                 if len(motion) < min_motion_len:  # 0907: dance data is longer.
#                     continue
#
#                 # now crop the ml3d motion data
#                 if name in ml3d_id_list:
#                     start_idx, end_idx = get_clip_idx(motion)
#                     if end_idx - start_idx < min_motion_len:
#                         continue
#                     if end_idx - start_idx < int(motion.shape[0] * 0.6):
#                         continue
#                     motion = motion[start_idx: end_idx]
#
#                 data_dict[name] = {'motion': motion,
#                                    'length': len(motion),
#                                    'name': name}
#                 new_name_list.append(name)
#                 length_list.append(len(motion))
#             except:
#                 # Some motion may not exist in KIT dataset
#                 pass
#
#         # name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
#
#         if opt.is_train:
#             # root_rot_velocity (B, seq_len, 1)
#             std[0:1] = std[0:1] / opt.feat_bias
#             # root_linear_velocity (B, seq_len, 2)
#             std[1:3] = std[1:3] / opt.feat_bias
#             # root_y (B, seq_len, 1)
#             std[3:4] = std[3:4] / opt.feat_bias
#             # ric_data (B, seq_len, (joint_num - 1)*3)
#             std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
#             # rot_data (B, seq_len, (joint_num - 1)*6)
#             std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3: 4 + (
#                     joints_num - 1) * 9] / 1.0
#             # local_velocity (B, seq_len, joint_num*3)
#             std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = std[
#                                                                                        4 + (joints_num - 1) * 9: 4 + (
#                                                                                                joints_num - 1) * 9 + joints_num * 3] / 1.0
#             # foot contact (B, seq_len, 4)
#             std[4 + (joints_num - 1) * 9 + joints_num * 3:] = std[
#                                                               4 + (
#                                                                           joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias
#
#             assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
#             np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
#             np.save(pjoin(opt.meta_dir, 'std.npy'), std)
#
#         self.mean = mean
#         self.std = std
#         self.length_arr = np.array(length_list)
#         self.data_dict = data_dict
#         self.name_list = new_name_list
#
#     def inv_transform(self, data):
#         return data * self.std + self.mean
#
#     def __len__(self):
#         return len(self.data_dict)
#
#     def __getitem__(self, item):
#         name = self.name_list[item]
#         data = self.data_dict[name]
#         motion, m_length = data['motion'], data['length']
#
#         m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
#
#         # idx = random.randint(0, len(motion) - m_length)  # not suitable for dance-music task.
#         idx = 0
#         motion = motion[idx:idx + m_length]
#
#         "Z Normalization"
#         motion = (motion - self.mean) / self.std
#
#         return motion, name



class AudioTokenizeDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        # min_motion_len = 40 if self.opt.dataset_name == 't2m' else 24
        min_audio_len = 40

        # joints_num = opt.joints_num

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                audio = np.load(pjoin(opt.audio_dir, name + '.npy'))
                # if (len(motion)) < min_motion_len or (len(motion) >= 600):
                if (len(audio)) < min_audio_len:
                    continue

                data_dict[name] = {'audio': audio,
                                   'length': len(audio),
                                   'name': name}
                new_name_list.append(name)
                length_list.append(len(audio))
            except:
                # Some motion may not exist in KIT dataset
                pass

        # name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = new_name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        name = self.name_list[item]
        data = self.data_dict[name]
        audio, m_length = data['audio'], data['length']

        m_length = (m_length // self.opt.unit_length) * self.opt.unit_length

        idx = random.randint(0, len(audio) - m_length)
        audio = audio[idx:idx + m_length]

        "Z Normalization"
        audio = (audio - self.mean) / self.std

        return audio, name


