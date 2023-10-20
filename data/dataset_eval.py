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

class TextMotionTokenDatasetV2_forT2MfinalEval(data.Dataset):
    '''
    0924: 尝试把text和motion等长的输出
    1009: 这里额外添加一个token的输出.
    '''
    def __init__(self, opt, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer

        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        id_list = id_list[:100]  # debug

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
                if i >= len(m_tokens):
                    word_ids.append(self.opt.txt_pad_idx)
                else:  # 这里把 >len(text) 和 <len(motion)部分, 用长度indicator代替.
                    word_ids.append(self.opt.txt_mlen_idx)
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

        return word_embeddings, word_ids, caption, sent_len, m_tokens, m_tokens_len, '_'.join(t_tokens)



class Motion2TextEvalDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_frame = opt.max_motion_frame
        min_motion_len = 40 if self.opt.dataset_name == 't2m' else 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:40]

        new_name_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 600):
                    continue
            except:
                # some motion are not include
                continue

            m_token_list = []

            with cs.open(pjoin(opt.m_token_dir, name + '.txt'), 'r') as f:
                for line in f.readlines():
                    m_token_list.append(line.strip().split(' '))
                    # if line.__contains__("419416"):
                    #     print(name)
                    #     print(line)

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
                            n_m_token_list = [tokens[int(f_tag * 5): int(to_tag * 5)] for tokens in m_token_list]

                            if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                continue
                            new_name = "%s_%f_%f" % (name, f_tag, to_tag)
                            data_dict[new_name] = {'motion': n_motion,
                                                   'm_token_list': n_m_token_list,
                                                   'length': len(n_motion),
                                                   'text': [text_dict]}
                            new_name_list.append(new_name)
                        except:
                            print(line_split)
                            print(line_split[2], line_split[3], f_tag, to_tag, name)
                            # break

            if flag:
                data_dict[name] = {'motion': motion,
                                   'm_token_list': m_token_list,
                                   'length': len(motion),
                                   'text': text_data}
                new_name_list.append(name)
        # except:
        # pass

        self.mean = mean
        self.std = std
        self.data_dict = data_dict
        self.name_list = new_name_list
        # print(len(se))

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[self.name_list[item]]
        motion, m_token_list, text_list = data['motion'], data['m_token_list'], data['text']

        m_tokens = random.choice(m_token_list)
        m_tokens = [int(token) for token in m_tokens]

        text_data = random.choice(text_list)
        caption, t_tokens = text_data['caption'], text_data['tokens']
        all_captions = [' '.join(
            [token.split('/')[0] for token in text_dic['tokens']]
        ) for text_dic in text_list]

        if len(t_tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + t_tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = t_tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        word_ids = []
        for i, token in enumerate(tokens):
            word_emb, pos_oh, word_id = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
            if i >= sent_len:
                word_ids.append(self.opt.txt_pad_idx)
            else:
                word_ids.append(word_id)
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)
        word_ids = np.array(word_ids, dtype=int)

        m_tokens = [self.opt.mot_start_idx] + \
                   m_tokens + \
                   [self.opt.mot_end_idx] + \
                   [self.opt.mot_pad_idx] * (self.opt.max_motion_token - len(m_tokens) - 2)

        # print(len(word_embeddings), sent_len, len(m_tokens))
        m_tokens = np.array(m_tokens, dtype=int)

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        m_length = len(motion)

        if m_length < self.max_motion_frame:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_frame - m_length, motion.shape[1]))
                                     ], axis=0)
        else:
            m_length = self.max_motion_frame
            motion = motion[:self.max_motion_frame]

        if len(all_captions) > 3:
            all_captions = all_captions[:3]
        elif len(all_captions) == 2:
            all_captions = all_captions + all_captions[0:1]
        elif len(all_captions) == 1:
            all_captions = all_captions * 3

        return word_embeddings, pos_one_hots, word_ids, caption, sent_len, motion, m_tokens, m_length, all_captions
        # return word_embeddings, pos_one_hots, caption, sent_len, motion, m_tokens, m_length, all_captions


# class Motion2AudioEvalDataset(data.Dataset):
#     def __init__(self, opt, mean, std, split_file):
#         self.opt = opt
#         self.max_length = 20
#         self.pointer = 0
#         # self.max_motion_frame = opt.max_motion_frame
#         min_motion_len = 40   # if self.opt.dataset_name == 't2m' else 24
#
#         data_dict = {}
#         id_list = []
#         with cs.open(split_file, 'r') as f:
#             for line in f.readlines():
#                 id_list.append(line.strip())
#         # id_list = id_list[:40]
#
#         new_name_list = []
#         for name in tqdm(id_list):
#             # try:
#             motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
#             if (len(motion)) < min_motion_len:  # or (len(motion) >= 600):
#                 continue
#
#             # m_token_list = []
#             # a_token_list = []
#
#             with cs.open(pjoin(opt.data_root, opt.tokenizer_name_motion, name + '.txt'), 'r') as f:
#                 for line in f.readlines():
#                     # m_token_list.append(line.strip().split(' '))
#                     m_token = line.strip().split(' ')
#                     # if line.__contains__("419416"):
#                     #     print(name)
#                     #     print(line)
#             with cs.open(pjoin(opt.data_root, opt.tokenizer_name_audio, name + '.txt'), 'r') as f:
#                 for line in f.readlines():
#                     # a_token_list.append(line.strip().split(' '))
#                     a_token = line.strip().split(' ')
#
#
#             data_dict[name] = {'motion': motion,
#                                # 'm_token_list': m_token_list,
#                                # 'a_token_list': a_token_list,
#                                'm_token': m_token,
#                                'a_token': a_token,
#                                'length': len(motion),
#                                }
#             new_name_list.append(name)
#         # except:
#         # pass
#
#         self.mean = mean
#         self.std = std
#         self.data_dict = data_dict
#         self.name_list = new_name_list
#         # print(len(se))
#
#     def inv_transform(self, data):
#         return data * self.std + self.mean
#
#     def __len__(self):
#         return len(self.data_dict)
#
#     def __getitem__(self, item):
#         data = self.data_dict[self.name_list[item]]
#         # motion, m_token_list, a_token_list = data['motion'], data['m_token_list'], data['a_token_list']
#         motion, m_tokens, a_tokens = data['motion'], data['m_token'], data['a_token']
#
#         # m_tokens = random.choice(m_token_list)
#         # m_tokens = [int(token) for token in m_tokens]
#         #
#         # a_tokens = random.choice(a_token_list)
#         # a_tokens = [int(token) for token in a_tokens]
#         #
#         # m_tokens = [self.opt.mot_start_idx] + \
#         #            m_tokens + \
#         #            [self.opt.mot_end_idx] + \
#         #            [self.opt.mot_pad_idx] * (self.opt.max_motion_token - len(m_tokens) - 2)
#         # a_tokens = [self.opt.mot_start_idx] + \
#         #            a_tokens + \
#         #            [self.opt.mot_end_idx] + \
#         #            [self.opt.mot_pad_idx] * (self.opt.max_motion_token - len(a_tokens) - 2)
#
#         # print(len(word_embeddings), sent_len, len(m_tokens))
#         a_tokens = [self.opt.txt_start_idx] + a_tokens #.tolist()
#         a_tokens = np.array(a_tokens, dtype=int)
#
#         m_tokens = [self.opt.mot_start_idx] + m_tokens #.tolist()
#         m_tokens = np.array(m_tokens, dtype=int)
#
#         "Z Normalization"
#         motion = (motion - self.mean) / self.std
#
#         m_length = len(motion)
#
#         # if m_length < self.max_motion_frame:
#         #     motion = np.concatenate([motion,
#         #                              np.zeros((self.max_motion_frame - m_length, motion.shape[1]))
#         #                              ], axis=0)
#         # else:
#         #     m_length = self.max_motion_frame
#         #     motion = motion[:self.max_motion_frame]
#
#         return 0, 0, a_tokens, self.name_list[item], m_length, motion, m_tokens, m_length, 0
#         # return 0, 0, a_tokens, str(a_tokens), m_length, motion, m_tokens, m_length, 0
#         # return word_embeddings, pos_one_hots, word_ids, caption, sent_len, motion, m_tokens, m_length, all_captions
#         # return word_embeddings, pos_one_hots, caption, sent_len, motion, m_tokens, m_length, all_captions



class Motion2AudioEvalDataset4ATM(data.Dataset):
    """
    改成aud_start_idx了, 这样和text的区分开.
    1022- 重新添加了audio_feature
    """
    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        self.max_length = 20
        self.pointer = 0
        # self.max_motion_frame = opt.max_motion_frame
        min_motion_len = 40   # if self.opt.dataset_name == 't2m' else 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:40]

        new_name_list = []
        for name in tqdm(id_list):
            # try:
            motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
            if (len(motion)) < min_motion_len:  # or (len(motion) >= 600):
                continue

            # m_token_list = []
            # a_token_list = []

            with cs.open(pjoin(opt.data_root, opt.tokenizer_name_motion, name + '.txt'), 'r') as f:
                for line in f.readlines():
                    # m_token_list.append(line.strip().split(' '))
                    m_token = line.strip().split(' ')
                    # if line.__contains__("419416"):
                    #     print(name)
                    #     print(line)
            # with cs.open(pjoin(opt.data_root, opt.tokenizer_name_audio, name + '.txt'), 'r') as f:
            #     for line in f.readlines():
            #         # a_token_list.append(line.strip().split(' '))
            #         a_token = line.strip().split(' ')

            # read audio feature 7.5fps
            audio_feature = np.load(pjoin(opt.audio_dir, name + '.npy')).astype('float32')

            data_dict[name] = {'motion': motion,
                               # 'm_token_list': m_token_list,
                               # 'a_token_list': a_token_list,
                               'm_token': m_token,
                               # 'a_token': a_token,
                               'audio_feature': audio_feature,
                               'length': len(motion),
                               }
            new_name_list.append(name)
        # except:
        # pass

        self.mean = mean
        self.std = std
        self.data_dict = data_dict
        self.name_list = new_name_list
        # print(len(se))

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[self.name_list[item]]
        # motion, m_token_list, a_token_list = data['motion'], data['m_token_list'], data['a_token_list']
        motion, m_tokens, audio_feature= data['motion'], data['m_token'], data['audio_feature']

        # a_tokens = [self.opt.aud_start_idx] + a_tokens #.tolist()
        # a_tokens = np.array(a_tokens, dtype=int)

        audio_feature = np.array(audio_feature)

        # m_tokens = [self.opt.mot_start_idx] + m_tokens #.tolist()
        m_tokens = np.array(m_tokens, dtype=int)

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        m_length = len(motion)

        # if m_length < self.max_motion_frame:
        #     motion = np.concatenate([motion,
        #                              np.zeros((self.max_motion_frame - m_length, motion.shape[1]))
        #                              ], axis=0)
        # else:
        #     m_length = self.max_motion_frame
        #     motion = motion[:self.max_motion_frame]

        audio_len = audio_feature.shape[0]
        m_tokens_len = m_tokens.shape[0]
        return audio_feature, 0, 0, self.name_list[item], audio_len, motion, m_tokens, m_tokens_len, 0
        # return 0, 0, a_tokens, self.name_list[item], m_length, motion, m_tokens, m_length, 0
        # return 0, 0, a_tokens, str(a_tokens), m_length, motion, m_tokens, m_length, 0
        # return word_embeddings, pos_one_hots, word_ids, caption, sent_len, motion, m_tokens, m_length, all_captions
        # return word_embeddings, pos_one_hots, caption, sent_len, motion, m_tokens, m_length, all_captions


class WildAudioEvalDataset4ATM(data.Dataset):
    """
    适配只有audio情况的dataloader.
    """
    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        # self.max_length = 20
        self.pointer = 0
        # self.max_motion_frame = opt.max_motion_frame
        # min_motion_len = 40   # if self.opt.dataset_name == 't2m' else 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:40]

        new_name_list = []
        for name in tqdm(id_list):
            # try:
            # motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
            # if (len(motion)) < min_motion_len:  # or (len(motion) >= 600):
            #     continue

            # m_token_list = []
            # a_token_list = []

            # with cs.open(pjoin(opt.data_root, opt.tokenizer_name_motion, name + '.txt'), 'r') as f:
            #     for line in f.readlines():
            #         # m_token_list.append(line.strip().split(' '))
            #         m_token = line.strip().split(' ')
            #         # if line.__contains__("419416"):
            #         #     print(name)
            #         #     print(line)
            # with cs.open(pjoin(opt.data_root, opt.tokenizer_name_audio, name + '.txt'), 'r') as f:
            #     for line in f.readlines():
            #         # a_token_list.append(line.strip().split(' '))
            #         a_token = line.strip().split(' ')

            # read audio feature 7.5fps
            audio_feature = np.load(pjoin(opt.audio_dir, name + '.npy')).astype('float32')

            data_dict[name] = {'motion': 0,
                               # 'm_token_list': m_token_list,
                               # 'a_token_list': a_token_list,
                               'm_token': 0,
                               # 'a_token': a_token,
                               'audio_feature': audio_feature,
                               'length': len(audio_feature),
                               }
            new_name_list.append(name)
        # except:
        # pass

        self.mean = mean
        self.std = std
        self.data_dict = data_dict
        self.name_list = new_name_list
        # print(len(se))

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[self.name_list[item]]
        # motion, m_token_list, a_token_list = data['motion'], data['m_token_list'], data['a_token_list']
        motion, m_tokens, audio_feature= data['motion'], data['m_token'], data['audio_feature']

        # a_tokens = [self.opt.aud_start_idx] + a_tokens #.tolist()
        # a_tokens = np.array(a_tokens, dtype=int)

        audio_feature = np.array(audio_feature)

        # m_tokens = [self.opt.mot_start_idx] + m_tokens #.tolist()
        # m_tokens = np.array(m_tokens, dtype=int)

        "Z Normalization"
        # motion = (motion - self.mean) / self.std
        #
        # m_length = len(motion)

        # if m_length < self.max_motion_frame:
        #     motion = np.concatenate([motion,
        #                              np.zeros((self.max_motion_frame - m_length, motion.shape[1]))
        #                              ], axis=0)
        # else:
        #     m_length = self.max_motion_frame
        #     motion = motion[:self.max_motion_frame]

        audio_len = audio_feature.shape[0]
        # m_tokens_len = m_tokens.shape[0]
        return audio_feature, 0, 0, self.name_list[item], audio_len, 0, 0, 0, 0
        # return audio_feature, 0, 0, self.name_list[item], audio_len, motion, m_tokens, m_tokens_len, 0
        # return 0, 0, a_tokens, self.name_list[item], m_length, motion, m_tokens, m_length, 0
        # return word_embeddings, pos_one_hots, word_ids, caption, sent_len, motion, m_tokens, m_length, all_captions
        # return word_embeddings, pos_one_hots, caption, sent_len, motion, m_tokens, m_length, all_captions



class WildAudioEvalDataset4MTP(data.Dataset):
    """
    适配只有audio情况的dataloader.
    1104: music text pair
    """
    def __init__(self, opt, mean, std, music_feat_path):
        self.opt = opt
        # self.max_length = 20
        self.pointer = 0
        # self.max_motion_frame = opt.max_motion_frame
        # min_motion_len = 40   # if self.opt.dataset_name == 't2m' else 24

        data_dict = {}
        id_list = [music_feat_path.split('/')[-1].split('.npy')[0]]
        # with cs.open(split_file, 'r') as f:
        #     for line in f.readlines():
        #         id_list.append(line.strip())
        # id_list = id_list[:40]

        new_name_list = []
        for name in tqdm(id_list):
            # audio_feature = np.load(pjoin(opt.audio_dir, name + '.npy')).astype('float32')
            audio_feature = np.load(music_feat_path).astype('float32')

            data_dict[name] = {'motion': 0,
                               'm_token': 0,
                               'audio_feature': audio_feature,
                               'length': len(audio_feature),
                               }
            new_name_list.append(name)

        self.mean = mean
        self.std = std
        self.data_dict = data_dict
        self.name_list = new_name_list
        # print(len(se))

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[self.name_list[item]]
        # motion, m_token_list, a_token_list = data['motion'], data['m_token_list'], data['a_token_list']
        motion, m_tokens, audio_feature= data['motion'], data['m_token'], data['audio_feature']

        audio_feature = np.array(audio_feature)
        audio_len = audio_feature.shape[0]
        return audio_feature, 0, 0, self.name_list[item], audio_len, 0, 0, 0, 0



class Motion2TextEvalSimpleDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_frame = opt.max_motion_frame
        min_motion_len = 40 if self.opt.dataset_name == 't2m' else 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:40]

        new_name_list = []
        for name in tqdm(id_list):
            # try:
            motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
            if (len(motion)) < min_motion_len or (len(motion) >= 600):
                continue

            m_token_list = []
            with cs.open(pjoin(opt.m_token_dir, name + '.txt'), 'r') as f:
                for line in f.readlines():
                    m_token_list.append(line.strip().split(' '))

            data_dict[name] = {'motion': motion,
                               'm_token_list': m_token_list,
                               'length': len(motion)
                               }
            new_name_list.append(name)
        # except:
        # pass

        self.mean = mean
        self.std = std
        self.data_dict = data_dict
        self.name_list = new_name_list
        # print(len(se))

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[self.name_list[item]]
        motion, m_token_list = data['motion'], data['m_token_list']

        m_tokens = random.choice(m_token_list)
        m_tokens = [int(token) for token in m_tokens]

        m_tokens = [self.opt.mot_start_idx] + \
                   m_tokens + \
                   [self.opt.mot_end_idx] + \
                   [self.opt.mot_pad_idx] * (self.opt.max_motion_token - len(m_tokens) - 2)

        # print(len(word_embeddings), sent_len, len(m_tokens))
        m_tokens = np.array(m_tokens, dtype=int)

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        m_length = len(motion)

        if m_length < self.max_motion_frame:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_frame - m_length, motion.shape[1]))
                                     ], axis=0)
        else:
            m_length = self.max_motion_frame
            motion = motion[:self.max_motion_frame]

        return motion, m_tokens, m_length


class RawTextDataset(data.Dataset):
    def __init__(self, opt, mean, std, text_file, w_vectorizer):
        self.mean = mean
        self.std = std
        self.opt = opt
        self.data_dict = []
        self.nlp = spacy.load('en_core_web_sm')

        with cs.open(text_file) as f:
            for line in f.readlines():
                word_list, pos_list = self.process_text(line.strip())
                tokens = ['%s/%s' % (word_list[i], pos_list[i]) for i in range(len(word_list))]
                self.data_dict.append({'caption': line.strip(), "tokens": tokens})

        self.w_vectorizer = w_vectorizer
        print("Total number of descriptions {}".format(len(self.data_dict)))

    def process_text(self, sentence):
        sentence = sentence.replace('-', '')
        doc = self.nlp(sentence)
        word_list = []
        pos_list = []
        for token in doc:
            word = token.text
            if not word.isalpha():
                continue
            if (token.pos_ == 'NOUN' or token.pos_ == 'VERB') and (word != 'left'):
                word_list.append(token.lemma_)
            else:
                word_list.append(word)
            pos_list.append(token.pos_)
        return word_list, pos_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[item]
        caption, tokens = data['caption'], data['tokens']

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
        word_ids = []
        for i, token in enumerate(tokens):
            word_emb, pos_oh, word_id = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
            if i >= sent_len:
                word_ids.append(self.opt.txt_pad_idx)
            else:
                word_ids.append(word_id)
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)
        word_ids = np.array(word_ids, dtype=int)

        return word_embeddings, pos_one_hots, word_ids, caption, sent_len
        # return word_embeddings, pos_one_hots, caption, sent_len


class RawTextDatasetV2(data.Dataset):
    '''
    0924: 添加了motion length indicator, 使得长度可控, 更容易和dance匹配.
    1022: 重新回到text feature的输入.
    '''
    def __init__(self, opt, mean, std, text_file, w_vectorizer):
        self.mean = mean
        self.std = std
        self.opt = opt
        self.data_dict = []
        self.mlen_list = []
        self.nlp = spacy.load('en_core_web_sm')

        with cs.open(text_file) as f:
            for line in f.readlines():
                word_list, pos_list = self.process_text(line.strip())
                tokens = ['%s/%s' % (word_list[i], pos_list[i]) for i in range(len(word_list))]
                self.data_dict.append({'caption': line.strip(), "tokens": tokens})

        mlen_file = text_file.replace('.txt', '_mleninfo.txt')
        with cs.open(mlen_file) as f:
            for line in f.readlines():
                line_key, line_num, start_key, start_num, end_key, end_num = line.replace(':', ' ').split()
                tmp_dict = {
                    line_key: int(line_num),
                    start_key: int(start_num),
                    end_key: int(end_num),
                }
                self.mlen_list.append(tmp_dict)

        self.w_vectorizer = w_vectorizer
        print("Total number of descriptions {}".format(len(self.data_dict)))

    def process_text(self, sentence):
        sentence = sentence.replace('-', '')
        doc = self.nlp(sentence)
        word_list = []
        pos_list = []
        for token in doc:
            word = token.text
            if not word.isalpha():
                continue
            if (token.pos_ == 'NOUN' or token.pos_ == 'VERB') and (word != 'left'):
                word_list.append(token.lemma_)
            else:
                word_list.append(word)
            pos_list.append(token.pos_)
        return word_list, pos_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[item]
        caption, tokens = data['caption'], data['tokens']

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
        word_ids = []
        for i, token in enumerate(tokens):
            word_emb, pos_oh, word_id = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
            expected_motion_len = self.mlen_list[item]['end'] - self.mlen_list[item]['start']
            if i >= sent_len:
                if i >= expected_motion_len:
                    word_ids.append(self.opt.txt_pad_idx)
                else:  # 这里把 >len(text) 和 <len(motion)部分, 用长度indicator代替.
                    word_ids.append(self.opt.txt_mlen_idx)
            else:
                word_ids.append(word_id)
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)
        word_ids = np.array(word_ids, dtype=int)

        # return word_embeddings, pos_one_hots, word_ids, caption, sent_len
        return word_embeddings, pos_one_hots, word_ids, caption, expected_motion_len  # 用这个来截断feature
        # return word_embeddings, pos_one_hots, caption, sent_len


class RawTextDatasetV3(data.Dataset):
    '''
    0924: 添加了motion length indicator, 使得长度可控, 更容易和dance匹配.
    1018: try random some text and see
    '''
    def __init__(self, opt, mean, std, lines, w_vectorizer):
        self.mean = mean
        self.std = std
        self.opt = opt
        self.data_dict = []
        self.mlen_list = []
        self.nlp = spacy.load('en_core_web_sm')

        for line in lines:
            word_list, pos_list = self.process_text(line.strip())
            tokens = ['%s/%s' % (word_list[i], pos_list[i]) for i in range(len(word_list))]
            self.data_dict.append({'caption': line.strip(), "tokens": tokens})

        mlen_file = 'input_mleninfo.txt'
        with cs.open(mlen_file) as f:
            for line in f.readlines():
                line_key, line_num, start_key, start_num, end_key, end_num = line.replace(':', ' ').split()
                tmp_dict = {
                    line_key: int(line_num),
                    start_key: int(start_num),
                    end_key: int(end_num),
                }
                self.mlen_list.append(tmp_dict)

        self.w_vectorizer = w_vectorizer
        print("Total number of descriptions {}".format(len(self.data_dict)))

    def process_text(self, sentence):
        sentence = sentence.replace('-', '')
        doc = self.nlp(sentence)
        word_list = []
        pos_list = []
        for token in doc:
            word = token.text
            if not word.isalpha():
                continue
            if (token.pos_ == 'NOUN' or token.pos_ == 'VERB') and (word != 'left'):
                word_list.append(token.lemma_)
            else:
                word_list.append(word)
            pos_list.append(token.pos_)
        return word_list, pos_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[item]
        caption, tokens = data['caption'], data['tokens']

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
        word_ids = []
        for i, token in enumerate(tokens):
            word_emb, pos_oh, word_id = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
            expected_motion_len = self.mlen_list[item]['end'] - self.mlen_list[item]['start']
            if i >= sent_len:
                if i >= expected_motion_len:
                    word_ids.append(self.opt.txt_pad_idx)
                else:  # 这里把 >len(text) 和 <len(motion)部分, 用长度indicator代替.
                    word_ids.append(self.opt.txt_mlen_idx)
            else:
                word_ids.append(word_id)
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)
        word_ids = np.array(word_ids, dtype=int)

        # return word_embeddings, pos_one_hots, word_ids, caption, sent_len
        return word_embeddings, pos_one_hots, word_ids, caption, expected_motion_len  # 用这个来截断feature
        # return word_embeddings, pos_one_hots, caption, sent_len


class RawTextDataset4MTP(data.Dataset):
    '''
    0924: 添加了motion length indicator, 使得长度可控, 更容易和dance匹配.
    1018: try random some text and see
    1104: mtp: music text pair

    '''
    def __init__(self, opt, mean, std, mtp_dict, w_vectorizer):
        self.mean = mean
        self.std = std
        self.opt = opt
        self.data_dict = []
        self.mlen_list = []
        self.nlp = spacy.load('en_core_web_sm')
        lines = mtp_dict['text_list']

        for line in lines:
            word_list, pos_list = self.process_text(line.strip())
            tokens = ['%s/%s' % (word_list[i], pos_list[i]) for i in range(len(word_list))]
            self.data_dict.append({'caption': line.strip(), "tokens": tokens})

        for i in range(len(lines)):
            tmp_dict = {
                'line': i,
                'start': mtp_dict['text_start_list'][i],
                'end': mtp_dict['text_start_list'][i] + mtp_dict['text_duration_list'][i],
            }
            self.mlen_list.append(tmp_dict)

        self.w_vectorizer = w_vectorizer
        print("Total number of descriptions {}".format(len(self.data_dict)))

    def process_text(self, sentence):
        sentence = sentence.replace('-', '')
        doc = self.nlp(sentence)
        word_list = []
        pos_list = []
        for token in doc:
            word = token.text
            if not word.isalpha():
                continue
            if (token.pos_ == 'NOUN' or token.pos_ == 'VERB') and (word != 'left'):
                word_list.append(token.lemma_)
            else:
                word_list.append(word)
            pos_list.append(token.pos_)
        return word_list, pos_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[item]
        caption, tokens = data['caption'], data['tokens']

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
        word_ids = []
        for i, token in enumerate(tokens):
            word_emb, pos_oh, word_id = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
            expected_motion_len = self.mlen_list[item]['end'] - self.mlen_list[item]['start']
            if i >= sent_len:
                if i >= expected_motion_len:
                    word_ids.append(self.opt.txt_pad_idx)
                else:  # 这里把 >len(text) 和 <len(motion)部分, 用长度indicator代替.
                    word_ids.append(self.opt.txt_mlen_idx)
            else:
                word_ids.append(word_id)
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)
        word_ids = np.array(word_ids, dtype=int)

        # return word_embeddings, pos_one_hots, word_ids, caption, sent_len
        return word_embeddings, pos_one_hots, word_ids, caption, expected_motion_len  # 用这个来截断feature
        # return word_embeddings, pos_one_hots, caption, sent_len
