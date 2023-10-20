import codecs as cs
import glob

import spacy
from torch.utils import data
from torch.utils.data import DataLoader

from networks.evaluator_wrapper import EvaluatorModelWrapper
from options.evaluate_options import TestT2MOptions
from scripts.motion_process import *
from tools.pose2feature_converter import Pose2FeatureConverter
from utils.get_opt import get_opt
from utils.utils import *
from utils.word_vectorizer import WordVectorizerV2

"""
input text and motion, return samilarity
"""
class TextMotionDistance(object):
    def __init__(self):
        parser = TestT2MOptions()
        opt = parser.parse()

        opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
        torch.autograd.set_detect_anomaly(True)
        if opt.gpu_id != -1:
            torch.cuda.set_device(opt.gpu_id)

        opt.joints_num = 22
        # opt.max_motion_token = 55
        # opt.max_motion_frame = 196
        # dim_pose = 263
        # radius = 4
        # fps = 20
        # kinematic_chain = paramUtil.t2m_kinematic_chain

        w_vectorizer = WordVectorizerV2('./glove', 'our_vab')
        n_txt_vocab = len(w_vectorizer) + 1
        _, _, opt.txt_start_idx = w_vectorizer['sos/OTHER']
        _, _, opt.txt_end_idx = w_vectorizer['eos/OTHER']
        opt.txt_pad_idx = len(w_vectorizer)

        self.opt = opt
        # self.nlp = spacy.load('en_core_web_sm')
        import en_core_web_sm
        self.nlp = en_core_web_sm.load()
        self.w_vectorizer = w_vectorizer

        similarity_eval_opt_path = './checkpoints/t2m/Comp_v6_KLD005/opt.txt'
        # device_id = 0
        device_id = opt.gpu_id
        device = torch.device('cuda:%d' % device_id if torch.cuda.is_available() else 'cpu')
        wrapper_opt = get_opt(similarity_eval_opt_path, device)
        self.eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
        self.MotionConvert = Pose2FeatureConverter()

    def measure_distance_byfeat60fps24j(self, sentence, feat60fps24j):
        feature_20fps22j = self.MotionConvert.normed_f24j_to_normed_f22j(feat60fps24j)
        return  self.measure_distance_byfeat20fps22j(sentence, feature_20fps22j)

    def measure_distance_byjoint_60fps24j(self, sentence, joint_60fps24j):
        feature_20fps22j = self.MotionConvert.joint60fps24j_to_f22j(joint_60fps24j)
        return self.measure_distance_byfeat20fps22j(sentence, feature_20fps22j)

    def measure_distance_byfeat20fps22j(self, sentence, feature_20fps22j):
        word_embeddings, pos_one_hots, caption, sent_len = self.process_text2embedding(sentence)

        motions = torch.from_numpy(feature_20fps22j).unsqueeze(0)
        m_lens = torch.tensor(np.int(motions.shape[1])).unsqueeze(0)

        word_embeddings = torch.from_numpy(word_embeddings).unsqueeze(0)
        pos_one_hots = torch.from_numpy(pos_one_hots).unsqueeze(0)
        sent_lens = torch.tensor(sent_len).unsqueeze(0)

        text_embeddings, motion_embeddings = self.eval_wrapper.get_co_embeddings(
            word_embs=word_embeddings,
            pos_ohot=pos_one_hots,
            cap_lens=sent_lens,
            motions=motions,
            m_lens=m_lens
        )
        dist_mat = euclidean_distance_matrix(text_embeddings.cpu().numpy(),
                                             motion_embeddings.cpu().numpy())
        matching_score = dist_mat.trace()
        # print(matching_score)
        return matching_score

    def process_text2embedding(self, sentence):
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
        # return word_list, pos_list

        tokens = ['%s/%s' % (word_list[i], pos_list[i]) for i in range(len(word_list))]
        data = {'caption':sentence, "tokens":tokens}

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
        for token in tokens:
            word_emb, pos_oh, _ = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len



def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists



if __name__ == '__main__':

    TMDist = TextMotionDistance()

    npy_path = '../TM2T/eval_results/aistppml3d/ATM_exp1_0926n/atmv2-lf3-r3-1007/animations/A person is jumping up and down./t2m_gen_motion_02_L240_00.npy'
    joint_60fps24j = np.load(npy_path)
    sentence = 'A person is jumping up and down.'

    matching_score = TMDist.measure_distance_byjoint_60fps24j(sentence, joint_60fps24j)
    print(matching_score)
    print('-------------')


    joint_60fps24j_folder = '../TM2T/eval_results/aistppml3d/ATM_exp1_0926n/atmv2-lf3-r3-1007/animations/A person is spinning.'
    joint_60fps24j_path_list = glob.glob(joint_60fps24j_folder + '/*.npy')
    joint_60fps24j_path_list.sort()

    for jpath in joint_60fps24j_path_list:
        joint_60fps24j = np.load(jpath)
        sentence = 'A person is spinning.'
        matching_score = TMDist.measure_distance_byjoint_60fps24j(sentence, joint_60fps24j)
        print(jpath)
        print(matching_score)
        print('-------------')

