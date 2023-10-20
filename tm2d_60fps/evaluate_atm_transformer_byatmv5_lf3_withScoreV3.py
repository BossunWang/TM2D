import os

from os.path import join as pjoin

import numpy as np
import copy
import utils.paramUtil as paramUtil
from options.evaluate_options import TestT2MOptions
from utils.plot_script import *

from networks.transformer_x_lf3 import TransformerV1, TransformerV2
from networks.quantizer import *
from networks.modules import *
from data.dataset_eval import Motion2AudioEvalDataset4ATM, RawTextDatasetV2, RawTextDatasetV3
from scripts.motion_process import *
from torch.utils.data import DataLoader
from utils.word_vectorizer import WordVectorizerV2
from utils.utils import *
from tools.measure_t2m_similarity import TextMotionDistance

"""
This one is used to evaluate the tm2d
define the input.txt and input_mleninfo.txt
output is the text2motion, music2dance, combined result.
"""


def plot_t2m_by_joint(joint, captions, save_dir, add_audio=False, audio_name=None):
    caption = captions[i]
    save_path = '%s_%02d.mp4' % (save_dir, i)
    np.save('%s_%02d.npy' % (save_dir, i), joint)
    plot_3d_motion(save_path, kinematic_chain, joint[::2], title=caption, fps=int(fps/2), radius=radius)
    if add_audio:
        audio_name = caption if audio_name is None else audio_name
        combine_audio_to_video(video_path=save_path, name=audio_name)

def plot_t2m(data, captions, save_dir, add_audio=False, audio_name=None):
    data = data * std + mean
    for i in range(len(data)):
        assert len(data) == 1, 'not consider batch-wise'
        joint_data = data[i]
        caption = captions[i]
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
        save_path = '%s_%02d.mp4' % (save_dir, i)
        np.save('%s_%02d.npy' % (save_dir, i), joint)
        plot_3d_motion(save_path, kinematic_chain, joint[::2], title=caption, fps=int(fps / 2), radius=radius)
        if add_audio:
            audio_name = caption if audio_name is None else audio_name
            combine_audio_to_video(video_path=save_path, name=audio_name)

def combine_score_name(score_list):
    score_name = 'Score'
    for score in score_list:
        score_name = score_name + '_%03d' % (score * 10)
    return score_name

def combine_title_list(aud_name, title_list):
    title = 'AudioName: {} '.format(aud_name)
    for title_line in title_list:
        title = title + '\n\n{}'.format(title_line)
    return title


def combine_audio_to_video(video_path=None, name=None):
    if 'cAll' in name:
        music_name = name[-9:-5] + '.wav'
        audio_dir = '../Bailando/aist_plusplus_final/all_musics'
    else:
        # for wild music case
        music_name = name + '.MP3'
        audio_dir = '../Bailando/extra'
    print('combining audio!')
    audio_dir_ = os.path.join(audio_dir, music_name)
    print(video_path)
    output_path = video_path.replace('.mp4', '_audio.mp4')
    cmd_audio = f"ffmpeg -i {video_path} -i {audio_dir_} -map 0:v -map 1:a -c:v copy -shortest -y {output_path} -loglevel quiet"
    os.system(cmd_audio)

def measure_text2mostion_similarity_by_token(t2m_sentence, pred_tokens):
    vq_latent = quantizer.get_codebook_entry(pred_tokens)
    gen_motion = vq_decoder(vq_latent)

    t2m_feat60fps24j = gen_motion.cpu().numpy()[0]
    score = wrap_measure_distance_byfeat60fps24j(t2m_sentence, t2m_feat60fps24j)
    return score

def wrap_measure_distance_byfeat60fps24j(t2m_sentence, t2m_feat60fps24j):
    KeyWrods = 'dancing and '
    score = TMDist.measure_distance_byfeat60fps24j(t2m_sentence, t2m_feat60fps24j)
    if KeyWrods in t2m_sentence:
        t2m_sentence_action = t2m_sentence.replace(KeyWrods, '')
        score_action = TMDist.measure_distance_byfeat60fps24j(t2m_sentence_action, t2m_feat60fps24j)
        print(t2m_sentence, score)
        print(t2m_sentence_action, score_action)
        score = max(score, score_action)
    return score


def build_models(opt):
    vq_decoder = VQDecoderV3(opt.dim_vq_latent, dec_channels, opt.n_resblk, opt.n_down)
    quantizer = None
    if opt.q_mode == 'ema':
        quantizer = EMAVectorQuantizer(opt.codebook_size, opt.dim_vq_latent, opt.lambda_beta)
    elif opt.q_mode == 'cmt':
        quantizer = Quantizer(opt.codebook_size, opt.dim_vq_latent, opt.lambda_beta)

    checkpoint = torch.load(
        pjoin(opt.checkpoints_dir, opt.dataset_name, opt.tokenizer_name, 'model', '%s.tar' % (opt.which_vqvae)),
        map_location=opt.device)
    vq_decoder.load_state_dict(checkpoint['vq_decoder'])
    quantizer.load_state_dict(checkpoint['quantizer'])

    a2d_transformer = TransformerV1(n_mot_vocab, opt.mot_pad_idx, d_src_word_vec=438, d_trg_word_vec=512,
                                    d_model=opt.d_model, d_inner=opt.d_inner_hid, n_enc_layers=opt.n_enc_layers,
                                    n_dec_layers=opt.n_dec_layers, n_head=opt.n_head, d_k=opt.d_k, d_v=opt.d_v,
                                    dropout=0.1,
                                    n_src_position=100, n_trg_position=100,
                                    trg_emb_prj_weight_sharing=opt.proj_share_weight)

    t2m_transformer = TransformerV2(n_txt_vocab, opt.txt_pad_idx, n_mot_vocab, opt.mot_pad_idx, d_src_word_vec=512,
                                    d_trg_word_vec=512,
                                    d_model=opt.d_model, d_inner=opt.d_inner_hid, n_enc_layers=opt.n_enc_layers,
                                    n_dec_layers=opt.n_dec_layers, n_head=opt.n_head, d_k=opt.d_k, d_v=opt.d_v,
                                    dropout=0.1,
                                    n_src_position=100, n_trg_position=100,
                                    trg_emb_prj_weight_sharing=opt.proj_share_weight
                                    )

    checkpoint = torch.load(
        pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'model', '%s.tar' % (opt.which_epoch)),
        map_location=opt.device)
    # t2m_transformer.load_state_dict(checkpoint['t2m_transformer'])
    a2d_transformer.load_state_dict(checkpoint['a2d_transformer'])
    print('Loading a2d_transformer model: Epoch %03d Total_Iter %03d' % (checkpoint['ep'], checkpoint['total_it']))

    t2m_transformer.load_state_dict(checkpoint['t2m_transformer'])
    print('Loading t2m_transformer model: Epoch %03d Total_Iter %03d' % (checkpoint['ep'], checkpoint['total_it']))
    return vq_decoder, quantizer, a2d_transformer, t2m_transformer


if __name__ == '__main__':
    parser = TestT2MOptions()
    opt = parser.parse()
    opt.max_text_len = 84  # 84 = 55x3/8x4# 55

    # EVAL_MODE = 'vis'  # 'metric'  # vis_lf_only
    EVAL_MODE = opt.eval_mode

    # add similarity measure
    TMDist = TextMotionDistance()

    # add search strategy when add text2motion in dance.
    total_num_try = 1
    search_times = 30
    search_threshold = 2.5
    plot_score_threshold = 3.0

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)
    if opt.gpu_id != -1:
        torch.cuda.set_device(opt.gpu_id)

    opt.result_dir = pjoin(opt.result_path, opt.dataset_name, opt.name, opt.ext)
    opt.joint_dir = pjoin(opt.result_dir, 'joints')
    opt.animation_dir = pjoin(opt.result_dir, 'animations')

    # os.makedirs(opt.joint_dir, exist_ok=True)
    os.makedirs(opt.animation_dir, exist_ok=True)

    if opt.dataset_name == 't2m':
        assert False, 'for audio'
    elif opt.dataset_name == 'aistpp':
        opt.data_root = './dataset/aistpp/'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.audio_dir = pjoin(opt.data_root, 'audio_vecs_7.5fps')
        opt.joints_num = 24
        # opt.max_motion_token = 55
        # opt.max_motion_frame = 196
        dim_pose = 287
        radius = 4
        fps = 60
        kinematic_chain = paramUtil.smpl24_kinematic_chain
    elif opt.dataset_name == 'aistppml3d':
        opt.data_root = './dataset/aistppml3d/'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.audio_dir = pjoin(opt.data_root, 'audio_vecs_7.5fps')
        opt.joints_num = 24
        # opt.max_motion_token = 55
        # opt.max_motion_frame = 196
        dim_pose = 287
        radius = 4
        fps = 60
        kinematic_chain = paramUtil.smpl24_kinematic_chain
    elif opt.dataset_name == 'kit':
        assert False, 'for audio'
    else:
        raise KeyError('Dataset Does Not Exist')

    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.tokenizer_name, 'meta', 'mean.npy'))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.tokenizer_name, 'meta', 'std.npy'))

    n_mot_vocab = opt.codebook_size + 3
    opt.mot_start_idx = opt.codebook_size
    opt.mot_end_idx = opt.codebook_size + 1
    opt.mot_pad_idx = opt.codebook_size + 2

    n_aud_vocab = opt.codebook_size + 3
    opt.aud_start_idx = opt.codebook_size
    opt.aud_end_idx = opt.codebook_size + 1
    opt.aud_pad_idx = opt.codebook_size + 2

    w_vectorizer = WordVectorizerV2('./glove', 'our_vab')
    n_txt_vocab = len(w_vectorizer) + 2
    _, _, opt.txt_start_idx = w_vectorizer['sos/OTHER']
    _, _, opt.txt_end_idx = w_vectorizer['eos/OTHER']
    opt.txt_mlen_idx = len(w_vectorizer)
    opt.txt_pad_idx = len(w_vectorizer) + 1

    # enc_channels = [1024, opt.dim_vq_latent]
    # dec_channels = [opt.dim_vq_latent, 1024, dim_pose]
    enc_channels = [1024, ] + [opt.dim_vq_latent, ] * (opt.n_down - 1)
    dec_channels = [opt.dim_vq_latent, ] * (opt.n_down - 1) + [1024, dim_pose]

    vq_decoder, quantizer, a2d_transformer, t2m_transformer = build_models(opt)

    # split_file = pjoin(opt.data_root, opt.split_file)
    if opt.aist_split_file is None:
        split_file = pjoin(opt.data_root, 'aistpp_test.txt')
    else:
        split_file = opt.aist_split_file

    # dataset = Motion2AudioEvalDataset(opt, mean, std, split_file)
    aud_dataset = Motion2AudioEvalDataset4ATM(opt, mean, std, split_file)
    aud_data_loader = DataLoader(aud_dataset, batch_size=1, num_workers=1, shuffle=False, pin_memory=True)

    ################################################
    # text part
    t2m_dataset = RawTextDatasetV2(opt, mean, std, opt.text_file, w_vectorizer)
    t2m_data_loader = DataLoader(t2m_dataset, batch_size=opt.batch_size, num_workers=1, pin_memory=True)

    # combien opinion
    text_start_list = []  # [75, 150]
    text_duration_list = []  # [30, 30]
    text_content_list = []
    for tmp_dict in t2m_dataset.mlen_list:
        text_start_list.append(tmp_dict['start'])
        text_duration_list.append(int(tmp_dict['end']-tmp_dict['start']))

    print('text_start_list: ', text_start_list)
    print('text_duration_list: ', text_duration_list)

    vq_decoder.to(opt.device)
    quantizer.to(opt.device)
    a2d_transformer.to(opt.device)
    t2m_transformer.to(opt.device)

    vq_decoder.eval()
    quantizer.eval()
    a2d_transformer.eval()
    t2m_transformer.eval()

    opt.repeat_times = opt.repeat_times if opt.sample else 1

    '''Generating t2m encoding'''
    print('Generating t2m encoding')
    t2m_result_dict = {}
    t2m_result_list = []
    item_dict = {}
    with torch.no_grad():
        for i, batch_data in enumerate(t2m_data_loader):
            print('%02d_%03d' % (i, opt.num_results))
            # word_emb, pos_ohot, captions, cap_lens = batch_data
            word_emb, pos_ohot, word_ids, captions, cap_lens = batch_data

            # word_emb = word_emb.detach().to(opt.device).float()
            word_ids = word_ids.detach().to(opt.device).long()

            print(captions[0])
            text_content_list.append(captions[0])
            # name = 'C%03d' % (i)
            item_dict = {
                'caption': captions,
                'length': cap_lens[0],
                # 'gt_motion': motions[:, :m_lens[0]].cpu().numpy()
            }

            enc_output, src_mask = t2m_transformer.encoding(word_ids)
            # enc_output, src_mask = t2m_transformer.encoding(word_emb, cap_lens)
            item_dict['enc_output'] = enc_output
            item_dict['src_mask'] = src_mask

            t2m_result_dict[captions[0]] = item_dict
            t2m_result_list.append(item_dict)

    '''Generating t2m decoding'''
    print('Generating t2m decoding')
    # result_dict = {}
    item_dict = {}
    with torch.no_grad():
        for i, batch_data in enumerate(t2m_data_loader):
            print('%02d_%03d' % (i, opt.num_results))
            # word_emb, pos_ohot, captions, sent_lens, motions, m_tokens, m_lens, _ = batch_data
            word_emb, pos_ohot, word_ids, captions, cap_lens = batch_data

            # word_emb = word_emb.detach().to(opt.device).float()
            word_ids = word_ids.detach().to(opt.device).long()
            # gt_tokens = motions[:, :m_lens[0]]

            print(captions[0])
            # load from previous saved
            name = captions[0]

            for t in range(opt.repeat_times):
                searched_best_score = np.inf
                searched_best_sub_dict = None

                for search_num in range(search_times):
                    pred_tokens = t2m_transformer.sample_with_enc_output(
                        word_ids,
                        enc_output=t2m_result_dict[name]['enc_output'][:, :],
                        trg_sos=opt.mot_start_idx, trg_eos=opt.mot_end_idx,
                        end_idx=cap_lens[0], sample=opt.sample, top_k=opt.top_k)
                    pred_tokens = pred_tokens[:, 1:]
                    print('Sampled Tokens %02d' % t)
                    # print(pred_tokens[0])
                    if len(pred_tokens[0]) == 0:
                        continue
                    vq_latent = quantizer.get_codebook_entry(pred_tokens)
                    gen_motion = vq_decoder(vq_latent)

                    sub_dict = {}
                    sub_dict['pred_tokens'] = pred_tokens.cpu().numpy()
                    sub_dict['motion'] = gen_motion.cpu().numpy()
                    sub_dict['length'] = len(gen_motion[0])
                    # add score here
                    sub_dict['score'] = wrap_measure_distance_byfeat60fps24j(captions[0], gen_motion[0].cpu().numpy())

                    # now apply search strategy:
                    if sub_dict['score'] < searched_best_score:
                        searched_best_score = sub_dict['score'] * 1.
                        searched_best_sub_dict = sub_dict
                    if searched_best_score < search_threshold:
                        break

                # t2m_result_dict[name]['result_%02d' % t] = sub_dict
                print('generate action with score {:.5} with search_num {} for caption {}'.format(
                    searched_best_score, search_num, captions[0]))
                t2m_result_dict[name]['result_%02d' % t] = searched_best_sub_dict


    result_dict = copy.deepcopy(t2m_result_dict)
    if not EVAL_MODE == 'vis_lf_only':
        print('Animating t2m Results')
        '''Animating t2m Results'''
        for i, (key, item) in enumerate(result_dict.items()):
            print('%02d_%03d' % (i, opt.num_results))
            captions = item['caption']
            # gt_motions = item['gt_motion']
            # joint_save_path = pjoin(opt.joint_dir, key)
            animation_save_path = pjoin(opt.animation_dir, key)

            # os.makedirs(joint_save_path, exist_ok=True)
            os.makedirs(animation_save_path, exist_ok=True)

            # np.save(pjoin(joint_save_path, 'gt_motions.npy'), gt_motions)
            # plot_t2m(gt_motions, aud_name, pjoin(animation_save_path, 'gt_motion'))
            for t in range(opt.repeat_times):
                sub_dict = item['result_%02d' % t]
                motion = sub_dict['motion']
                score = sub_dict['score'] * 10
                # np.save(pjoin(joint_save_path, 'gen_motion_%02d_L%03d.npy' % (t, motion.shape[1])), motion)
                plot_t2m(motion, captions, pjoin(animation_save_path, 't2m_gen_motion_%02d_L%03d_Score%03d' % (t, motion.shape[1], score)))

    ################################################################################
    ################################################################################
    '''Generating a2d encoding'''
    print('Generating a2d encoding')
    a2d_result_dict = {}
    a2d_result_list = []
    item_dict = {}
    with torch.no_grad():
        for i, batch_data in enumerate(aud_data_loader):
            print('%02d_%03d' % (i, opt.num_results))
            # word_emb, pos_ohot, captions, cap_lens = batch_data
            audio_feature, _, _, aud_name, a_tokens_len, motions, m_tokens, m_lens, _ = batch_data

            audio_feature = audio_feature.detach().to(opt.device).float()
            # a_tokens = a_tokens.detach().to(opt.device).long()

            print(aud_name[0])
            # name = 'C%03d' % (i)
            name = aud_name[0]
            item_dict = {
                'aud_name': aud_name[0],
                'caption': aud_name,
                'sent_lens': a_tokens_len[0],
                # 'gt_motion': motions[:, :m_lens[0]].cpu().numpy()
            }

            chunk_len = 50
            chunk_overlap = 1
            chunk_stride = chunk_len - chunk_overlap

            enc_output_list = []
            src_mask_list = []
            for start_idx in range(0, audio_feature.shape[1], chunk_stride):
                end_idx = start_idx + chunk_len
                enc_output, src_mask = a2d_transformer.encoding(audio_feature[:, start_idx:end_idx],
                                                                src_non_pad_lens=torch.tensor([audio_feature[:, start_idx:end_idx].shape[1]]))
                if start_idx == 0:
                    enc_output_list.append(enc_output)
                    src_mask_list.append(src_mask)
                else:
                    enc_output_list.append(enc_output[:, chunk_overlap:])
                    src_mask_list.append(src_mask[:, :, chunk_overlap:])

            # enc_output, src_mask = a2d_transformer.encoding(a_tokens)
            item_dict['enc_output'] = torch.cat(enc_output_list, dim=1)
            item_dict['src_mask'] = torch.cat(src_mask_list, dim=2)
            a2d_result_dict[name] = item_dict
            a2d_result_list.append(item_dict)

            if i + 1 >= opt.num_results:
                break

    '''Generating a2d decoding'''
    print('Generating a2d decoding')
    # result_dict = {}
    item_dict = {}
    with torch.no_grad():
        for i, batch_data in enumerate(aud_data_loader):
            print('%02d_%03d' % (i, opt.num_results))
            # word_emb, pos_ohot, captions, sent_lens, motions, m_tokens, m_lens, _ = batch_data
            audio_feature, _, _, aud_name, _, motions, m_tokens, m_lens, _ = batch_data

            audio_feature = audio_feature.detach().to(opt.device).float()
            m_tokens = m_tokens.detach().to(opt.device).long()
            # a_tokens = a_tokens.detach().to(opt.device).long()
            # gt_tokens = motions[:, :m_lens[0]]

            print(aud_name[0])

            # name = 'L%03dC%03d' % (m_lens[0], i)
            name = aud_name[0]
            
            for t in range(opt.repeat_times):
                # repeatly generate sequence, trained - 30 to 30
                pred_tokens_list = []
                start_idx = 0
                end_idx = 0
                # for start_idx in range(0, audio_feature.shape[1], chunk_stride):
                while end_idx < audio_feature.shape[1]:
                    end_idx = start_idx + chunk_len
                    if end_idx > audio_feature.shape[1]:
                        end_idx = audio_feature.shape[1]

                    if start_idx == 0:
                        # # random a starting token for dance
                        # trg_seq = torch.LongTensor(audio_feature.size(0), 1).fill_(np.random.randint(1024)).to(
                        #     audio_feature).long()
                        # random a starting token for dance, select from token data
                        trg_seq = torch.LongTensor(audio_feature.size(0), 1).fill_(random.choice(m_tokens.reshape(-1))).to(
                            audio_feature).long()

                        pred_tokens = a2d_transformer.sample_with_trg_seq_enc_output(
                            audio_feature[:, start_idx:end_idx],
                            src_non_pad_lens=torch.tensor([audio_feature[:, start_idx:end_idx].shape[1]]),
                            enc_output=a2d_result_dict[name]['enc_output'][:, start_idx:end_idx],
                            trg_sos=opt.mot_start_idx, trg_eos=opt.mot_end_idx,
                            trg_seq=trg_seq,
                            start_idx=start_idx+1, end_idx=end_idx, sample=opt.sample, top_k=opt.top_k)
                        pred_tokens_list.append(pred_tokens[:, :])

                    else:
                        pred_tokens = a2d_transformer.sample_with_trg_seq_enc_output(
                            audio_feature[:, start_idx:end_idx],
                            src_non_pad_lens=torch.tensor([audio_feature[:, start_idx:end_idx].shape[1]]),
                            enc_output=a2d_result_dict[name]['enc_output'][:, start_idx:end_idx],
                            trg_sos=opt.mot_start_idx, trg_eos=opt.mot_end_idx,
                            trg_seq=pred_tokens[:, -chunk_overlap:],
                            start_idx=start_idx+chunk_overlap, end_idx=end_idx, sample=opt.sample, top_k=opt.top_k)
                        pred_tokens_list.append(pred_tokens[:, chunk_overlap:])
                    # update start_idx
                    start_idx = end_idx - chunk_overlap

                # pred_tokens = pred_tokens[:, 1:]
                pred_tokens = torch.cat(pred_tokens_list, dim=-1)
                print('Sampled Tokens %02d' % t)
                # print(pred_tokens[0])
                if len(pred_tokens[0]) == 0:
                    continue
                # double check if the generated motion = feature length
                assert pred_tokens.shape[1] == audio_feature.shape[1], 'duration mismatch!'

                vq_latent = quantizer.get_codebook_entry(pred_tokens)
                gen_motion = vq_decoder(vq_latent)

                sub_dict = {}
                sub_dict['pred_tokens'] = pred_tokens.cpu().numpy()
                sub_dict['motion'] = gen_motion.cpu().numpy()
                sub_dict['length'] = len(gen_motion[0])
                a2d_result_dict[name]['result_%02d' % t] = sub_dict

            # result_dict[name] = item_dict
            if i + 1 >= opt.num_results:
                break

    result_dict = copy.deepcopy(a2d_result_dict)

    if not EVAL_MODE == 'vis_lf_only':
        print('Animating a2d Results')
        '''Animating a2d Results'''
        for i, (key, item) in enumerate(result_dict.items()):
            print('%02d_%03d' % (i, opt.num_results))
            aud_name = item['caption']
            # gt_motions = item['gt_motion']
            # joint_save_path = pjoin(opt.joint_dir, key)
            animation_save_path = pjoin(opt.animation_dir, key)

            # os.makedirs(joint_save_path, exist_ok=True)
            os.makedirs(animation_save_path, exist_ok=True)

            # np.save(pjoin(joint_save_path, 'gt_motions.npy'), gt_motions)
            # plot_t2m(gt_motions, aud_name, pjoin(animation_save_path, 'gt_motion'))
            for t in range(opt.repeat_times):
                sub_dict = item['result_%02d' % t]
                motion = sub_dict['motion']
                # np.save(pjoin(joint_save_path, 'gen_motion_%02d_L%03d.npy' % (t, motion.shape[1])), motion)
                plot_t2m(motion, aud_name, pjoin(animation_save_path, 'a2d_gen_motion_%02d_L%03d' % (t, motion.shape[1])), add_audio=True)


    ########################################################################################################################
    from tools.weight_mask import *

    '''Generating atm decoding'''
    print('Generating atm decoding')
    for mix_mode in ['cosinepeak-08']:

        atm_result_dict = copy.deepcopy(a2d_result_dict)
        result_dict = {}
        item_dict = {}
        with torch.no_grad():
            for i, batch_data in enumerate(aud_data_loader):
                print('%02d_%03d' % (i, opt.num_results))
                # word_emb, pos_ohot, captions, sent_lens, motions, m_tokens, m_lens, _ = batch_data
                audio_feature, _, _, aud_name, _, motions, m_tokens, m_lens, _ = batch_data

                # word_emb = word_emb.detach().to(opt.device).float()
                audio_feature = audio_feature.detach().to(opt.device).float()
                m_tokens = m_tokens.detach().to(opt.device).long()
                # a_tokens = a_tokens.detach().to(opt.device).long()
                # gt_tokens = motions[:, :m_lens[0]]

                print(aud_name[0])

                # name = 'L%03dC%03d' % (m_lens[0], i)
                name = aud_name[0]

                for item in text_duration_list:
                    assert item < chunk_len - chunk_overlap
                atm_result_dict[name]['enc_output_a'] = torch.zeros_like(atm_result_dict[name]['enc_output'])
                atm_result_dict[name]['src_mask_a'] = torch.zeros_like(atm_result_dict[name]['src_mask'])
                atm_result_dict[name]['weight_a'] = torch.zeros(audio_feature.shape[1])
                atm_result_dict[name]['enc_output_b'] = atm_result_dict[name]['enc_output'].clone().detach()
                atm_result_dict[name]['src_mask_b'] = atm_result_dict[name]['src_mask'].clone().detach()
                for i, txt_key in enumerate(t2m_result_dict):
                    s = text_start_list[i]
                    d = text_duration_list[i]
                    print('[ATM] now add text {} at time {}s'.format(t2m_result_dict[txt_key]['caption'][0],
                                                                     (s * 8 / 60)))
                    atm_result_dict[name]['enc_output_a'][:, s:s + d] = t2m_result_dict[txt_key]['enc_output'][:, :d]
                    atm_result_dict[name]['src_mask_a'][:, :, s:s + d] = t2m_result_dict[txt_key]['src_mask'][:, :, :d]
                    atm_result_dict[name]['weight_a'][s:s + d] = get_mix_weight(d, mode=mix_mode)

                for t in range(opt.repeat_times):
                    # t = 0
                    num_try = 0
                    while num_try < total_num_try:
                        # repeatly generate sequence, trained - 30 to 30
                        pred_tokens_list = []
                        atm_result_dict[name]['result_%02d' % t]['score_list'] = []
                        atm_result_dict[name]['result_%02d' % t]['title_list'] = []
                        # for start_idx in range(0, audio_feature.shape[1], chunk_stride):
                        next_text_idx = 0
                        start_idx = 0
                        end_idx = 0
                        while end_idx < audio_feature.shape[1]:
                            flag_involve_text = False
                            if start_idx + chunk_len < text_start_list[next_text_idx]:
                                end_idx = start_idx + chunk_len
                            else:  # if start_idx + chunk_len >= text_start_list[next_text_idx]
                                # this is to make sure the new start will start from text description.
                                end_idx = text_start_list[next_text_idx]
                                next_text_idx = next_text_idx + 1
                                text_start_list.append(np.inf)

                            if end_idx > audio_feature.shape[1]:  # make sure the last end_idx is the last one
                                end_idx = audio_feature.shape[1]

                            # check if text involved in current round
                            if next_text_idx >= 1:
                                current_text_idx = next_text_idx - 1
                                if start_idx < text_start_list[current_text_idx] < end_idx:
                                    flag_involve_text = True
                                    end_idx = text_start_list[current_text_idx] + text_duration_list[
                                        current_text_idx] + chunk_overlap

                            searched_best_score = np.inf
                            searched_best_pred_tokens = None

                            if start_idx == 0:
                                for search_num in range(search_times):
                                    # random a starting token for dance, select from token data
                                    trg_seq = torch.LongTensor(audio_feature.size(0), 1).fill_(
                                        random.choice(m_tokens.reshape(-1))).to(
                                        audio_feature).long()

                                    pred_tokens = a2d_transformer.sample_with_trg_seq_double_enc_output(
                                        audio_feature[:, start_idx:end_idx],
                                        enc_output_a=atm_result_dict[name]['enc_output_a'][:, start_idx:end_idx],
                                        src_mask_a=atm_result_dict[name]['src_mask_a'][:, :, start_idx:end_idx],
                                        weight_a=atm_result_dict[name]['weight_a'][start_idx + chunk_overlap:end_idx],
                                        enc_output_b=atm_result_dict[name]['enc_output_b'][:, start_idx:end_idx],
                                        src_mask_b=atm_result_dict[name]['src_mask_b'][:, :, start_idx:end_idx],
                                        trg_sos=opt.mot_start_idx, trg_eos=opt.mot_end_idx,
                                        trg_seq=trg_seq,
                                        start_idx=start_idx + 1, end_idx=end_idx,
                                        sample=opt.sample, top_k=opt.top_k)

                                    if not flag_involve_text:
                                        break
                                    else:
                                        s = chunk_overlap
                                        d = chunk_overlap + text_duration_list[current_text_idx]
                                        score = measure_text2mostion_similarity_by_token(
                                            t2m_sentence, pred_tokens[:, s:d])
                                        if score < searched_best_score:
                                            searched_best_score = score * 1.
                                            searched_best_pred_tokens = pred_tokens
                                        if searched_best_score < search_threshold:
                                            break
                                if flag_involve_text:
                                    print('[ATM] generate action with score {:.5} with search_num {} for caption '
                                          '{}'.format(searched_best_score, search_num, t2m_sentence))
                                # pred_tokens_list.append(pred_tokens[:, 1:])
                                pred_tokens_list.append(pred_tokens[:, :]) # 1027
                            else:
                                for search_num in range(search_times):
                                    pred_tokens = a2d_transformer.sample_with_trg_seq_double_enc_output(
                                        audio_feature[:, start_idx:end_idx],
                                        enc_output_a=atm_result_dict[name]['enc_output_a'][:, start_idx:end_idx],
                                        src_mask_a=atm_result_dict[name]['src_mask_a'][:, :, start_idx:end_idx],
                                        weight_a=atm_result_dict[name]['weight_a'][start_idx + chunk_overlap:end_idx],
                                        enc_output_b=atm_result_dict[name]['enc_output_b'][:, start_idx:end_idx],
                                        src_mask_b=atm_result_dict[name]['src_mask_b'][:, :, start_idx:end_idx],
                                        trg_sos=opt.mot_start_idx, trg_eos=opt.mot_end_idx,
                                        # trg_seq=pred_tokens[:, -chunk_overlap:],
                                        trg_seq=pred_tokens_list[-1][:, -chunk_overlap:],  # 1027改.
                                        start_idx=start_idx + chunk_overlap, end_idx=end_idx,
                                        sample=opt.sample, top_k=opt.top_k)

                                    if not flag_involve_text:
                                        break
                                    else:
                                        s = chunk_overlap
                                        d = chunk_overlap + text_duration_list[current_text_idx]
                                        t2m_sentence = text_content_list[current_text_idx]
                                        score = measure_text2mostion_similarity_by_token(
                                            t2m_sentence, pred_tokens[:, s:d])
                                        if score < searched_best_score:
                                            searched_best_score = score * 1.
                                            searched_best_pred_tokens = pred_tokens
                                        if searched_best_score < search_threshold:
                                            break

                                if flag_involve_text:
                                    print('[ATM] generate action with score {:.5} with search_num {} for caption '
                                          '{}'.format(searched_best_score, search_num, t2m_sentence))
                                pred_tokens_list.append(pred_tokens[:, chunk_overlap:])

                            # update start_idx
                            start_idx = end_idx - chunk_overlap

                        # pred_tokens = pred_tokens[:, 1:]
                        pred_tokens = torch.cat(pred_tokens_list, dim=-1)
                        print('Sampled Tokens %02d' % t)
                        # print(pred_tokens[0])
                        if len(pred_tokens[0]) == 0:
                            continue
                        # double check if the generated motion = feature length
                        assert pred_tokens.shape[1] == audio_feature.shape[1], 'duration mismatch!'

                        vq_latent = quantizer.get_codebook_entry(pred_tokens)
                        gen_motion = vq_decoder(vq_latent)


                        atm_result_dict[name]['result_%02d' % t]['motion'] = gen_motion.cpu().numpy()
                        atm_result_dict[name]['result_%02d' % t]['length'] = len(gen_motion[0])

                        print('calculate the text2motion score in new generated dance')
                        for i, txt_key in enumerate(t2m_result_dict):
                            s = text_start_list[i] * int(
                                2 ** opt.n_down)  # because this is on frame level, not token level
                            d = text_duration_list[i] * int(
                                2 ** opt.n_down)  # because this is on frame level, not token level
                            t2m_feat60fps24j = gen_motion.cpu().numpy()[0, s:s + d]
                            t2m_sentence = t2m_result_dict[txt_key]['caption'][0]
                            score = wrap_measure_distance_byfeat60fps24j(t2m_sentence, t2m_feat60fps24j)
                            print(
                                'mix_mode: [{}] generated new t2m pred-motion token {} with Score {:.5} at time {}s'.format(
                                    mix_mode, t2m_result_dict[txt_key]['caption'][0], score, (s / 60)))
                            atm_result_dict[name]['result_%02d' % t]['score_list'].append(score)

                            title_line = ' caption: {}\nstart_time: {}s duration: {}s score: {:.3}'.format(
                                t2m_sentence, (s / 60), (d / 60), score
                            )
                            atm_result_dict[name]['result_%02d' % t]['title_list'].append(title_line)

                        num_try = num_try + 1
                        if max(atm_result_dict[name]['result_%02d' % t]['score_list']) < plot_score_threshold:
                            print('plot_score_threshold ok at trying time {}/{}'.format(t, opt.repeat_times))
                            break

                # result_dict[name] = item_dict
                if i + 1 >= opt.num_results:
                    break

        result_dict = copy.deepcopy(atm_result_dict)
        print('Animating atm Results')
        '''Animating atm Results'''
        for i, (key, item) in enumerate(result_dict.items()):
            print('%02d_%03d' % (i, opt.num_results))
            aud_name = item['caption']
            # gt_motions = item['gt_motion']
            # joint_save_path = pjoin(opt.joint_dir, key)
            animation_save_path = pjoin(opt.animation_dir, key)

            # os.makedirs(joint_save_path, exist_ok=True)
            os.makedirs(animation_save_path, exist_ok=True)

            # np.save(pjoin(joint_save_path, 'gt_motions.npy'), gt_motions)
            # plot_t2m(gt_motions, aud_name, pjoin(animation_save_path, 'gt_motion'))
            for t in range(opt.repeat_times):
                sub_dict = item['result_%02d' % t]
                motion = sub_dict['motion']
                if max(sub_dict['score_list']) > plot_score_threshold:
                    continue
                score_name = combine_score_name(sub_dict['score_list'])
                # np.save(pjoin(joint_save_path, 'gen_motion_%02d_L%03d.npy' % (t, motion.shape[1])), motion)
                title_cap = [combine_title_list(aud_name, sub_dict['title_list'])]
                plot_t2m(motion, title_cap,
                         pjoin(animation_save_path, 'atm_lf3_{}_'.format(mix_mode) + time.strftime("%H%M%S") +
                               'gen_motion_%02d_L%03d_%s' % (t, motion.shape[1], score_name)),
                         add_audio=True, audio_name=aud_name[0])

