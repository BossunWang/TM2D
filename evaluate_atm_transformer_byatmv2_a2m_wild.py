import os

from os.path import join as pjoin

import utils.paramUtil as paramUtil
from options.evaluate_options import TestT2MOptions
from utils.plot_script import *

from networks.transformer_x_lf3 import TransformerV1
from networks.quantizer import *
from networks.modules import *
from data.dataset_eval import Motion2AudioEvalDataset4ATM, RawTextDatasetV2, WildAudioEvalDataset4ATM
from scripts.motion_process import *
from torch.utils.data import DataLoader
from utils.word_vectorizer import WordVectorizerV2
from utils.utils import *
import copy

"""
This file is to evaluate the music2dance
a subset of : evaluate_atm_transformer_byatmv5_lf3_withScoreV3.py
"""


def plot_t2m(data, captions, save_dir, add_audio=False, audio_name=None):
    data = data * std + mean
    for i in range(len(data)):
        assert len(data) == 1, 'not consider batch-wise'
        joint_data = data[i]
        caption = captions[i]
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
        # joint = motion_temporal_filter(joint)
        save_path = '%s_%02d.mp4' % (save_dir, i)
        np.save('%s_%02d.npy' % (save_dir.replace('animations', 'joints'), i), joint)
        # plot_3d_motion(save_path, kinematic_chain, joint, title=caption, fps=fps, radius=radius)
        plot_3d_motion(save_path, kinematic_chain, joint[::2], title=caption, fps=int(fps / 2), radius=radius)
        if add_audio:
            audio_name = caption if audio_name is None else audio_name
            combine_audio_to_video(video_path=save_path, name=audio_name)


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
    print(audio_dir_)
    print(video_path)
    output_path = video_path.replace('.mp4', '_audio.mp4')
    cmd_audio = f"ffmpeg -i {video_path} -i {audio_dir_} -map 0:v -map 1:a -c:v copy -shortest -y {output_path} -loglevel quiet"
    # ffmpeg -i a2d_gen_motion_00_L17384_00.mp4 -i ../Bailando/extra/Lollipop-Batte-Forte-Remix.MP3 -map 0:v -map 1:a -c:v copy -shortest -y a2d_gen_motion_00_L17384_00_audio.mp4 -loglevel quiet
    print('cmd_audio: ', cmd_audio)
    os.system(cmd_audio)


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

    # if opt.t2m_v2:
    a2d_transformer = TransformerV1(n_mot_vocab, opt.mot_pad_idx, d_src_word_vec=438, d_trg_word_vec=512,
                                    d_model=opt.d_model, d_inner=opt.d_inner_hid, n_enc_layers=opt.n_enc_layers,
                                    n_dec_layers=opt.n_dec_layers, n_head=opt.n_head, d_k=opt.d_k, d_v=opt.d_v,
                                    dropout=0.1,
                                    n_src_position=100, n_trg_position=100,
                                    trg_emb_prj_weight_sharing=opt.proj_share_weight)


    checkpoint = torch.load(
        pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'model', '%s.tar' % (opt.which_epoch)),
        map_location=opt.device)
    # t2m_transformer.load_state_dict(checkpoint['t2m_transformer'])
    a2d_transformer.load_state_dict(checkpoint['a2d_transformer'])
    print('Loading a2d_transformer model: Epoch %03d Total_Iter %03d' % (checkpoint['ep'], checkpoint['total_it']))

    return vq_decoder, quantizer, a2d_transformer


if __name__ == '__main__':
    parser = TestT2MOptions()
    parser.parser.add_argument('--temperature', type=float, default=1, help='temperature')
    parser.parser.add_argument('--slidwindow_len', type=int, default=50, help='temperature')
    parser.parser.add_argument('--slidwindow_overlap', type=int, default=1, help='temperature')
    opt = parser.parse()
    opt.max_text_len = 84  # 84 = 55x3/8x4# 55

    # EVAL_MODE = 'vis'  # 'metric'
    EVAL_MODE = opt.eval_mode

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
        opt.audio_dir = pjoin(opt.data_root, 'audio_vecs_wild_7.5fps')
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
        opt.audio_dir = pjoin(opt.data_root, 'audio_vecs_wild_7.5fps')
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


    # enc_channels = [1024, opt.dim_vq_latent]
    # dec_channels = [opt.dim_vq_latent, 1024, dim_pose]
    enc_channels = [1024, ] + [opt.dim_vq_latent, ] * (opt.n_down - 1)
    dec_channels = [opt.dim_vq_latent, ] * (opt.n_down - 1) + [1024, dim_pose]

    # vq_decoder, quantizer, a2d_transformer, t2m_transformer = build_models(opt)
    vq_decoder, quantizer, a2d_transformer = build_models(opt)

    # split_file = pjoin(opt.data_root, opt.split_file)
    # split_file = pjoin(opt.data_root, 'aistpp_test.txt')
    split_file = pjoin(opt.data_root, 'audio_wild.txt')

    aud_dataset = WildAudioEvalDataset4ATM(opt, mean, std, split_file)
    aud_data_loader = DataLoader(aud_dataset, batch_size=1, num_workers=1, shuffle=False, pin_memory=True)

    vq_decoder.to(opt.device)
    quantizer.to(opt.device)
    a2d_transformer.to(opt.device)
    # t2m_transformer.to(opt.device)

    vq_decoder.eval()
    quantizer.eval()
    a2d_transformer.eval()
    # t2m_transformer.eval()

    opt.repeat_times = opt.repeat_times if opt.sample else 1


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
            audio_feature, _, _, aud_name, _, _, _, _, _ = batch_data

            audio_feature = audio_feature.detach().to(opt.device).float()
            # a_tokens = a_tokens.detach().to(opt.device).long()

            print(aud_name[0])
            # name = 'C%03d' % (i)
            name = aud_name[0]
            item_dict = {
                'aud_name': aud_name[0],
                'caption': aud_name,
                # 'sent_lens': sent_lens[0],
                # 'gt_motion': motions[:, :m_lens[0]].cpu().numpy()
            }

            # chunk_len = 50
            # chunk_overlap = 1
            chunk_len = opt.slidwindow_len
            chunk_overlap = opt.slidwindow_overlap
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
            audio_feature, _, _, aud_name, _, _, _, _, _ = batch_data

            audio_feature = audio_feature.detach().to(opt.device).float()

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
                        # random a starting token for dance
                        trg_seq = torch.LongTensor(audio_feature.size(0), 1).fill_(np.random.randint(1024)).to(
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
    print('Animating a2d Results')
    '''Animating a2d Results'''
    if opt.repeat_times == 1:
        for i, (key, item) in enumerate(result_dict.items()):
            print('%02d_%03d' % (i, opt.num_results))
            aud_name = item['caption']
            # gt_motions = item['gt_motion']
            # joint_save_path = pjoin(opt.joint_dir, key)
            joint_save_path = opt.joint_dir
            animation_save_path = opt.animation_dir

            os.makedirs(joint_save_path, exist_ok=True)
            os.makedirs(animation_save_path, exist_ok=True)

            # for t in range(opt.repeat_times):
            t = 0
            sub_dict = item['result_%02d' % t]
            motion = sub_dict['motion']

            if EVAL_MODE == 'vis':
                # np.save(pjoin(joint_save_path, 'gen_motion_%02d_L%03d.npy' % (t, motion.shape[1])), motion)
                plot_t2m(motion, aud_name, pjoin(animation_save_path, aud_name[0]), add_audio=True)
            else:
                data = motion * std + mean
                joint_data = data[0]
                joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
                np.save(pjoin(joint_save_path, '{}.npy'.format(aud_name[0])), joint)
    else:
        for i, (key, item) in enumerate(result_dict.items()):
            print('%02d_%03d' % (i, opt.num_results))
            aud_name = item['caption']
            # gt_motions = item['gt_motion']
            # joint_save_path = pjoin(opt.joint_dir, key)
            joint_save_path = opt.joint_dir
            animation_save_path = pjoin(opt.animation_dir, key)

            os.makedirs(joint_save_path, exist_ok=True)
            os.makedirs(animation_save_path, exist_ok=True)

            # np.save(pjoin(joint_save_path, 'gt_motions.npy'), gt_motions)
            # plot_t2m(gt_motions, aud_name, pjoin(animation_save_path, 'gt_motion'))
            for t in range(opt.repeat_times):
                sub_dict = item['result_%02d' % t]
                motion = sub_dict['motion']

                if EVAL_MODE == 'vis':
                    # np.save(pjoin(joint_save_path, 'gen_motion_%02d_L%03d.npy' % (t, motion.shape[1])), motion)
                    plot_t2m(motion, aud_name, pjoin(animation_save_path, 'a2d_gen_motion_%02d_L%03d' % (t, motion.shape[1])), add_audio=True)
                else:
                    data = motion * std + mean
                    joint_data = data[0]
                    joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
                    np.save(pjoin(joint_save_path, '{}.npy'.format(aud_name[0])), joint)