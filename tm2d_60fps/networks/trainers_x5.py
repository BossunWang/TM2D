import torch
import random
from networks.modules import *
from networks.transformer import *
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import tensorflow as tf
from collections import OrderedDict
from utils.utils import *
from os.path import join as pjoin
from data.dataset import collate_fn
import codecs as cs


class Logger(object):
    def __init__(self, log_dir):
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()


class Trainer(object):

    @staticmethod
    def zero_grad(opt_list):
        for opt in opt_list:
            opt.zero_grad()

    @staticmethod
    def clip_norm(network_list):
        for network in network_list:
            clip_grad_norm_(network.parameters(), 0.5)

    @staticmethod
    def step(opt_list):
        for opt in opt_list:
            opt.step()

    def ones_like(self, tensor, val=1.):
        return torch.FloatTensor(tensor.size()).fill_(val).to(self.opt.gpu_id)

    # @staticmethod
    def zeros_like(self, tensor, val=0.):
        return torch.FloatTensor(tensor.size()).fill_(val).to(self.opt.gpu_id)

    def forward(self, batch_data):
        pass

    def backward(self):
        pass

    def update(self):
        pass


class TransformerATMTrainerV2(Trainer):
    def __init__(self, args, atm_transformer):
        self.opt = args
        self.atm_transformer = atm_transformer

        self.a2d_transformer = atm_transformer.a2d_transformer

        self.t2m_transformer = atm_transformer.t2m_transformer
        self.m2m_transformer = atm_transformer.m2m_transformer
        # self.quantizer = quantizer
        # self.vq_decoder = vq_decoder
        self.device = args.device

        # self.trg_pad_index = args.trg_pad_index
        # self.trg_start_index = args.trg_start_index
        # self.trg_end_index = args.trg_end_index
        # self.trg_num_vocab = args.trg_num_vocab

        if args.is_train:
            self.logger = Logger(args.log_dir)
            self.l1_criterion = torch.nn.SmoothL1Loss()
            self.l1_byelement = torch.nn.SmoothL1Loss(reduction='none')

    def masked_loss(self, source, target, mask):
        """
        :param source: b x t x 1/512
        :param target: b x t x 1
        :param mask:  b x 1 x t
        :return:
        """
        tmp_mask = mask.squeeze().unsqueeze(-1)
        num_mask = torch.sum(tmp_mask) * source.shape[-1]
        loss_byelement = self.l1_byelement(source, target)
        loss_byelement_masked = loss_byelement * tmp_mask
        loss_mean = torch.sum(loss_byelement_masked) / num_mask
        return loss_mean

    def forward(self, batch_data):
        # audio_tokens, d_tokens, word_tokens, m_tokens = batch_data
        audio_feature, a_tokens_len, d_tokens, word_embeddings, word_tokens, sent_len, m_tokens = batch_data

        # prepare a dict for each process
        self.gold = OrderedDict({})
        self.trg_pred = OrderedDict({})
        self.enc_output = OrderedDict({})
        self.src_mask = OrderedDict({})

        # audio - dance
        self.d_tokens = d_tokens.detach().to(self.device).long()
        self.audio_feature = audio_feature.detach().to(self.device)
        self.a_tokens_len = a_tokens_len

        self.forward_a2d()
        # self.forward_d2d()

        # text - motion
        self.m_tokens = m_tokens.detach().to(self.device).long()
        self.word_tokens = word_tokens.detach().to(self.device).long()
        # self.word_embeddings = word_embeddings.detach().to(self.device)
        # self.sent_len = sent_len

        self.forward_t2m()
        # self.forward_m2m()

    def forward_a2d(self):
        """a2d encoder -> a2d decoder"""
        trg_input, self.gold['a2d'] = self.d_tokens[:, :-1], self.d_tokens[:, 1:]
        self.enc_output['a2d'], self.src_mask['a2d'] = self.a2d_transformer.encoding(self.audio_feature, self.a_tokens_len)
        self.trg_pred['a2d'] = self.a2d_transformer.decoding(trg_input, self.enc_output['a2d'],
                                                             self.src_mask['a2d'])

    # def forward_d2d(self):
    #     """d2d encoder -> a2d decoder"""
    #     self.enc_output['d2d'], self.src_mask['d2d'] = self.m2m_transformer.encoding(self.d_tokens)
    #     trg_input, self.gold['d2d'] = self.d_tokens[:, :-1], self.d_tokens[:, 1:]
    #     self.trg_pred['d2d'] = self.a2d_transformer.decoding(trg_input, self.enc_output['d2d'],
    #                                                          self.src_mask['d2d'])

    def forward_t2m(self):
        """t2m encoder -> t2m decoder"""
        trg_input, self.gold['t2m'] = self.m_tokens[:, :-1], self.m_tokens[:, 1:]
        self.enc_output['t2m'], self.src_mask['t2m'] = self.t2m_transformer.encoding(self.word_tokens)
        # self.trg_pred['t2m'] = self.t2m_transformer.decoding(trg_input, self.enc_output['t2m'],
        #                                                      self.src_mask['t2m'])
        self.trg_pred['t2m'] = self.a2d_transformer.decoding(trg_input, self.enc_output['t2m'],
                                                             self.src_mask['t2m'])

    # def forward_m2m(self):
    #     """m2m encoder -> t2m decoder"""
    #     trg_input, self.gold['m2m'] = self.m_tokens[:, :-1], self.m_tokens[:, 1:]
    #     # self.trg_pred['m2m'] = self.t2m_transformer.decoding(trg_input, self.enc_output['m2m'],
    #     #                                                      self.src_mask['m2m'])
    #     self.enc_output['m2m'], self.src_mask['m2m'] = self.m2m_transformer.encoding(self.m_tokens)
    #     self.trg_pred['m2m'] = self.a2d_transformer.decoding(trg_input, self.enc_output['m2m'],
    #                                                          self.src_mask['m2m'])

    def backward(self):
        self.loss = OrderedDict({})
        self.pred_seq = OrderedDict({})
        self.n_correct = OrderedDict({})
        self.n_word = OrderedDict({})
        loss_logs = OrderedDict({})
        # for key in ['a2d', 'd2d']:
        for key in ['a2d']:
            sub_loss_logs = self.backward_by_key(key)
            loss_logs.update(sub_loss_logs)

        # for key in ['t2m', 'm2m']:
        for key in ['t2m']:
            sub_loss_logs = self.backward_by_key(key)
            loss_logs.update(sub_loss_logs)

        self.loss['loss/net_a2d'] = self.loss['a2d'] * self.opt.lambda_a2d
        self.loss['loss/net_t2m'] = self.loss['t2m'] * self.opt.lambda_t2m

        self.loss['loss/net'] = self.loss['loss/net_a2d'] + self.loss['loss/net_t2m']
        loss_logs['loss/net_a2d'] = self.loss['loss/net_a2d'].item()
        loss_logs['loss/net_t2m'] = self.loss['loss/net_t2m'].item()
        loss_logs['loss/net'] = self.loss['loss/net'].item()

        loss_logs['accuracy/net_a2d'] = loss_logs['accuracy/a2d']
        loss_logs['accuracy/net_t2m'] = loss_logs['accuracy/t2m']

        loss_logs['accuracy/net'] = (loss_logs['accuracy/net_a2d'] + loss_logs['accuracy/net_t2m']) / 2
        return loss_logs

    def backward_by_key(self, key):
        # print(self.trg_pred.shape, self.gold.shape)
        trg_pred = self.trg_pred[key].view(-1, self.trg_pred[key].shape[-1]).clone()
        # print(trg_pred[0])
        gold = self.gold[key].contiguous().view(-1).clone()

        if key in ['a2d', 'd2d', 't2m', 'm2m']:
            pad_idx = self.opt.mot_pad_idx
        else:
            assert False, 'key error {}'.format(key)

        self.loss[key], self.pred_seq[key], self.n_correct[key], self.n_word[key] \
            = cal_performance(trg_pred, gold, pad_idx, smoothing=self.opt.label_smoothing)
        # print(gold, self.pred_seq)
        # loss is by sum in cal_performance, do average here
        self.loss[key] = self.loss[key] / self.n_word[key]

        loss_logs = OrderedDict({})
        loss_logs['loss/{}'.format(key)] = self.loss[key].item()  # / self.n_word[key]
        loss_logs['accuracy/{}'.format(key)] = self.n_correct[key] / self.n_word[key]

        return loss_logs

    def update(self):
        self.zero_grad([self.opt_atm_transformer])
        # time2_0 = time.time()
        # print("\t\t Zero Grad:%5f" % (time2_0 - time1))
        loss_logs = self.backward()
        self.loss['loss/net'].backward()

        # time2_3 = time.time()
        # print("\t\t Clip Norm :%5f" % (time2_3 - time2_2))
        self.step([self.opt_atm_transformer])

        return loss_logs

    def save(self, file_name, ep, total_it):

        # model_pos.load_state_dict(self.model_pos_train.state_dict())
        self.t2m_transformer.decoder.load_state_dict(self.a2d_transformer.decoder.state_dict())

        state = {
            # 'atm_transformer': self.atm_transformer.state_dict(),  # save space if resume not in use.
            'opt_atm_transformer': self.opt_atm_transformer.state_dict(),
            'a2d_transformer': self.a2d_transformer.state_dict(),
            't2m_transformer': self.t2m_transformer.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.atm_transformer.load_state_dict(checkpoint['atm_transformer'])

        self.opt_atm_transformer.load_state_dict(checkpoint['opt_atm_transformer'])
        # if self.opt.use_gan:
        #     self.discriminator.load_state_dict(checkpoint['discriminator'])
        #     self.opt_discriminator.load_state_dict(checkpoint['opt_discriminator'])
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_dataloader, val_dataloader, plot_eval):
        self.atm_transformer.to(self.device)
        # self.vq_decoder.to(self.device)
        # self.quantizer.to(self.device)

        self.opt_atm_transformer = optim.Adam(self.atm_transformer.parameters(), lr=self.opt.lr)

        epoch = 0
        it = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_dataloader)
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_dataloader), len(val_dataloader)))
        val_loss = 0
        val_accuracy = 0
        min_val_loss = np.inf
        min_val_epoch = epoch
        logs = OrderedDict()
        val_logs = OrderedDict()
        while epoch < self.opt.max_epoch:
            for i, batch_data in enumerate(train_dataloader):
                self.atm_transformer.train()

                self.forward(batch_data)

                log_dict = self.update()
                # continue
                # time3 = time.time()
                # print('Update Time: %.5f s' % (time3 - time2))
                # time0 = time3
                for k, v in log_dict.items():
                    if k not in logs:
                        logs[k] = v
                    else:
                        logs[k] += v

                it += 1
                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict({'val_loss': val_loss, 'val_accuracy': val_accuracy})
                    self.logger.scalar_summary('val_loss', val_loss, it)
                    self.logger.scalar_summary('val_accuracy', val_accuracy, it)

                    # plot the training log (loss, acc) in a sliding average by log_every
                    for tag, value in logs.items():
                        self.logger.scalar_summary(tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = OrderedDict()
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch, i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            epoch += 1
            if epoch % self.opt.save_every_e == 0:
                self.save(pjoin(self.opt.model_dir, 'E%04d.tar' % (epoch)), epoch, total_it=it)

            print('Validation time:')

            val_loss = 0
            val_accuracy = 0
            with torch.no_grad():
                for i, batch_data in enumerate(val_dataloader):
                    self.forward(batch_data)
                    val_log_dict = self.backward()
                    # val_loss += self.loss.item() / self.n_word
                    # val_accuracy += self.n_correct / self.n_word
                    val_loss += val_log_dict['loss/net']
                    val_accuracy += val_log_dict['accuracy/net']
                    # val_loss_rec += self.l1_criterion(self.recon_motions, self.motions).item()
                    # val_loss_emb += self.embedding_loss.item()
                    for k, v in val_log_dict.items():
                        if 'val_{}'.format(k) not in val_logs:
                            val_logs['val_{}'.format(k)] = v
                        else:
                            val_logs['val_{}'.format(k)] += v

            val_loss = val_loss / len(val_dataloader)
            val_accuracy = val_accuracy / len(val_dataloader)

            # plot the training log (loss, acc) in a sliding average by log_every
            for tag, value in val_logs.items():
                self.logger.scalar_summary(tag, value / len(val_dataloader), epoch)
            val_logs = OrderedDict()


            print('Validation Loss: %.5f Validation Accuracy: %.4f' % (val_loss, val_accuracy))

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                min_val_epoch = epoch
                self.save(pjoin(self.opt.model_dir, 'finest.tar'), epoch, it)
                print('Best Validation Model So Far!~')

            if epoch - min_val_epoch >= 5:
                print('Early Stopping!~')
                # break


class TransformerA2DTrainerV2(Trainer):
    """
    cp from TransformerATMTrainerV2()
    remove all but a2d part for training.
    """
    def __init__(self, args, a2d_transformer):
        self.opt = args
        self.a2d_transformer = a2d_transformer

        # self.quantizer = quantizer
        # self.vq_decoder = vq_decoder
        self.device = args.device

        if args.is_train:
            self.logger = Logger(args.log_dir)
            self.l1_criterion = torch.nn.SmoothL1Loss()
            self.l1_byelement = torch.nn.SmoothL1Loss(reduction='none')


    def forward(self, batch_data):
        # audio_tokens, d_tokens, word_tokens, m_tokens = batch_data
        # audio_feature, a_feature_len, d_tokens, word_embeddings, word_tokens, sent_len, m_tokens = batch_data
        audio_feature, _, _, a_feature_len, d_tokens, d_tokens_len = batch_data

        # prepare a dict for each process
        self.gold = OrderedDict({})
        self.trg_pred = OrderedDict({})
        self.enc_output = OrderedDict({})
        self.src_mask = OrderedDict({})

        # audio - dance
        self.d_tokens = d_tokens.detach().to(self.device).long()
        self.audio_feature = audio_feature.detach().to(self.device)
        self.a_feature_len = a_feature_len

        self.forward_a2d()
        # self.forward_d2d()

    def forward_a2d(self):
        """a2d encoder -> a2d decoder"""
        trg_input, self.gold['a2d'] = self.d_tokens[:, :-1], self.d_tokens[:, 1:]
        self.enc_output['a2d'], self.src_mask['a2d'] = self.a2d_transformer.encoding(self.audio_feature, self.a_feature_len)
        self.trg_pred['a2d'] = self.a2d_transformer.decoding(trg_input, self.enc_output['a2d'],
                                                             self.src_mask['a2d'])

    # def forward_d2d(self):
    #     """d2d encoder -> a2d decoder"""
    #     self.enc_output['d2d'], self.src_mask['d2d'] = self.m2m_transformer.encoding(self.d_tokens)
    #     trg_input, self.gold['d2d'] = self.d_tokens[:, :-1], self.d_tokens[:, 1:]
    #     self.trg_pred['d2d'] = self.a2d_transformer.decoding(trg_input, self.enc_output['d2d'],
    #                                                          self.src_mask['d2d'])

    def backward(self):
        self.loss = OrderedDict({})
        self.pred_seq = OrderedDict({})
        self.n_correct = OrderedDict({})
        self.n_word = OrderedDict({})
        loss_logs = OrderedDict({})
        # for key in ['a2d', 'd2d']:
        for key in ['a2d']:
            sub_loss_logs = self.backward_by_key(key)
            loss_logs.update(sub_loss_logs)

        # additional loss:
        # KL loss
        # for key in ['a2d']:

        self.loss['loss/net_a2d'] = self.loss['a2d'] * self.opt.lambda_a2d
        self.loss['loss/net'] = self.loss['loss/net_a2d']
        loss_logs['loss/net_a2d'] = self.loss['loss/net_a2d'].item()
        loss_logs['loss/net'] = self.loss['loss/net'].item()

        loss_logs['accuracy/net_a2d'] = loss_logs['accuracy/a2d']
        loss_logs['accuracy/net'] = loss_logs['accuracy/net_a2d']
        return loss_logs

    def backward_by_key(self, key):
        # print(self.trg_pred.shape, self.gold.shape)
        trg_pred = self.trg_pred[key].view(-1, self.trg_pred[key].shape[-1]).clone()
        # print(trg_pred[0])
        gold = self.gold[key].contiguous().view(-1).clone()

        if key in ['a2d', 'd2d', 't2m', 'm2m']:
            pad_idx = self.opt.mot_pad_idx
        else:
            assert False, 'key error {}'.format(key)

        self.loss[key], self.pred_seq[key], self.n_correct[key], self.n_word[key] \
            = cal_performance(trg_pred, gold, pad_idx, smoothing=self.opt.label_smoothing)
        # print(gold, self.pred_seq)
        # loss is by sum in cal_performance, do average here
        self.loss[key] = self.loss[key] / self.n_word[key]

        loss_logs = OrderedDict({})
        loss_logs['loss/{}'.format(key)] = self.loss[key].item()  # / self.n_word[key]
        loss_logs['accuracy/{}'.format(key)] = self.n_correct[key] / self.n_word[key]

        return loss_logs

    def update(self):
        self.zero_grad([self.opt_a2d_transformer])
        # time2_0 = time.time()
        # print("\t\t Zero Grad:%5f" % (time2_0 - time1))
        loss_logs = self.backward()
        self.loss['loss/net'].backward()

        # time2_3 = time.time()
        # print("\t\t Clip Norm :%5f" % (time2_3 - time2_2))
        self.step([self.opt_a2d_transformer])

        return loss_logs

    def save(self, file_name, ep, total_it):

        # model_pos.load_state_dict(self.model_pos_train.state_dict())
        # self.t2m_transformer.decoder.load_state_dict(self.a2d_transformer.decoder.state_dict())

        state = {
            'opt_a2d_transformer': self.opt_a2d_transformer.state_dict(),
            'a2d_transformer': self.a2d_transformer.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.a2d_transformer.load_state_dict(checkpoint['a2d_transformer'])
        self.opt_a2d_transformer.load_state_dict(checkpoint['opt_a2d_transformer'])
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_dataloader, val_dataloader, plot_eval):
        self.a2d_transformer.to(self.device)
        # self.vq_decoder.to(self.device)
        # self.quantizer.to(self.device)

        self.opt_a2d_transformer = optim.Adam(self.a2d_transformer.parameters(), lr=self.opt.lr)

        epoch = 0
        it = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_dataloader)
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_dataloader), len(val_dataloader)))
        val_loss = 0
        val_accuracy = 0
        min_val_loss = np.inf
        min_val_epoch = epoch
        logs = OrderedDict()
        val_logs = OrderedDict()
        while epoch < self.opt.max_epoch:
            for i, batch_data in enumerate(train_dataloader):
                self.a2d_transformer.train()

                self.forward(batch_data)

                log_dict = self.update()
                # continue
                # time3 = time.time()
                # print('Update Time: %.5f s' % (time3 - time2))
                # time0 = time3
                for k, v in log_dict.items():
                    if k not in logs:
                        logs[k] = v
                    else:
                        logs[k] += v

                it += 1
                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict({'val_loss': val_loss, 'val_accuracy': val_accuracy})
                    self.logger.scalar_summary('val_loss', val_loss, it)
                    self.logger.scalar_summary('val_accuracy', val_accuracy, it)

                    # plot the training log (loss, acc) in a sliding average by log_every
                    for tag, value in logs.items():
                        self.logger.scalar_summary(tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = OrderedDict()
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch, i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            epoch += 1
            if epoch % self.opt.save_every_e == 0:
                self.save(pjoin(self.opt.model_dir, 'E%04d.tar' % (epoch)), epoch, total_it=it)

            print('Validation time:')

            val_loss = 0
            val_accuracy = 0
            with torch.no_grad():
                for i, batch_data in enumerate(val_dataloader):
                    self.forward(batch_data)
                    val_log_dict = self.backward()
                    # val_loss += self.loss.item() / self.n_word
                    # val_accuracy += self.n_correct / self.n_word
                    val_loss += val_log_dict['loss/net']
                    val_accuracy += val_log_dict['accuracy/net']
                    # val_loss_rec += self.l1_criterion(self.recon_motions, self.motions).item()
                    # val_loss_emb += self.embedding_loss.item()
                    for k, v in val_log_dict.items():
                        if 'val_{}'.format(k) not in val_logs:
                            val_logs['val_{}'.format(k)] = v
                        else:
                            val_logs['val_{}'.format(k)] += v

            val_loss = val_loss / len(val_dataloader)
            val_accuracy = val_accuracy / len(val_dataloader)

            # plot the training log (loss, acc) in a sliding average by log_every
            for tag, value in val_logs.items():
                self.logger.scalar_summary(tag, value / len(val_dataloader), epoch)
            val_logs = OrderedDict()

            print('Validation Loss: %.5f Validation Accuracy: %.4f' % (val_loss, val_accuracy))

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                min_val_epoch = epoch
                self.save(pjoin(self.opt.model_dir, 'finest.tar'), epoch, it)
                print('Best Validation Model So Far!~')

            if epoch - min_val_epoch >= 5:
                print('Early Stopping!~')
                # break
