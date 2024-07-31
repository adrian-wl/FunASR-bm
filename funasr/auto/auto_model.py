#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import json
import time
import copy
import torch
import random
import string
import logging
import os.path
import numpy as np
from tqdm import tqdm

from funasr.utils.misc import deep_update
from funasr.register import tables
from funasr.utils.load_utils import load_bytes
from funasr.download.file import download_from_url
from funasr.utils.timestamp_tools import timestamp_sentence
from funasr.download.download_from_hub import download_model
from funasr.utils.vad_utils import slice_padding_audio_samples
from funasr.utils.load_utils import load_audio_text_image_video
from funasr.train_utils.set_all_random_seed import set_all_random_seed
from funasr.train_utils.load_pretrained_model import load_pretrained_model
from funasr.utils import export_utils

try:
    from funasr.models.campplus.utils import sv_chunk, postprocess, distribute_spk
    from funasr.models.campplus.cluster_backend import ClusterBackend
except:
    pass


def prepare_data_iterator(data_in, input_len=None, data_type=None, key=None):
    """

    :param input:
    :param input_len:
    :param data_type:
    :param frontend:
    :return:
    """
    data_list = []
    key_list = []
    filelist = [".scp", ".txt", ".json", ".jsonl", ".text"]

    chars = string.ascii_letters + string.digits
    if isinstance(data_in, str) and data_in.startswith('http'): # url
        data_in = download_from_url(data_in)

    if isinstance(data_in, str) and os.path.exists(data_in): # wav_path; filelist: wav.scp, file.jsonl;text.txt;
        _, file_extension = os.path.splitext(data_in)
        file_extension = file_extension.lower()
        if file_extension in filelist: #filelist: wav.scp, file.jsonl;text.txt;
            with open(data_in, encoding='utf-8') as fin:
                for line in fin:
                    key = "rand_key_" + ''.join(random.choice(chars) for _ in range(13))
                    if data_in.endswith(".jsonl"): #file.jsonl: json.dumps({"source": data})
                        lines = json.loads(line.strip())
                        data = lines["source"]
                        key = data["key"] if "key" in data else key
                    else: # filelist, wav.scp, text.txt: id \t data or data
                        lines = line.strip().split(maxsplit=1)
                        data = lines[1] if len(lines)>1 else lines[0]
                        key = lines[0] if len(lines)>1 else key

                    data_list.append(data)
                    key_list.append(key)
        else:
            if key is None:
                key = "rand_key_" + ''.join(random.choice(chars) for _ in range(13))
            data_list = [data_in]
            key_list = [key]
    elif isinstance(data_in, (list, tuple)):
        if data_type is not None and isinstance(data_type, (list, tuple)): # mutiple inputs
            data_list_tmp = []
            for data_in_i, data_type_i in zip(data_in, data_type):
                key_list, data_list_i = prepare_data_iterator(data_in=data_in_i, data_type=data_type_i)
                data_list_tmp.append(data_list_i)
            data_list = []
            for item in zip(*data_list_tmp):
                data_list.append(item)
        else:
            # [audio sample point, fbank, text]
            data_list = data_in
            key_list = ["rand_key_" + ''.join(random.choice(chars) for _ in range(13)) for _ in range(len(data_in))]
    else: # raw text; audio sample point, fbank; bytes
        if isinstance(data_in, bytes): # audio bytes
            data_in = load_bytes(data_in)
        if key is None:
            key = "rand_key_" + ''.join(random.choice(chars) for _ in range(13))
        data_list = [data_in]
        key_list = [key]

    return key_list, data_list


class AutoModel:

    def __init__(self, **kwargs):
        if not kwargs.get("disable_log", True):
            tables.print()
        model, kwargs = self.build_model(**kwargs)

        # if vad_model is not None, build vad model else None
        vad_model = kwargs.get("vad_model", None)
        vad_kwargs = {} if kwargs.get("vad_kwargs", {}) is None else kwargs.get("vad_kwargs", {})
        if vad_model is not None:
            logging.info("Building VAD model.")
            vad_kwargs["model"] = vad_model
            vad_kwargs["model_revision"] = kwargs.get("vad_model_revision", "master")
            vad_kwargs["device"] = kwargs["device"]
            vad_kwargs["dev_id"] = kwargs["dev_id"]
            vad_model, vad_kwargs = self.build_model(**vad_kwargs)

        # if punc_model is not None, build punc model else None
        punc_model = kwargs.get("punc_model", None)
        punc_kwargs = {} if kwargs.get("punc_kwargs", {}) is None else kwargs.get("punc_kwargs", {})
        if punc_model is not None:
            logging.info("Building punc model.")
            punc_kwargs["model"] = punc_model
            punc_kwargs["model_revision"] = kwargs.get("punc_model_revision", "master")
            punc_kwargs["device"] = kwargs["device"]
            punc_kwargs["dev_id"] = kwargs["dev_id"]
            punc_model, punc_kwargs = self.build_model(**punc_kwargs)

        # if spk_model is not None, build spk model else None
        spk_model = kwargs.get("spk_model", None)
        spk_kwargs = {} if kwargs.get("spk_kwargs", {}) is None else kwargs.get("spk_kwargs", {})
        if spk_model is not None:
            logging.info("Building SPK model.")
            spk_kwargs["model"] = spk_model
            spk_kwargs["model_revision"] = kwargs.get("spk_model_revision", "master")
            spk_kwargs["device"] = kwargs["device"]
            spk_model, spk_kwargs = self.build_model(**spk_kwargs)
            self.cb_model = ClusterBackend().to(kwargs["device"])
            spk_mode = kwargs.get("spk_mode", 'punc_segment')
            if spk_mode not in ["default", "vad_segment", "punc_segment"]:
                logging.error("spk_mode should be one of default, vad_segment and punc_segment.")
            self.spk_mode = spk_mode

        self.kwargs = kwargs
        self.model = model
        self.vad_model = vad_model
        self.vad_kwargs = vad_kwargs
        self.punc_model = punc_model
        self.punc_kwargs = punc_kwargs
        self.spk_model = spk_model
        self.spk_kwargs = spk_kwargs
        self.model_path = kwargs.get("model_path")

    def build_model(self, **kwargs):
        assert "model" in kwargs
        # if "model_conf" not in kwargs:
        #     logging.info("download models from model hub: {}".format(kwargs.get("hub", "ms")))
        #     print(kwargs['model'])
        #     kwargs = download_model(**kwargs)
        dev_id = kwargs.get("dev_id", 0)
        if kwargs['model'] == 'iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404':
            #kwargs = {'model': 'ContextualParaformer', 'model_conf': {'ctc_weight': 0.0, 'lsm_weight': 0.1, 'length_normalized_loss': True, 'predictor_weight': 1.0, 'predictor_bias': 1, 'sampling_ratio': 0.75, 'inner_dim': 512}, 'encoder': 'SANMEncoder', 'encoder_conf': {'output_size': 512, 'attention_heads': 4, 'linear_units': 2048, 'num_blocks': 50, 'dropout_rate': 0.1, 'positional_dropout_rate': 0.1, 'attention_dropout_rate': 0.1, 'input_layer': 'pe', 'pos_enc_class': 'SinusoidalPositionEncoder', 'normalize_before': True, 'kernel_size': 11, 'sanm_shfit': 0, 'selfattention_layer_type': 'sanm'}, 'decoder': 'ContextualParaformerDecoder', 'decoder_conf': {'attention_heads': 4, 'linear_units': 2048, 'num_blocks': 16, 'dropout_rate': 0.1, 'positional_dropout_rate': 0.1, 'self_attention_dropout_rate': 0.1, 'src_attention_dropout_rate': 0.1, 'att_layer_num': 16, 'kernel_size': 11, 'sanm_shfit': 0}, 'predictor': 'CifPredictorV2', 'predictor_conf': {'idim': 512, 'threshold': 1.0, 'l_order': 1, 'r_order': 1, 'tail_threshold': 0.45}, 'frontend': 'WavFrontend', 'frontend_conf': {'fs': 16000, 'window': 'hamming', 'n_mels': 80, 'frame_length': 25, 'frame_shift': 10, 'lfr_m': 7, 'lfr_n': 6, 'cmvn_file': './bmodel/asr/am.mvn'}, 'specaug': 'SpecAugLFR', 'specaug_conf': {'apply_time_warp': False, 'time_warp_window': 5, 'time_warp_mode': 'bicubic', 'apply_freq_mask': True, 'freq_mask_width_range': [0, 30], 'lfr_rate': 6, 'num_freq_mask': 1, 'apply_time_mask': True, 'time_mask_width_range': [0, 12], 'num_time_mask': 1}, 'train_conf': {'accum_grad': 1, 'grad_clip': 5, 'max_epoch': 150, 'val_scheduler_criterion': ['valid', 'acc'], 'best_model_criterion': [['valid', 'acc', 'max']], 'keep_nbest_models': 10, 'log_interval': 50}, 'optim': 'adam', 'optim_conf': {'lr': 0.0005}, 'scheduler': 'warmuplr', 'scheduler_conf': {'warmup_steps': 30000}, 'dataset': 'AudioDatasetHotword', 'dataset_conf': {'index_ds': 'IndexDSJsonl', 'batch_sampler': 'DynamicBatchLocalShuffleSampler', 'batch_type': 'example', 'batch_size': 1, 'max_token_length': 2048, 'buffer_size': 500, 'shuffle': True, 'num_workers': 0}, 'tokenizer': 'CharTokenizer', 'tokenizer_conf': {'unk_symbol': '<unk>', 'split_with_space': True, 'token_list': './bmodel/asr/tokens.json', 'seg_dict_file': './bmodel/asr/seg_dict'}, 'ctc_conf': {'dropout_rate': 0.0, 'ctc_type': 'builtin', 'reduce': True, 'ignore_nan_grad': True}, 'normalize': None, 'init_param': '/root/.cache/modelscope/hub/iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404/model.pt', 'config': '/root/.cache/modelscope/hub/iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404/config.yaml', 'vad_model': 'iic/speech_fsmn_vad_zh-cn-16k-common-pytorch', 'punc_model': 'iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch', 'model_path': '/root/.cache/modelscope/hub/iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404'}
            kwargs = {'model': 'ContextualParaformer', 'model_conf': {'ctc_weight': 0.0, 'lsm_weight': 0.1, 'length_normalized_loss': True, 'predictor_weight': 1.0, 'predictor_bias': 1, 'sampling_ratio': 0.75, 'inner_dim': 512}, 'encoder': 'SANMEncoder', 'encoder_conf': {'output_size': 512, 'attention_heads': 4, 'linear_units': 2048, 'num_blocks': 50, 'dropout_rate': 0.1, 'positional_dropout_rate': 0.1, 'attention_dropout_rate': 0.1, 'input_layer': 'pe', 'pos_enc_class': 'SinusoidalPositionEncoder', 'normalize_before': True, 'kernel_size': 11, 'sanm_shfit': 0, 'selfattention_layer_type': 'sanm'}, 'decoder': 'ContextualParaformerDecoder', 'decoder_conf': {'attention_heads': 4, 'linear_units': 2048, 'num_blocks': 16, 'dropout_rate': 0.1, 'positional_dropout_rate': 0.1, 'self_attention_dropout_rate': 0.1, 'src_attention_dropout_rate': 0.1, 'att_layer_num': 16, 'kernel_size': 11, 'sanm_shfit': 0}, 'predictor': 'CifPredictorV2', 'predictor_conf': {'idim': 512, 'threshold': 1.0, 'l_order': 1, 'r_order': 1, 'tail_threshold': 0.45}, 'frontend': 'WavFrontend', 'frontend_conf': {'fs': 16000, 'window': 'hamming', 'n_mels': 80, 'frame_length': 25, 'frame_shift': 10, 'lfr_m': 7, 'lfr_n': 6, 'cmvn_file': './bmodel/asr/am.mvn'}, 'specaug': 'SpecAugLFR', 'specaug_conf': {'apply_time_warp': False, 'time_warp_window': 5, 'time_warp_mode': 'bicubic', 'apply_freq_mask': True, 'freq_mask_width_range': [0, 30], 'lfr_rate': 6, 'num_freq_mask': 1, 'apply_time_mask': True, 'time_mask_width_range': [0, 12], 'num_time_mask': 1}, 'train_conf': {'accum_grad': 1, 'grad_clip': 5, 'max_epoch': 150, 'val_scheduler_criterion': ['valid', 'acc'], 'best_model_criterion': [['valid', 'acc', 'max']], 'keep_nbest_models': 10, 'log_interval': 50}, 'optim': 'adam', 'optim_conf': {'lr': 0.0005}, 'scheduler': 'warmuplr', 'scheduler_conf': {'warmup_steps': 30000}, 'dataset': 'AudioDatasetHotword', 'dataset_conf': {'index_ds': 'IndexDSJsonl', 'batch_sampler': 'DynamicBatchLocalShuffleSampler', 'batch_type': 'example', 'batch_size': 1, 'max_token_length': 2048, 'buffer_size': 500, 'shuffle': True, 'num_workers': 0}, 'tokenizer': 'CharTokenizer', 'tokenizer_conf': {'unk_symbol': '<unk>', 'split_with_space': True, 'token_list': './bmodel/asr/tokens.json', 'seg_dict_file': './bmodel/asr/seg_dict'}, 'ctc_conf': {'dropout_rate': 0.0, 'ctc_type': 'builtin', 'reduce': True, 'ignore_nan_grad': True}, 'normalize': None, 'init_param': '/root/.cache/modelscope/hub/iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404/model.pt', 'config': '/root/.cache/modelscope/hub/iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404/config.yaml', 'model_path': '/root/.cache/modelscope/hub/iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404'}
        elif kwargs['model'] == 'iic/speech_fsmn_vad_zh-cn-16k-common-pytorch':
            kwargs = {'frontend': 'WavFrontendOnline', 'frontend_conf': {'fs': 16000, 'window': 'hamming', 'n_mels': 80, 'frame_length': 25, 'frame_shift': 10, 'dither': 0.0, 'lfr_m': 5, 'lfr_n': 1, 'cmvn_file': './bmodel/fsmn/am.mvn'}, 'model': 'FsmnVADStreaming', 'model_conf': {'sample_rate': 16000, 'detect_mode': 1, 'snr_mode': 0, 'max_end_silence_time': 800, 'max_start_silence_time': 3000, 'do_start_point_detection': True, 'do_end_point_detection': True, 'window_size_ms': 200, 'sil_to_speech_time_thres': 150, 'speech_to_sil_time_thres': 150, 'speech_2_noise_ratio': 1.0, 'do_extend': 1, 'lookback_time_start_point': 200, 'lookahead_time_end_point': 100, 'max_single_segment_time': 60000, 'snr_thres': -100.0, 'noise_frame_num_used_for_snr': 100, 'decibel_thres': -100.0, 'speech_noise_thres': 0.6, 'fe_prior_thres': 0.0001, 'silence_pdf_num': 1, 'sil_pdf_ids': [0], 'speech_noise_thresh_low': -0.1, 'speech_noise_thresh_high': 0.3, 'output_frame_probs': False, 'frame_in_ms': 10, 'frame_length_ms': 25}, 'encoder': 'FSMN', 'encoder_conf': {'input_dim': 400, 'input_affine_dim': 140, 'fsmn_layers': 4, 'linear_dim': 250, 'proj_dim': 128, 'lorder': 20, 'rorder': 0, 'lstride': 1, 'rstride': 0, 'output_affine_dim': 140, 'output_dim': 248}, 'init_param': '/root/.cache/modelscope/hub/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch/model.pt', 'config': '/root/.cache/modelscope/hub/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch/config.yaml', 'model_revision': 'master', 'device': 'cpu', 'model_path': '/root/.cache/modelscope/hub/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch'}
        elif kwargs['model'] == 'iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch':
            kwargs = {'model': 'CTTransformer', 'model_conf': {'ignore_id': 0, 'embed_unit': 256, 'att_unit': 256, 'dropout_rate': 0.1, 'punc_list': ['<unk>', '_', '，', '。', '？', '、'], 'punc_weight': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'sentence_end_id': 3}, 'encoder': 'SANMEncoder', 'encoder_conf': {'input_size': 256, 'output_size': 256, 'attention_heads': 8, 'linear_units': 1024, 'num_blocks': 4, 'dropout_rate': 0.1, 'positional_dropout_rate': 0.1, 'attention_dropout_rate': 0.0, 'input_layer': 'pe', 'pos_enc_class': 'SinusoidalPositionEncoder', 'normalize_before': True, 'kernel_size': 11, 'sanm_shfit': 0, 'selfattention_layer_type': 'sanm', 'padding_idx': 0}, 'tokenizer': 'CharTokenizer', 'tokenizer_conf': {'unk_symbol': '<unk>', 'token_list': './bmodel/punc/tokens.json'}, 'init_param': '/root/.cache/modelscope/hub/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/model.pt', 'config': '/root/.cache/modelscope/hub/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/config.yaml', 'model_revision': 'master', 'device': 'cpu', 'model_path': '/root/.cache/modelscope/hub/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch'}
        elif kwargs['model'] == 'iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online':
            kwargs = {'model': 'ParaformerStreaming', 'model_conf': {'ctc_weight': 0.0, 'lsm_weight': 0.1, 'length_normalized_loss': True, 'predictor_weight': 1.0, 'predictor_bias': 1, 'sampling_ratio': 0.75}, 'encoder': 'SANMEncoderChunkOpt', 'encoder_conf': {'output_size': 512, 'attention_heads': 4, 'linear_units': 2048, 'num_blocks': 50, 'dropout_rate': 0.1, 'positional_dropout_rate': 0.1, 'attention_dropout_rate': 0.1, 'input_layer': 'pe_online', 'pos_enc_class': 'SinusoidalPositionEncoder', 'normalize_before': True, 'kernel_size': 11, 'sanm_shfit': 0, 'selfattention_layer_type': 'sanm', 'chunk_size': [12, 15], 'stride': [8, 10], 'pad_left': [0, 0], 'encoder_att_look_back_factor': [4, 4], 'decoder_att_look_back_factor': [1, 1]}, 'decoder': 'ParaformerSANMDecoder', 'decoder_conf': {'attention_heads': 4, 'linear_units': 2048, 'num_blocks': 16, 'dropout_rate': 0.1, 'positional_dropout_rate': 0.1, 'self_attention_dropout_rate': 0.1, 'src_attention_dropout_rate': 0.1, 'att_layer_num': 16, 'kernel_size': 11, 'sanm_shfit': 5}, 'predictor': 'CifPredictorV2', 'predictor_conf': {'idim': 512, 'threshold': 1.0, 'l_order': 1, 'r_order': 1, 'tail_threshold': 0.45}, 'frontend': 'WavFrontendOnline', 'frontend_conf': {'fs': 16000, 'window': 'hamming', 'n_mels': 80, 'frame_length': 25, 'frame_shift': 10, 'lfr_m': 7, 'lfr_n': 6, 'cmvn_file': './bmodel/asr_online/am.mvn'}, 'specaug': 'SpecAugLFR', 'specaug_conf': {'apply_time_warp': False, 'time_warp_window': 5, 'time_warp_mode': 'bicubic', 'apply_freq_mask': True, 'freq_mask_width_range': [0, 30], 'lfr_rate': 6, 'num_freq_mask': 1, 'apply_time_mask': True, 'time_mask_width_range': [0, 12], 'num_time_mask': 1}, 'train_conf': {'accum_grad': 1, 'grad_clip': 5, 'max_epoch': 150, 'val_scheduler_criterion': ['valid', 'acc'], 'best_model_criterion': [['valid', 'acc', 'max']], 'keep_nbest_models': 10, 'log_interval': 50}, 'optim': 'adam', 'optim_conf': {'lr': 0.0005}, 'scheduler': 'warmuplr', 'scheduler_conf': {'warmup_steps': 30000}, 'dataset': 'AudioDataset', 'dataset_conf': {'index_ds': 'IndexDSJsonl', 'batch_sampler': 'DynamicBatchLocalShuffleSampler', 'batch_type': 'example', 'batch_size': 1, 'max_token_length': 2048, 'buffer_size': 500, 'shuffle': True, 'num_workers': 0}, 'tokenizer': 'CharTokenizer', 'tokenizer_conf': {'unk_symbol': '<unk>', 'split_with_space': True, 'token_list': './bmodel/asr_online/tokens.json', 'seg_dict_file': './bmodel/asr_online/seg_dict'}, 'ctc_conf': {'dropout_rate': 0.0, 'ctc_type': 'builtin', 'reduce': True, 'ignore_nan_grad': True}, 'normalize': None, 'init_param': '/root/.cache/modelscope/hub/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/model.pt', 'config': '/root/.cache/modelscope/hub/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/config.yaml', 'model_revision': 'v2.0.4', 'ngpu': 1, 'ncpu': 4, 'device': 'cuda', 'disable_pbar': True, 'disable_log': True, 'model_path': '/root/.cache/modelscope/hub/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online'}
        elif kwargs['model'] == "CTTransformerStreaming":
            kwargs = {'model': 'CTTransformerStreaming', 'model_conf': {'ignore_id': 0, 'embed_unit': 256, 'att_unit': 256, 'dropout_rate': 0.1, 'punc_list': ['<unk>', '_', '，', '。', '？', '、'], 'punc_weight': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'sentence_end_id': 3}, 'encoder': 'SANMVadEncoder', 'encoder_conf': {'input_size': 256, 'output_size': 256, 'attention_heads': 8, 'linear_units': 1024, 'num_blocks': 3, 'dropout_rate': 0.1, 'positional_dropout_rate': 0.1, 'attention_dropout_rate': 0.0, 'input_layer': 'pe', 'pos_enc_class': 'SinusoidalPositionEncoder', 'normalize_before': True, 'kernel_size': 11, 'sanm_shfit': 5, 'selfattention_layer_type': 'sanm', 'padding_idx': 0}, 'tokenizer': 'CharTokenizer', 'tokenizer_conf': {'unk_symbol': '<unk>', 'token_list': '/root/.cache/modelscope/hub/iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727/tokens.json'}, 'init_param': '/root/.cache/modelscope/hub/iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727/model.pt', 'config': '/root/.cache/modelscope/hub/iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727/config.yaml', 'model_revision': 'master', 'device': 'cpu', 'dev_id': 0, 'model_path': '/root/.cache/modelscope/hub/iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727'}
        kwargs['dev_id'] = dev_id
        set_all_random_seed(kwargs.get("seed", 0))

        device = kwargs.get("device", "cuda")
        if not torch.cuda.is_available() or kwargs.get("ngpu", 1) == 0:
            device = "cpu"
            kwargs["batch_size"] = 1
        device = "cpu"
        kwargs["device"] = device

        torch.set_num_threads(kwargs.get("ncpu", 4))

        # build tokenizer
        tokenizer = kwargs.get("tokenizer", None)
        if tokenizer is not None:
            tokenizer_class = tables.tokenizer_classes.get(tokenizer)
            tokenizer = tokenizer_class(**kwargs.get("tokenizer_conf", {}))
            kwargs["token_list"] = tokenizer.token_list if hasattr(tokenizer, "token_list") else None
            kwargs["token_list"] = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else kwargs["token_list"]
            vocab_size = len(kwargs["token_list"]) if kwargs["token_list"] is not None else -1
        else:
            vocab_size = -1
        kwargs["tokenizer"] = tokenizer

        # build frontend
        frontend = kwargs.get("frontend", None)
        kwargs["input_size"] = None
        if frontend is not None:
            frontend_class = tables.frontend_classes.get(frontend)
            frontend = frontend_class(**kwargs.get("frontend_conf", {}))
            kwargs["input_size"] = frontend.output_size() if hasattr(frontend, "output_size") else None
        kwargs["frontend"] = frontend
        # build model
        model_class = tables.model_classes.get(kwargs["model"])
        from funasr.models.contextual_paraformer.model import ContextualParaformer
        from funasr.models.ct_transformer.model import CTTransformer
        from funasr.models.fsmn_vad_streaming.model import FsmnVADStreaming
        from funasr.models.paraformer_streaming.model import ParaformerStreaming
        from funasr.models.ct_transformer_streaming.model import CTTransformerStreaming
        if kwargs["model"] == 'ContextualParaformer': model_class = ContextualParaformer
        elif kwargs["model"] == 'CTTransformer': model_class = CTTransformer
        elif kwargs["model"] == 'FsmnVADStreaming': model_class = FsmnVADStreaming
        elif kwargs["model"] == 'ParaformerStreaming': model_class = ParaformerStreaming
        elif kwargs["model"] == 'CTTransformerStreaming': model_class = CTTransformerStreaming
        model_conf = {}
        deep_update(model_conf, kwargs.get("model_conf", {}))
        deep_update(model_conf, kwargs)
        model = model_class(**model_conf, vocab_size=vocab_size)
        # if kwargs['model'] == 'ContextualParaformer':
        #     emb_weight = torch.load("/workspace/tpu-mlir/case/asr/contextual_paraformer/embedding_weight.pt")
        #     model.bias_embed.weight = emb_weight
        model.to(device)

        # #init_param
        # init_param = kwargs.get("init_param", None)
        # if init_param is not None:
        #     if os.path.exists(init_param):
        #         logging.info(f"Loading pretrained params from {init_param}")
        #         load_pretrained_model(
        #             model=model,
        #             path=init_param,
        #             ignore_init_mismatch=kwargs.get("ignore_init_mismatch", True),
        #             oss_bucket=kwargs.get("oss_bucket", None),
        #             scope_map=kwargs.get("scope_map", []),
        #             excludes=kwargs.get("excludes", None),
        #         )
        #     else:
        #         print(f"error, init_param does not exist!: {init_param}")

        # if kwargs['model'] == 'ContextualParaformer':
        #     torch.save(model.bias_embed.weight, '/workspace/tpu-mlir/case/asr/contextual_paraformer/embedding_weight.pt')

        return model, kwargs

    def __call__(self, *args, **cfg):
        kwargs = self.kwargs
        deep_update(kwargs, cfg)
        res = self.model(*args, kwargs)
        return res

    def generate(self, input, input_len=None, **cfg):
        if self.vad_model is None:
            return self.inference(input, input_len=input_len, **cfg)

        else:
            return self.inference_with_vad(input, input_len=input_len, **cfg)

    def inference(self, input, input_len=None, model=None, kwargs=None, key=None, **cfg):
        kwargs = self.kwargs if kwargs is None else kwargs
        deep_update(kwargs, cfg)
        model = self.model if model is None else model
        model.eval()

        batch_size = kwargs.get("batch_size", 1)
        # if kwargs.get("device", "cpu") == "cpu":
        #     batch_size = 1

        key_list, data_list = prepare_data_iterator(input, input_len=input_len, data_type=kwargs.get("data_type", None), key=key)

        speed_stats = {}
        asr_result_list = []
        num_samples = len(data_list)
        disable_pbar = self.kwargs.get("disable_pbar", False)
        pbar = tqdm(colour="blue", total=num_samples, dynamic_ncols=True) if not disable_pbar else None
        time_speech_total = 0.0
        time_escape_total = 0.0
        for beg_idx in range(0, num_samples, batch_size):
            end_idx = min(num_samples, beg_idx + batch_size)
            data_batch = data_list[beg_idx:end_idx]
            key_batch = key_list[beg_idx:end_idx]
            batch = {"data_in": data_batch, "key": key_batch}

            if (end_idx - beg_idx) == 1 and kwargs.get("data_type", None) == "fbank": # fbank
                batch["data_in"] = data_batch[0]
                batch["data_lengths"] = input_len

            time1 = time.perf_counter()
            with torch.no_grad():
                 res = model.inference(**batch, **kwargs)
                 if isinstance(res, (list, tuple)):
                    results = res[0]
                    meta_data = res[1] if len(res) > 1 else {}
            time2 = time.perf_counter()

            asr_result_list.extend(results)

            # batch_data_time = time_per_frame_s * data_batch_i["speech_lengths"].sum().item()
            batch_data_time = meta_data.get("batch_data_time", -1)
            time_escape = time2 - time1
            speed_stats["load_data"] = meta_data.get("load_data", 0.0)
            speed_stats["extract_feat"] = meta_data.get("extract_feat", 0.0)
            speed_stats["forward"] = f"{time_escape:0.3f}"
            speed_stats["batch_size"] = f"{len(results)}"
            speed_stats["rtf"] = f"{(time_escape) / batch_data_time:0.3f}"
            description = (
                f"{speed_stats}, "
            )
            if pbar:
                pbar.update(1)
                pbar.set_description(description)
            time_speech_total += batch_data_time
            time_escape_total += time_escape

        if pbar:
            # pbar.update(1)
            pbar.set_description(f"rtf_avg: {time_escape_total/time_speech_total:0.3f}")
        torch.cuda.empty_cache()
        return asr_result_list

    def inference_with_vad(self, input, input_len=None, **cfg):
        kwargs = self.kwargs
        # step.1: compute the vad model
        deep_update(self.vad_kwargs, cfg)
        beg_vad = time.time()
        res = self.inference(input, input_len=input_len, model=self.vad_model, kwargs=self.vad_kwargs, **cfg)
        end_vad = time.time()


        # step.2 compute asr model
        model = self.model
        deep_update(kwargs, cfg)
        batch_size = max(int(kwargs.get("batch_size_s", 300))*1000, 1)
        batch_size_threshold_ms = int(kwargs.get("batch_size_threshold_s", 60))*1000
        kwargs["batch_size"] = batch_size

        key_list, data_list = prepare_data_iterator(input, input_len=input_len, data_type=kwargs.get("data_type", None))
        results_ret_list = []
        time_speech_total_all_samples = 1e-6

        beg_total = time.time()
        pbar_total = tqdm(colour="red", total=len(res), dynamic_ncols=True) if not kwargs.get("disable_pbar", False) else None
        for i in range(len(res)):
            key = res[i]["key"]
            vadsegments = res[i]["value"]
            input_i = data_list[i]
            fs = kwargs["frontend"].fs if hasattr(kwargs["frontend"], "fs") else 16000
            speech = load_audio_text_image_video(input_i, fs=fs, audio_fs=kwargs.get("fs", 16000))
            speech_lengths = len(speech)
            n = len(vadsegments)
            data_with_index = [(vadsegments[i], i) for i in range(n)]
            sorted_data = sorted(data_with_index, key=lambda x: x[0][1] - x[0][0])
            results_sorted = []

            if not len(sorted_data):
                logging.info("decoding, utt: {}, empty speech".format(key))
                continue

            if len(sorted_data) > 0 and len(sorted_data[0]) > 0:
                batch_size = max(batch_size, sorted_data[0][0][1] - sorted_data[0][0][0])

            batch_size_ms_cum = 0
            beg_idx = 0
            beg_asr_total = time.time()
            time_speech_total_per_sample = speech_lengths/16000
            time_speech_total_all_samples += time_speech_total_per_sample

            # pbar_sample = tqdm(colour="blue", total=n, dynamic_ncols=True)

            all_segments = []
            for j, _ in enumerate(range(0, n)):
                # pbar_sample.update(1)
                batch_size_ms_cum += (sorted_data[j][0][1] - sorted_data[j][0][0])
                if j < n - 1 and (
                    batch_size_ms_cum + sorted_data[j + 1][0][1] - sorted_data[j + 1][0][0]) < batch_size and (
                    sorted_data[j + 1][0][1] - sorted_data[j + 1][0][0]) < batch_size_threshold_ms and (
                    j + 1 - beg_idx < 10):  # 10 is upper limit of asr_bmodel's batch:
                    continue
                batch_size_ms_cum = 0
                end_idx = j + 1
                speech_j, speech_lengths_j = slice_padding_audio_samples(speech, speech_lengths, sorted_data[beg_idx:end_idx])
                results = self.inference(speech_j, input_len=None, model=model, kwargs=kwargs, **cfg)
                if self.spk_model is not None:
                    # compose vad segments: [[start_time_sec, end_time_sec, speech], [...]]
                    for _b in range(len(speech_j)):
                        vad_segments = [[sorted_data[beg_idx:end_idx][_b][0][0]/1000.0,
                                        sorted_data[beg_idx:end_idx][_b][0][1]/1000.0,
                                        np.array(speech_j[_b])]]
                        segments = sv_chunk(vad_segments)
                        all_segments.extend(segments)
                        speech_b = [i[2] for i in segments]
                        spk_res = self.inference(speech_b, input_len=None, model=self.spk_model, kwargs=kwargs, **cfg)
                        results[_b]['spk_embedding'] = spk_res[0]['spk_embedding']
                beg_idx = end_idx
                if len(results) < 1:
                    continue
                results_sorted.extend(results)

            # end_asr_total = time.time()
            # time_escape_total_per_sample = end_asr_total - beg_asr_total
            # pbar_sample.update(1)
            # pbar_sample.set_description(f"rtf_avg_per_sample: {time_escape_total_per_sample / time_speech_total_per_sample:0.3f}, "
            #                      f"time_speech_total_per_sample: {time_speech_total_per_sample: 0.3f}, "
            #                      f"time_escape_total_per_sample: {time_escape_total_per_sample:0.3f}")

            restored_data = [0] * n
            for j in range(n):
                index = sorted_data[j][1]
                restored_data[index] = results_sorted[j]
            result = {}

            # results combine for texts, timestamps, speaker embeddings and others
            # TODO: rewrite for clean code
            for j in range(n):
                for k, v in restored_data[j].items():
                    if k.startswith("timestamp"):
                        if k not in result:
                            result[k] = []
                        for t in restored_data[j][k]:
                            t[0] += vadsegments[j][0]
                            t[1] += vadsegments[j][0]
                        result[k].extend(restored_data[j][k])
                    elif k == 'spk_embedding':
                        if k not in result:
                            result[k] = restored_data[j][k]
                        else:
                            result[k] = torch.cat([result[k], restored_data[j][k]], dim=0)
                    elif 'text' in k:
                        if k not in result:
                            result[k] = restored_data[j][k]
                        else:
                            result[k] += " " + restored_data[j][k]
                    else:
                        if k not in result:
                            result[k] = restored_data[j][k]
                        else:
                            result[k] += restored_data[j][k]

            return_raw_text = kwargs.get('return_raw_text', False)
            # step.3 compute punc model
            if self.punc_model is not None:
                if not len(result["text"]):
                    if return_raw_text:
                        result['raw_text'] = ''
                else:
                    deep_update(self.punc_kwargs, cfg)
                    punc_res = self.inference(result["text"], model=self.punc_model, kwargs=self.punc_kwargs, **cfg)
                    raw_text = copy.copy(result["text"])
                    if return_raw_text: result['raw_text'] = raw_text
                    result["text"] = punc_res[0]["text"]
            else:
                raw_text = None

            # speaker embedding cluster after resorted
            if self.spk_model is not None and kwargs.get('return_spk_res', True):
                if raw_text is None:
                    logging.error("Missing punc_model, which is required by spk_model.")
                all_segments = sorted(all_segments, key=lambda x: x[0])
                spk_embedding = result['spk_embedding']
                labels = self.cb_model(spk_embedding.cpu(), oracle_num=kwargs.get('preset_spk_num', None))
                # del result['spk_embedding']
                sv_output = postprocess(all_segments, None, labels, spk_embedding.cpu())
                if self.spk_mode == 'vad_segment':  # recover sentence_list
                    sentence_list = []
                    for res, vadsegment in zip(restored_data, vadsegments):
                        if 'timestamp' not in res:
                            logging.error("Only 'iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch' \
                                           and 'iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'\
                                           can predict timestamp, and speaker diarization relies on timestamps.")
                        sentence_list.append({"start": vadsegment[0],
                                              "end": vadsegment[1],
                                              "sentence": res['text'],
                                              "timestamp": res['timestamp']})
                elif self.spk_mode == 'punc_segment':
                    if 'timestamp' not in result:
                        logging.error("Only 'iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch' \
                                       and 'iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'\
                                       can predict timestamp, and speaker diarization relies on timestamps.")
                    sentence_list = timestamp_sentence(punc_res[0]['punc_array'],
                                                       result['timestamp'],
                                                       raw_text,
                                                       return_raw_text=return_raw_text)
                distribute_spk(sentence_list, sv_output)
                result['sentence_info'] = sentence_list
            elif kwargs.get("sentence_timestamp", False):
                if not len(result['text']):
                    sentence_list = []
                else:
                    sentence_list = timestamp_sentence(punc_res[0]['punc_array'],
                                                       result['timestamp'],
                                                       raw_text,
                                                       return_raw_text=return_raw_text)
                result['sentence_info'] = sentence_list
            if "spk_embedding" in result: del result['spk_embedding']

            result["key"] = key
            results_ret_list.append(result)
            end_asr_total = time.time()
            time_escape_total_per_sample = end_asr_total - beg_asr_total
            if pbar_total:
                pbar_total.update(1)
                pbar_total.set_description(f"rtf_avg: {time_escape_total_per_sample / time_speech_total_per_sample:0.3f}, "
                                 f"time_speech: {time_speech_total_per_sample: 0.3f}, "
                                 f"time_escape: {time_escape_total_per_sample:0.3f}")


        # end_total = time.time()
        # time_escape_total_all_samples = end_total - beg_total
        # print(f"rtf_avg_all: {time_escape_total_all_samples / time_speech_total_all_samples:0.3f}, "
        #                      f"time_speech_all: {time_speech_total_all_samples: 0.3f}, "
        #                      f"time_escape_all: {time_escape_total_all_samples:0.3f}")
        return results_ret_list

    def export(self, input=None, **cfg):

        """

        :param input:
        :param type:
        :param quantize:
        :param fallback_num:
        :param calib_num:
        :param opset_version:
        :param cfg:
        :return:
        """

        device = cfg.get("device", "cpu")
        model = self.model.to(device=device)
        kwargs = self.kwargs
        deep_update(kwargs, cfg)
        kwargs["device"] = device
        del kwargs["model"]
        model.eval()

        type = kwargs.get("type", "onnx")

        key_list, data_list = prepare_data_iterator(input, input_len=None, data_type=kwargs.get("data_type", None), key=None)
        #import pdb; pdb.set_trace()
        with torch.no_grad():

            if type == "onnx":
                export_dir = export_utils.export_onnx(
                                        model=model,
                                        data_in=data_list,
                                        **kwargs)
            else:
                export_dir = export_utils.export_torchscripts(
                                        model=model,
                                        data_in=data_list,
                                        **kwargs)

        return export_dir
