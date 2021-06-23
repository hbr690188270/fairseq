import collections
from fairseq.models.wav2vec.wav2vec import Wav2VecModel
import itertools
import os
import math
import torch
import copy

# from fairseq import distributed_utils, options, progress_bar, tasks, utils
from fairseq import distributed_utils, options, progress_bar, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf

from fairseq.data import iterators
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter
from fairseq.tasks.speech_to_text import SpeechToTextTask_15
from fairseq.data.encoders import gpt2_bpe
import logging
from fairseq.models.speech_to_text.wav_bart import BART_Tokenizer
from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model
from fairseq.models.wav2vec.wav2vec2_asr import Wav2Vec2CtcConfig, Wav2VecCtc, Wav2VecEncoder
from fairseq.tasks.audio_pretraining import AudioPretrainingTask

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

path = '/data1/private/houbairu/model_cache/wav2vec_model/wav2vec_small_100h.pt'
# path = '/data1/private/houbairu/model_cache/wav2vec_model/wav2vec_small.pt'

param_dict = torch.load(path)
print(param_dict.keys())

cfg = param_dict['args']
print(type(cfg))
cfg = convert_namespace_to_omegaconf(cfg)  ##此处的cfg为wav2vec_asr的cfg，需要提取出wav2vec2model的cfg


tgt_dict_path = "/data1/private/houbairu/audio_dataset/orig_librispeech/aux_files/dict.ltr.txt"
cfg.task.data = "/data1/private/houbairu/audio_dataset/orig_librispeech/aux_files/"
cfg.task.labels = "ltr"

# print(cfg.model)
print(cfg.model.keys())

task = AudioPretrainingTask.setup_task(cfg = cfg.task)


wav2vec_asr = Wav2VecCtc.build_model(cfg.model, task)

model = param_dict['model']

wav2vec_asr.load_state_dict(model)
# print(wav2vec_asr.w2v_encoder)


