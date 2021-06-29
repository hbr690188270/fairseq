# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os.path as op
from argparse import Namespace

from torch.nn.modules.loss import CrossEntropyLoss

from fairseq.data import Dictionary, encoders
from fairseq.data.audio.speech_to_text_dataset import (
    SpeechToTextDataset2,
    SpeechToTextDataset_En, 
    SpeechToTextDataset_ENBart, 
    SpeechToTextDataset_ENBart_prev,
    S2TDataConfig,
    SpeechToTextDataset,
    SpeechToTextDatasetCreator,
    SpeechToTextDataset_libri_15,
    get_features_or_waveform
)
from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.models.speech_to_text.wav_bart import ASRModel, ASRModel_lstm_decoder, ASRModel_transformer_decoder, ASRModel_v2
import pickle
from fairseq.criterions import label_smoothed_cross_entropy, cross_entropy

logger = logging.getLogger(__name__)


@register_task("speech_to_text")
class SpeechToTextTask(LegacyFairseqTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument("data", help="manifest root path")
        parser.add_argument(
            "--config-yaml",
            type=str,
            default="config.yaml",
            help="Configuration YAML filename (under manifest root)",
        )
        parser.add_argument(
            "--max-source-positions",
            default=6000,
            type=int,
            metavar="N",
            help="max number of tokens in the source sequence",
        )
        parser.add_argument(
            "--max-target-positions",
            default=1024,
            type=int,
            metavar="N",
            help="max number of tokens in the target sequence",
        )

    def __init__(self, args, tgt_dict):
        super().__init__(args)
        self.tgt_dict = tgt_dict
        self.data_cfg = S2TDataConfig(op.join(args.data, args.config_yaml))

    @classmethod
    def setup_task(cls, args, **kwargs):
        data_cfg = S2TDataConfig(op.join(args.data, args.config_yaml))
        dict_path = op.join(args.data, data_cfg.vocab_filename)
        if not op.isfile(dict_path):
            raise FileNotFoundError(f"Dict not found: {dict_path}")
        tgt_dict = Dictionary.load(dict_path)
        logger.info(
            f"dictionary size ({data_cfg.vocab_filename}): " f"{len(tgt_dict):,}"
        )

        if getattr(args, "train_subset", None) is not None:
            if not all(s.startswith("train") for s in args.train_subset.split(",")):
                raise ValueError('Train splits should be named like "train*".')
        return cls(args, tgt_dict)

    def build_criterion(self, args):
        from fairseq import criterions

        if self.data_cfg.prepend_tgt_lang_tag and args.ignore_prefix_size != 1:
            raise ValueError(
                'Please set "--ignore-prefix-size 1" since '
                "target language ID token is prepended as BOS."
            )
        return criterions.build_criterion(args, self)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)
        self.datasets[split] = SpeechToTextDatasetCreator.from_tsv(
            self.args.data,
            self.data_cfg,
            split,
            self.tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.args.seed,
        )

    @property
    def target_dictionary(self):
        return self.tgt_dict

    @property
    def source_dictionary(self):
        return None

    def max_positions(self):
        return self.args.max_source_positions, self.args.max_target_positions

    def build_model(self, args):
        args.input_feat_per_channel = self.data_cfg.input_feat_per_channel
        args.input_channels = self.data_cfg.input_channels
        return super(SpeechToTextTask, self).build_model(args)

    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
    ):
        if self.data_cfg.prepend_tgt_lang_tag and args.prefix_size != 1:
            raise ValueError(
                'Please set "--prefix-size 1" since '
                "target language ID token is prepended as BOS."
            )
        lang_token_ids = {
            i
            for s, i in self.tgt_dict.indices.items()
            if SpeechToTextDataset.is_lang_tag(s)
        }
        extra_gen_cls_kwargs = {"symbols_to_strip_from_output": lang_token_ids}
        return super().build_generator(
            models, args, seq_gen_cls=None, extra_gen_cls_kwargs=extra_gen_cls_kwargs
        )

    def build_tokenizer(self, args):
        logger.info(f"pre-tokenizer: {self.data_cfg.pre_tokenizer}")
        return encoders.build_tokenizer(Namespace(**self.data_cfg.pre_tokenizer))

    def build_bpe(self, args):
        logger.info(f"tokenizer: {self.data_cfg.bpe_tokenizer}")
        return encoders.build_bpe(Namespace(**self.data_cfg.bpe_tokenizer))

    def get_interactive_tokens_and_lengths(self, lines, encode_fn):
        n_frames = [get_features_or_waveform(p).shape[0] for p in lines]
        return lines, n_frames

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        return SpeechToTextDataset(
            "interactive", False, self.data_cfg, src_tokens, src_lengths
        )

@register_task("speech_to_text_15") ## 2015 librispeech
class SpeechToTextTask_15(LegacyFairseqTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument("data", help="manifest root path")
        parser.add_argument(
            "--config-yaml",
            type=str,
            default="config.yaml",
            help="Configuration YAML filename (under manifest root)",
        )
        parser.add_argument(
            "--max-source-positions",
            default=6000,
            type=int,
            metavar="N",
            help="max number of tokens in the source sequence",
        )
        parser.add_argument(
            "--max-target-positions",
            default=1024,
            type=int,
            metavar="N",
            help="max number of tokens in the target sequence",
        )

    def __init__(self, args, tgt_dict):
        super().__init__(args)
        self.tgt_dict = tgt_dict

    @classmethod
    def setup_task(cls, args, tgt_dict_path, **kwargs):
        with open(tgt_dict_path,'rb') as f:
            tgt_dict = pickle.load(f)
        return cls(args, tgt_dict)

    def build_criterion(self, args):
        from fairseq import criterions

        if self.data_cfg.prepend_tgt_lang_tag and args.ignore_prefix_size != 1:
            raise ValueError(
                'Please set "--ignore-prefix-size 1" since '
                "target language ID token is prepended as BOS."
            )
        return criterions.build_criterion(args, self)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)
        self.datasets[split] = SpeechToTextDatasetCreator.from_tsv(
            self.args.data,
            self.data_cfg,
            split,
            self.tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.args.seed,
        )

    @property
    def target_dictionary(self):
        return self.tgt_dict

    @property
    def source_dictionary(self):
        return None

    def max_positions(self):
        return self.args.max_source_positions, self.args.max_target_positions

    def build_model(self, args):
        args.input_feat_per_channel = self.data_cfg.input_feat_per_channel
        args.input_channels = self.data_cfg.input_channels
        return super(SpeechToTextTask, self).build_model(args)

    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
    ):
        if self.data_cfg.prepend_tgt_lang_tag and args.prefix_size != 1:
            raise ValueError(
                'Please set "--prefix-size 1" since '
                "target language ID token is prepended as BOS."
            )
        lang_token_ids = {
            i
            for s, i in self.tgt_dict.indices.items()
            if SpeechToTextDataset.is_lang_tag(s)
        }
        extra_gen_cls_kwargs = {"symbols_to_strip_from_output": lang_token_ids}
        return super().build_generator(
            models, args, seq_gen_cls=None, extra_gen_cls_kwargs=extra_gen_cls_kwargs
        )

    def build_tokenizer(self, args):
        logger.info(f"pre-tokenizer: {self.data_cfg.pre_tokenizer}")
        return encoders.build_tokenizer(Namespace(**self.data_cfg.pre_tokenizer))

    def build_bpe(self, args):
        logger.info(f"tokenizer: {self.data_cfg.bpe_tokenizer}")
        return encoders.build_bpe(Namespace(**self.data_cfg.bpe_tokenizer))

    def get_interactive_tokens_and_lengths(self, lines, encode_fn):
        n_frames = [get_features_or_waveform(p).shape[0] for p in lines]
        return lines, n_frames

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        return SpeechToTextDataset(
            "interactive", False, self.data_cfg, src_tokens, src_lengths
        )


@register_task("speech_to_text2") ##2018 librispeech
class SpeechToTextTask2(LegacyFairseqTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument("data", help="manifest root path")
        parser.add_argument(
            "--max-source-positions",
            default=6000,
            type=int,
            metavar="N",
            help="max number of tokens in the source sequence",
        )
        parser.add_argument(
            "--max-target-positions",
            default=1024,
            type=int,
            metavar="N",
            help="max number of tokens in the target sequence",
        )

    def __init__(self, args, tgt_dict):
        super().__init__(args)
        self.tgt_dict = tgt_dict

    @classmethod
    def setup_task(cls, args, tgt_dict_path):
        with open(tgt_dict_path,'rb') as f:
            tgt_dict = pickle.load(f)  ## Dictironary
        # tgt_dict.finalize(nwords = 50000)
        # tgt_dict.finalize()

        print("bos idx: ", tgt_dict.bos_index)
        print("pad idx: ", tgt_dict.pad_index)
        print("eos idx: ", tgt_dict.eos_index)
        print("unk idx: ", tgt_dict.unk_index)

        logger.info(
            "dictionary size: " f"{len(tgt_dict):,}"
        )
        print(tgt_dict)
        print(len(tgt_dict))

        return cls(args, tgt_dict)

    def build_criterion(self, args):
        from fairseq import criterions

        # if self.data_cfg.prepend_tgt_lang_tag and args.ignore_prefix_size != 1:
        #     raise ValueError(
        #         'Please set "--ignore-prefix-size 1" since '
        #         "target language ID token is prepended as BOS."
        #     )
        # return criterions.build_criterion(args, self)
        return label_smoothed_cross_entropy.LabelSmoothedCrossEntropyCriterion(self, sentence_avg = False)
        # return cross_entropy.CrossEntropyCriterion2(self, sentence_avg = False)

    def load_dataset_prev(self, split, max_len = None, debug = False, max_frames = None, bpe_tokenizer = None):
        is_train_split = split.startswith("train")
        self.datasets[split] = SpeechToTextDataset_ENBart_prev(split = split,tgt_dict = self.tgt_dict, 
                            max_len = None, debug = debug, max_frames = max_frames, bpe_tokenizer = bpe_tokenizer)                            

    def load_dataset(self, split, max_len = None, debug = False, max_frames = None, bpe_tokenizer = None):
        is_train_split = split.startswith("train")
        # pre_tokenizer = self.build_tokenizer(self.args)
        # bpe_tokenizer = self.build_bpe(self.args)
        # self.datasets[split] = SpeechToTextDataset2(split = split,tgt_dict = self.tgt_dict, 
        #                     max_len = None, debug = debug, max_frames = max_frames)
        # self.datasets[split] = SpeechToTextDataset_En(split = split,tgt_dict = self.tgt_dict, 
        #                     max_len = None, debug = debug, max_frames = max_frames)

        # self.datasets[split] = SpeechToTextDataset_ENBart(split = split,tgt_dict = self.tgt_dict, 
        #                     max_len = None, debug = debug, max_frames = max_frames, bpe_tokenizer = bpe_tokenizer)                            
        self.datasets[split] = SpeechToTextDataset_libri_15(split = split,tgt_dict = self.tgt_dict, 
                            max_len = None, debug = debug, max_frames = max_frames, bpe_tokenizer = bpe_tokenizer)                            


    @property
    def target_dictionary(self):
        return self.tgt_dict

    @property
    def source_dictionary(self):
        return None

    def max_positions(self):
        return self.args.max_source_positions, self.args.max_target_positions

    def build_model(self, args, vocab_size):
        # return ASRModel(vocab_size = vocab_size, word_dictionary = self.tgt_dict)
        # return ASRModel_lstm_decoder(vocab_size = vocab_size, word_dictionary = self.tgt_dict)
        # return ASRModel_transformer_decoder(vocab_size = vocab_size, word_dictionary = self.tgt_dict, args = args)
        return ASRModel_v2(vocab_size = vocab_size, word_dictionary = self.tgt_dict)


    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
    ):
        if self.data_cfg.prepend_tgt_lang_tag and args.prefix_size != 1:
            raise ValueError(
                'Please set "--prefix-size 1" since '
                "target language ID token is prepended as BOS."
            )
        lang_token_ids = {
            i
            for s, i in self.tgt_dict.indices.items()
            if SpeechToTextDataset.is_lang_tag(s)
        }
        extra_gen_cls_kwargs = {"symbols_to_strip_from_output": lang_token_ids}
        return super().build_generator(
            models, args, seq_gen_cls=None, extra_gen_cls_kwargs=extra_gen_cls_kwargs
        )

    def build_tokenizer(self, args):
        logger.info(f"pre-tokenizer: {self.data_cfg.pre_tokenizer}")
        return encoders.build_tokenizer(Namespace(**self.data_cfg.pre_tokenizer))

    def build_bpe(self, args):
        logger.info(f"tokenizer: {self.data_cfg.bpe_tokenizer}")
        return encoders.build_bpe(Namespace(**self.data_cfg.bpe_tokenizer))

    def get_interactive_tokens_and_lengths(self, lines, encode_fn):
        n_frames = [get_features_or_waveform(p).shape[0] for p in lines]
        return lines, n_frames

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        return SpeechToTextDataset2(
            "interactive", False, self.data_cfg, src_tokens, src_lengths
        )

