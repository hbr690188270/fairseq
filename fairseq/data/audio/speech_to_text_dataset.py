# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import csv
import io
import logging
import os.path as op
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from fairseq.data import (
    ConcatDataset,
    Dictionary,
    FairseqDataset,
    ResamplingDataset,
    data_utils as fairseq_data_utils,
)
from fairseq.data.audio.audio_utils import get_fbank, get_waveform
from fairseq.data.audio.feature_transforms import CompositeAudioFeatureTransform
import soundfile as sf

logger = logging.getLogger(__name__)


class S2TDataConfig(object):
    """Wrapper class for data config YAML"""

    def __init__(self, yaml_path):
        try:
            import yaml
        except ImportError:
            print("Please install PyYAML to load YAML files for " "S2T data config")
        self.config = {}
        if op.isfile(yaml_path):
            try:
                with open(yaml_path) as f:
                    self.config = yaml.load(f, Loader=yaml.FullLoader)
            except Exception as e:
                logger.info(f"Failed to load config from {yaml_path}: {e}")
        else:
            logger.info(f"Cannot find {yaml_path}")

    @property
    def vocab_filename(self):
        """fairseq vocabulary file under data root"""
        return self.config.get("vocab_filename", "dict.txt")

    @property
    def shuffle(self) -> bool:
        """Shuffle dataset samples before batching"""
        return self.config.get("shuffle", False)

    @property
    def pre_tokenizer(self) -> Dict:
        """Pre-tokenizer to apply before subword tokenization. Returning
        a dictionary with `tokenizer` providing the tokenizer name and
        the other items providing the tokenizer-specific arguments.
        Tokenizers are defined in `fairseq.data.encoders.*`"""
        return self.config.get("pre_tokenizer", {"tokenizer": None})

    @property
    def bpe_tokenizer(self) -> Dict:
        """Subword tokenizer to apply after pre-tokenization. Returning
        a dictionary with `bpe` providing the tokenizer name and
        the other items providing the tokenizer-specific arguments.
        Tokenizers are defined in `fairseq.data.encoders.*`"""
        return self.config.get("bpe_tokenizer", {"bpe": None})

    @property
    def prepend_tgt_lang_tag(self) -> bool:
        """Prepend target lang ID token as the target BOS (e.g. for to-many
        multilingual setting). During inference, this requires `--prefix-size 1`
        to force BOS to be lang ID token."""
        return self.config.get("prepend_tgt_lang_tag", False)

    @property
    def input_feat_per_channel(self):
        """The dimension of input features (per audio channel)"""
        return self.config.get("input_feat_per_channel", 80)

    @property
    def input_channels(self):
        """The number of channels in the input audio"""
        return self.config.get("input_channels", 1)

    @property
    def sampling_alpha(self):
        """Hyper-parameter alpha = 1/T for temperature-based resampling.
        (alpha = 1 for no resampling)"""
        return self.config.get("sampling_alpha", 1.0)

    @property
    def use_audio_input(self):
        """Needed by the dataset loader to see if the model requires
        raw audio as inputs."""
        return self.config.get("use_audio_input", False)

    @property
    def audio_root(self):
        """Audio paths in the manifest TSV can be relative and this provides
        the root path. Set this to empty string when using absolute paths."""
        return self.config.get("audio_root", "")

    def get_feature_transforms(self, split, is_train):
        """Split-specific feature transforms. Allowing train set wildcard `_train`,
        evaluation set wildcard `_eval` and general wildcard `*` for matching."""
        from copy import deepcopy

        cfg = deepcopy(self.config)
        _cur = cfg.get("transforms", {})
        cur = _cur.get(split)
        cur = _cur.get("_train") if cur is None and is_train else cur
        cur = _cur.get("_eval") if cur is None and not is_train else cur
        cur = _cur.get("*") if cur is None else cur
        cfg["transforms"] = cur
        return cfg


def is_npy_data(data: bytes) -> bool:
    return data[0] == 147 and data[1] == 78


def is_flac_or_wav_data(data: bytes) -> bool:
    is_flac = data[0] == 102 and data[1] == 76
    is_wav = data[0] == 82 and data[1] == 73
    return is_flac or is_wav


def read_from_uncompressed_zip(file_path, offset, file_size) -> bytes:
    with open(file_path, "rb") as f:
        f.seek(offset)
        data = f.read(file_size)
    return data


def get_features_from_npy_or_audio(path):
    ext = op.splitext(op.basename(path))[1]
    if ext not in {".npy", ".flac", ".wav"}:
        raise ValueError(f'Unsupported file format for "{path}"')
    return np.load(path) if ext == ".npy" else get_fbank(path)


def get_features_or_waveform_from_uncompressed_zip(
    path, byte_offset, byte_size, need_waveform=False
):
    assert path.endswith(".zip")
    data = read_from_uncompressed_zip(path, byte_offset, byte_size)
    f = io.BytesIO(data)
    if is_npy_data(data):
        features_or_waveform = np.load(f)
    elif is_flac_or_wav_data(data):
        features_or_waveform = \
            get_waveform(f, always_2d=False)[0] if need_waveform else get_fbank(f)
    else:
        raise ValueError(f'Unknown file format for "{path}"')
    return features_or_waveform


def get_features_or_waveform(path: str, need_waveform=False):
    """Get speech features from .npy file or waveform from .wav/.flac file.
    The file may be inside an uncompressed ZIP file and is accessed via byte
    offset and length.

    Args:
        path (str): File path in the format of "<.npy/.wav/.flac path>" or
        "<zip path>:<byte offset>:<byte length>".
        need_waveform (bool): return waveform instead of features.

    Returns:
        features_or_waveform (numpy.ndarray): speech features or waveform.
    """
    _path, *extra = path.split(":")
    if not op.exists(_path):
        raise FileNotFoundError(f"File not found: {_path}")

    if len(extra) == 0:
        if need_waveform:
            return get_waveform(_path, always_2d=False)
        return get_features_from_npy_or_audio(_path)
    elif len(extra) == 2:
        extra = [int(i) for i in extra]
        features_or_waveform = get_features_or_waveform_from_uncompressed_zip(
            _path, extra[0], extra[1], need_waveform=need_waveform
        )
    else:
        raise ValueError(f"Invalid path: {path}")

    return features_or_waveform


def _collate_frames(
    frames: List[torch.Tensor], is_audio_input: bool = False
) -> torch.Tensor:
    """
    Convert a list of 2D frames into a padded 3D tensor
    Args:
        frames (list): list of 2D frames of size L[i]*f_dim. Where L[i] is
            length of i-th frame and f_dim is static dimension of features
    Returns:
        3D tensor of size len(frames)*len_max*f_dim where len_max is max of L[i]
    """
    max_len = max(frame.size(0) for frame in frames)
    if is_audio_input:
        out = frames[0].new_zeros((len(frames), max_len))
    else:
        out = frames[0].new_zeros((len(frames), max_len, frames[0].size(1)))
    for i, v in enumerate(frames):
        out[i, : v.size(0)] = v
    return out


def _collate_frames2(
    frames: List[torch.Tensor], is_audio_input: bool = True, max_len = None
) -> torch.Tensor:
    """
    Convert a list of 2D frames into a padded 3D tensor
    Args:
        frames (list): list of 2D frames of size L[i]*f_dim. Where L[i] is
            length of i-th frame and f_dim is static dimension of features
    Returns:
        3D tensor of size len(frames)*len_max*f_dim where len_max is max of L[i]
    """
    if max_len == None:
        max_len = max(frame.size(0) for frame in frames)
    if is_audio_input:
        out = frames[0].new_zeros((len(frames), max_len))
    else:
        out = frames[0].new_zeros((len(frames), max_len, frames[0].size(1)))
    for i, v in enumerate(frames):
        out[i, : min([max_len, v.size(0)])] = v[:min([max_len, v.size(0)])]
    return out


class SpeechToTextDataset(FairseqDataset):
    LANG_TAG_TEMPLATE = "<lang:{}>"

    def __init__(
        self,
        split: str,
        is_train_split: bool,
        data_cfg: S2TDataConfig,
        audio_paths: List[str],
        n_frames: List[int],
        src_texts: Optional[List[str]] = None,
        tgt_texts: Optional[List[str]] = None,
        speakers: Optional[List[str]] = None,
        src_langs: Optional[List[str]] = None,
        tgt_langs: Optional[List[str]] = None,
        ids: Optional[List[str]] = None,
        tgt_dict: Optional[Dictionary] = None,
        pre_tokenizer=None,
        bpe_tokenizer=None,
    ):
        self.split, self.is_train_split = split, is_train_split
        self.data_cfg = data_cfg
        self.audio_paths, self.n_frames = audio_paths, n_frames
        self.n_samples = len(audio_paths)
        assert len(n_frames) == self.n_samples > 0
        assert src_texts is None or len(src_texts) == self.n_samples
        assert tgt_texts is None or len(tgt_texts) == self.n_samples
        assert speakers is None or len(speakers) == self.n_samples
        assert src_langs is None or len(src_langs) == self.n_samples
        assert tgt_langs is None or len(tgt_langs) == self.n_samples
        assert ids is None or len(ids) == self.n_samples
        assert (tgt_dict is None and tgt_texts is None) or (
            tgt_dict is not None and tgt_texts is not None
        )
        self.src_texts, self.tgt_texts = src_texts, tgt_texts
        self.src_langs, self.tgt_langs = src_langs, tgt_langs
        self.tgt_dict = tgt_dict
        self.check_tgt_lang_tag()
        self.ids = ids
        self.shuffle = data_cfg.shuffle if is_train_split else False

        self.feature_transforms = CompositeAudioFeatureTransform.from_config_dict(
            self.data_cfg.get_feature_transforms(split, is_train_split)
        )

        self.pre_tokenizer = pre_tokenizer
        self.bpe_tokenizer = bpe_tokenizer

        logger.info(self.__repr__())

    def __repr__(self):
        return (
            self.__class__.__name__
            + f'(split="{self.split}", n_samples={self.n_samples}, '
            f"prepend_tgt_lang_tag={self.data_cfg.prepend_tgt_lang_tag}, "
            f"shuffle={self.shuffle}, transforms={self.feature_transforms})"
        )

    @classmethod
    def is_lang_tag(cls, token):
        pattern = cls.LANG_TAG_TEMPLATE.replace("{}", "(.*)")
        return re.match(pattern, token)

    def check_tgt_lang_tag(self):
        if self.data_cfg.prepend_tgt_lang_tag:
            assert self.tgt_langs is not None and self.tgt_dict is not None
            tgt_lang_tags = [
                self.LANG_TAG_TEMPLATE.format(t) for t in set(self.tgt_langs)
            ]
            assert all(t in self.tgt_dict for t in tgt_lang_tags)

    def tokenize_text(self, text: str):
        if self.pre_tokenizer is not None:
            text = self.pre_tokenizer.encode(text)
        if self.bpe_tokenizer is not None:
            text = self.bpe_tokenizer.encode(text)
        return text

    def __getitem__(
        self, index: int
    ) -> Tuple[int, torch.Tensor, Optional[torch.Tensor]]:
        source = get_features_or_waveform(
            self.audio_paths[index], need_waveform=self.data_cfg.use_audio_input
        )
        if self.feature_transforms is not None:
            assert not self.data_cfg.use_audio_input
            source = self.feature_transforms(source)
        source = torch.from_numpy(source).float()

        target = None
        if self.tgt_texts is not None:
            tokenized = self.tokenize_text(self.tgt_texts[index])
            target = self.tgt_dict.encode_line(
                tokenized, add_if_not_exist=False, append_eos=True
            ).long()
            if self.data_cfg.prepend_tgt_lang_tag:
                lang_tag = self.LANG_TAG_TEMPLATE.format(self.tgt_langs[index])
                lang_tag_idx = self.tgt_dict.index(lang_tag)
                target = torch.cat((torch.LongTensor([lang_tag_idx]), target), 0)
        return index, source, target

    def __len__(self):
        return self.n_samples


    def collater(self, samples: List[Tuple[int, torch.Tensor, torch.Tensor]]) -> Dict:
        if len(samples) == 0:
            return {}
        indices = torch.tensor([i for i, _, _ in samples], dtype=torch.long)
        frames = _collate_frames(
            [s for _, s, _ in samples], self.data_cfg.use_audio_input
        )
        # sort samples by descending number of frames
        n_frames = torch.tensor([s.size(0) for _, s, _ in samples], dtype=torch.long)
        n_frames, order = n_frames.sort(descending=True)
        indices = indices.index_select(0, order)
        frames = frames.index_select(0, order)

        target, target_lengths = None, None
        prev_output_tokens = None
        ntokens = None
        if self.tgt_texts is not None:
            target = fairseq_data_utils.collate_tokens(
                [t for _, _, t in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            )
            target = target.index_select(0, order)
            target_lengths = torch.tensor(
                [t.size(0) for _, _, t in samples], dtype=torch.long
            ).index_select(0, order)
            prev_output_tokens = fairseq_data_utils.collate_tokens(
                [t for _, _, t in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, order)
            ntokens = sum(t.size(0) for _, _, t in samples)

        out = {
            "id": indices,
            "net_input": {
                "src_tokens": frames,
                "src_lengths": n_frames,
                "prev_output_tokens": prev_output_tokens,
            },
            "target": target,
            "target_lengths": target_lengths,
            "ntokens": ntokens,
            "nsentences": len(samples),
        }
        return out

    def num_tokens(self, index):
        return self.n_frames[index]

    def size(self, index):
        t_len = 0
        if self.tgt_texts is not None:
            tokenized = self.tokenize_text(self.tgt_texts[index])
            t_len = len(tokenized.split(" "))
        return self.n_frames[index], t_len

    @property
    def sizes(self):
        return np.array(self.n_frames)

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return True

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        # first by descending order of # of frames then by original/random order
        order.append([-n for n in self.n_frames])
        return np.lexsort(order)

    def prefetch(self, indices):
        raise False



class ASRTokenizer():
    def __init__(self, 
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|",
        do_normalize=False,
        return_attention_mask=False,
        max_length = 16000*15.6,
        do_lower_case = False,
        ):

        self._word_delimiter_token = word_delimiter_token

        self.do_lower_case = do_lower_case
        self.return_attention_mask = return_attention_mask
        self.do_normalize = do_normalize
        self.max_length = max_length


    def __call__(self, 
        raw_speech,
        padding,
        max_length,
        pad_to_multiple_of = None,
        return_tensors = False,
    ):

        is_batched = bool(
            isinstance(raw_speech, (list, tuple))
            and (isinstance(raw_speech[0], np.ndarray) or isinstance(raw_speech[0], (tuple, list)))
        )

        # make sure input is in list format
        if is_batched and not isinstance(raw_speech[0], np.ndarray):
            raw_speech = [np.asarray(speech) for speech in raw_speech]
        elif not is_batched and not isinstance(raw_speech, np.ndarray):
            raw_speech = np.asarray(raw_speech)

        # always return batch
        if not is_batched:
            raw_speech = [raw_speech]
        
        if self.do_normalize:
            raw_speech = [(x - np.mean(x)) / np.sqrt(np.var(x) + 1e-5) for x in raw_speech]

        batchencoding = {
            'input_values':raw_speech,
        }
        padding_inputs = self.pad(batchencoding, padding = True, max_length = self.max_length)
        return padding_inputs
    
    def pad(self, encoded_inputs,
        padding = True,
        max_length = None,
        return_attention_mask = None,
        return_tensors = None,
        verbose: bool = True,):
        
        input_values = encoded_inputs['input_values']
        pad_values = []
        attention_masks = []
        for i in range(len(input_values)):
            input_array = input_values[i]
            if len(input_array) < max_length:
                attention_mask = [0] * len(input_array) + [1] * (max_length - len(input_array))
                input_array = np.concatenate([input_array, np.array([0] * (max_length - len(input_array)))])
                pad_values.append(input_array)
                attention_masks.append(attention_mask)
        encoded_inputs['tokens'] = np.array(pad_values)
        encoded_inputs['attention_masks'] = np.array(attention_masks)
        return encoded_inputs

class SpeechToTextDataset2(FairseqDataset):
    LANG_TAG_TEMPLATE = "<lang:{}>"

    def __init__(
        self,
        split = 'train',
        audio_paths = '/data/private/houbairu/audio_dataset/librispeech/',
        # n_frames = 15.6*16000,
        max_frames = int(5*16000),
        tgt_dict = None,
        audio_tokenizer =ASRTokenizer(),
        prepend_bos = True
    ):
        '''
        use tgt dict to tokenize the text output
        '''

        self.split =split
        self.audio_file_prefix = audio_paths + self.split + '/audiofiles/'
        self.text_file = audio_paths + self.split + '/' + self.split + '.fr'
        self.meta_file = audio_paths + self.split + '/alignments.meta'
        self.audio_tokenizer = audio_tokenizer
        self.tgt_dict = tgt_dict
        self.max_frames = max_frames
        self.audio_list, self.tgt_texts, self.n_frames = self.read_sound_file()
        self.pre_tokenizer = None
        self.bpe_tokenizer = None
        self.prepend_bos = prepend_bos
        if self.split == 'train':
            self.shuffle = True
        else:
            self.shuffle = False
        self.n_samples = len(self.tgt_texts)
        print(self.n_samples)
        # self.n_frames = self.audio_list ### n_frames应该是每个audio的time step数量，后续需要修正

    def read_sound_file(self,):
        audio_name_list = []
        audio_list = []
        fr_text_list = []
        max_value = 20
        audio_length_list = []

        count = 0
        with open(self.meta_file,'r',encoding = 'utf-8') as f:
            next(f)
            for line in f:
                count += 1
                if count >= max_value:
                    break

                book_id, chap_id, reader_id, _, audio_name, duration = line.strip().split('\t',5)
                audio_name_list.append(audio_name)

        count = 0
        with open(self.text_file, 'r', encoding = 'utf-8') as f:
            for line in f:
                count += 1
                if count >= max_value:
                    break
                # text = line.strip().split()
                text = line.strip()
                fr_text_list.append(text)


        for i in range(len(audio_name_list)):
            audio_name = audio_name_list[i]
            file_path = self.audio_file_prefix + audio_name + '.wav'
            audio_input, sample_rate = sf.read(file_path)
            audio_list.append(audio_input)
            audio_length_list.append(len(audio_input))
        print(len(audio_list), len(fr_text_list), len(audio_length_list))
        return audio_list, fr_text_list, np.array(audio_length_list)

    def tokenize_text(self, text: str):
        if self.pre_tokenizer is not None:
            text = self.pre_tokenizer.encode(text)
        if self.bpe_tokenizer is not None:
            text = self.bpe_tokenizer.encode(text)
        return text

    def __getitem__(
        self, index: int
    ) -> Tuple[int, torch.Tensor, Optional[torch.Tensor]]:
        source = torch.tensor(self.audio_list[index])

        target = None
        if self.tgt_texts is not None:
            tokenized = self.tokenize_text(self.tgt_texts[index])
            target = self.tgt_dict.encode_line(
                tokenized, add_if_not_exist=False, append_eos=True
            ).long()
            if self.prepend_bos:
                lang_tag_idx = self.tgt_dict.bos_index
                target = torch.cat((torch.LongTensor([lang_tag_idx]), target), 0)
        return index, source, target

    def __len__(self):
        return self.n_samples

    def collater(self, samples: List[Tuple[int, torch.Tensor, torch.Tensor]]) -> Dict:
        if len(samples) == 0:
            return {}
        indices = torch.tensor([i for i, _, _ in samples], dtype=torch.long)
        frames = _collate_frames2(
            [s for _, s, _ in samples], is_audio_input = True, max_len = self.max_frames
        )
        # sort samples by descending number of frames
        n_frames = torch.tensor([s.size(0) for _, s, _ in samples], dtype=torch.long)
        n_frames, order = n_frames.sort(descending=True)
        indices = indices.index_select(0, order)
        frames = frames.index_select(0, order)

        target, target_lengths = None, None
        prev_output_tokens = None
        ntokens = None
        if self.tgt_texts is not None:
            target = fairseq_data_utils.collate_tokens(
                [t for _, _, t in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            )
            target = target.index_select(0, order)
            target_lengths = torch.tensor(
                [t.size(0) for _, _, t in samples], dtype=torch.long
            ).index_select(0, order)
            prev_output_tokens = fairseq_data_utils.collate_tokens(
                [t for _, _, t in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, order)
            ntokens = sum(t.size(0) for _, _, t in samples)

        out = {
            "id": indices,
            "net_input": {
                "src_tokens": frames,
                "src_lengths": n_frames,
                "prev_output_tokens": prev_output_tokens,
            },
            "target": target,
            "target_lengths": target_lengths,
            "ntokens": ntokens,
            "nsentences": len(samples),
        }
        return out

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        # first by descending order of # of frames then by original/random order
        order.append([-n for n in self.n_frames])
        # print(order)
        return np.lexsort(order)
        # return order

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.n_frames[index]

    def num_tokens_vec(self, indices):
        """Return the number of tokens for a set of positions defined by indices.
        This value is used to enforce ``--max-tokens`` during batching."""
        int_indices = indices.astype(np.int32)
        return self.n_frames[indices]

    def size(self, index):
        return self.n_frames[index]
        # return (self.n_frames[index], len(self.tgt_texts[index].split()))


class SpeechToTextDatasetCreator(object):
    # mandatory columns
    KEY_ID, KEY_AUDIO, KEY_N_FRAMES = "id", "audio", "n_frames"
    KEY_TGT_TEXT = "tgt_text"
    # optional columns
    KEY_SPEAKER, KEY_SRC_TEXT = "speaker", "src_text"
    KEY_SRC_LANG, KEY_TGT_LANG = "src_lang", "tgt_lang"
    # default values
    DEFAULT_SPEAKER = DEFAULT_SRC_TEXT = DEFAULT_LANG = ""

    @classmethod
    def _from_list(
        cls,
        split_name: str,
        is_train_split,
        samples: List[List[Dict]],
        data_cfg: S2TDataConfig,
        tgt_dict,
        pre_tokenizer,
        bpe_tokenizer,
    ) -> SpeechToTextDataset:
        audio_paths, n_frames, src_texts, tgt_texts, ids = [], [], [], [], []
        speakers, src_langs, tgt_langs = [], [], []
        for s in samples:
            ids.extend([ss[cls.KEY_ID] for ss in s])
            audio_paths.extend(
                [op.join(data_cfg.audio_root, ss[cls.KEY_AUDIO]) for ss in s]
            )
            n_frames.extend([int(ss[cls.KEY_N_FRAMES]) for ss in s])
            tgt_texts.extend([ss[cls.KEY_TGT_TEXT] for ss in s])
            src_texts.extend(
                [ss.get(cls.KEY_SRC_TEXT, cls.DEFAULT_SRC_TEXT) for ss in s]
            )
            speakers.extend([ss.get(cls.KEY_SPEAKER, cls.DEFAULT_SPEAKER) for ss in s])
            src_langs.extend([ss.get(cls.KEY_SRC_LANG, cls.DEFAULT_LANG) for ss in s])
            tgt_langs.extend([ss.get(cls.KEY_TGT_LANG, cls.DEFAULT_LANG) for ss in s])
        return SpeechToTextDataset(
            split_name,
            is_train_split,
            data_cfg,
            audio_paths,
            n_frames,
            src_texts,
            tgt_texts,
            speakers,
            src_langs,
            tgt_langs,
            ids,
            tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
        )

    @classmethod
    def _get_size_ratios(cls, ids: List[str], sizes: List[int], alpha: float = 1.0):
        """Size ratios for temperature-based sampling
        (https://arxiv.org/abs/1907.05019)"""
        _sizes = np.array(sizes)
        prob = _sizes / _sizes.sum()
        smoothed_prob = prob ** alpha
        smoothed_prob = smoothed_prob / smoothed_prob.sum()
        size_ratio = (smoothed_prob * _sizes.sum()) / _sizes

        o_str = str({_i: f"{prob[i]:.3f}" for i, _i in enumerate(ids)})
        logger.info(f"original sampling probability: {o_str}")
        p_str = str({_i: f"{smoothed_prob[i]:.3f}" for i, _i in enumerate(ids)})
        logger.info(f"balanced sampling probability: {p_str}")
        sr_str = str({_id: f"{size_ratio[i]:.3f}" for i, _id in enumerate(ids)})
        logger.info(f"balanced sampling size ratio: {sr_str}")
        return size_ratio.tolist()

    @classmethod
    def from_tsv(
        cls,
        root: str,
        data_cfg: S2TDataConfig,
        splits: str,
        tgt_dict,
        pre_tokenizer,
        bpe_tokenizer,
        is_train_split: bool,
        epoch: int,
        seed: int,
    ) -> SpeechToTextDataset:
        samples = []
        _splits = splits.split(",")
        for split in _splits:
            tsv_path = op.join(root, f"{split}.tsv")
            if not op.isfile(tsv_path):
                raise FileNotFoundError(f"Dataset not found: {tsv_path}")
            with open(tsv_path) as f:
                reader = csv.DictReader(
                    f,
                    delimiter="\t",
                    quotechar=None,
                    doublequote=False,
                    lineterminator="\n",
                    quoting=csv.QUOTE_NONE,
                )
                samples.append([dict(e) for e in reader])
                assert len(samples) > 0

        datasets = [
            cls._from_list(
                name,
                is_train_split,
                [s],
                data_cfg,
                tgt_dict,
                pre_tokenizer,
                bpe_tokenizer,
            )
            for name, s in zip(_splits, samples)
        ]

        if is_train_split and len(_splits) > 1 and data_cfg.sampling_alpha != 1.0:
            # temperature-based sampling
            size_ratios = cls._get_size_ratios(
                _splits, [len(s) for s in samples], alpha=data_cfg.sampling_alpha
            )
            datasets = [
                ResamplingDataset(
                    d, size_ratio=r, seed=seed, epoch=epoch, replace=(r >= 1.0)
                )
                for d, r in zip(datasets, size_ratios)
            ]
        return ConcatDataset(datasets)
