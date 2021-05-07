import torch
import torch.nn as nn
from fairseq.models.bart import BARTModel, BARTHubInterface
from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model
from fairseq import hub_utils, file_utils,checkpoint_utils
from fairseq.dataclass.utils import (
    convert_namespace_to_omegaconf,
    overwrite_args_by_name,
)
import soundfile as sf
from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC, BartForConditionalGeneration, GPT2Model,GPT2LMHeadModel
# from ..preprocessing import data_util
from fairseq.optim import adam
from torch.optim import AdamW


def get_bart_hubs():
    return {
            "bart.base": "http://dl.fbaipublicfiles.com/fairseq/models/bart.base.tar.gz",
            "bart.large": "http://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz",
            "bart.large.mnli": "http://dl.fbaipublicfiles.com/fairseq/models/bart.large.mnli.tar.gz",
            "bart.large.cnn": "http://dl.fbaipublicfiles.com/fairseq/models/bart.large.cnn.tar.gz",
            "bart.large.xsum": "http://dl.fbaipublicfiles.com/fairseq/models/bart.large.xsum.tar.gz",
        }

def get_wav2vec2_hubs():
    return {
        'wav2vec2.base':'https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt',
        # 'wav2vec2.base':'http://dl.fbaipublicfiles.com/fairseq/models/wav2vec_small.tar.gz',

        'wav2vec2.large':'https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_new.pt'
    }

class ASRModel(nn.Module):
    def __init__(self, wav2vec2_model_type = 'wav2vec2.base',bart_model_type = 'bart.base', 
                    wav2vec2_cache_dir = '/data/private/houbairu/model_cache/wav2vec_model/',
                    bart_model_cache_dir = '/data/private/houbairu/model_cache/bart_model/',
                    word_dictionary = None,
                    wav2vec2_output_dim = 768, bart_hidden_dim = 768,
                    decode_max_length = 50,
                    device = torch.device('cuda'),
                    vocab_size = 10000,
                    ):
        '''
        word_dictionary: fairseq.data.dictionary.Dictionary
        '''
        super().__init__()
        self.wav2vec2_model_type = wav2vec2_model_type
        self.bart_model_type = bart_model_type
        self.wav2vec2_cache_dir = wav2vec2_cache_dir
        self.bart_model_cache_dir = bart_model_cache_dir

        self.vocab_size = vocab_size
        self.wav2vec2_output_dim = wav2vec2_output_dim
        self.bart_hidden_dim = bart_hidden_dim

        self.word_dictionary = word_dictionary
        self.vocab_size = vocab_size
        self.decode_max_length = decode_max_length

        self.device = device

        # self.load_bart_decoder()
        self.wav2vec_encoder = self.load_wav2vec_encoder().to(self.device)
        self.bart_decoder = self.load_bart_decoder().to(self.device)



    def load_wav2vec_encoder(self,):
        '''
        return: fairseq.models.wav2vec.wav2vec2.Wav2Vec2Model
        A bad implementation of model loading. Need further improvement
        '''
        task = None
        from fairseq import tasks
        archive_map = get_wav2vec2_hubs()
        model_url = archive_map[self.wav2vec2_model_type]
        resolved_archive_file = file_utils.cached_path(model_url, cache_dir = self.wav2vec2_cache_dir)
        # state = torch.load(resolved_archive_file)
        state = checkpoint_utils.load_checkpoint_to_cpu(resolved_archive_file)
        # print(state['model'])
        if "args" in state and state["args"] is not None:
            cfg = convert_namespace_to_omegaconf(state["args"])
        elif "cfg" in state and state["cfg"] is not None:
            cfg = state["cfg"]
        else:
            raise RuntimeError(
                f"Neither args nor cfg exist in state keys = {state.keys()}"
            )
        if task is None:
            task = tasks.setup_task(cfg.task)

        if "task_state" in state:
            task.load_state_dict(state["task_state"])
        model = task.build_model(cfg.model)
        model.load_state_dict(state_dict = state['model'],strict = True,model_cfg = cfg.model)
        return model


    def load_bart_decoder(self):
        '''
        return: fairseq.models.TransformerDecoder
        '''
        x = hub_utils.from_pretrained(
            model_name_or_path = self.bart_model_type, checkpoint_file = 'model.pt',
            data_name_or_path = '.',archive_map = get_bart_hubs(),
            bpe = 'gpt2',load_checkpoint_heads=True, sample_break_mode = 'eos', 
            cache_dir = self.bart_model_cache_dir)
        # bart_model = x['models'][0]
        bart_model = x['models'][0].decoder
        # print(bart_model)
        # print(type(bart_model))
        bart_model.output_projection = nn.Linear(self.bart_hidden_dim, self.vocab_size)
        nn.init.normal_(bart_model.output_projection.weight, mean = 0,std = 0.2)
        bart_model.dictionary = self.word_dictionary
        return bart_model
    
    def forward(self, batch_wav_input, padding_mask = None, tgt_tokens = None):
        '''
        batch_wav_input: batch_size * input_sequence_length
        padding_mask: batch_size * input_sequence_length, 0/1
        tgt_tokens: batch_size * target_sequence_length

        return: batch_size * target_sequence_length * vocab_size
        '''
        wav2vec2_output = self.wav2vec_encoder(batch_wav_input, padding_mask = padding_mask)['x']
        bart_output, _ = self.bart_decoder(prev_output_tokens = tgt_tokens,)
        return bart_output
