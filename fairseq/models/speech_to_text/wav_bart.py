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
# from ..preprocessing import data_util
from fairseq.optim import adam
from torch.optim import AdamW
from .. import (
    FairseqEncoder, FairseqIncrementalDecoder, FairseqModel,
    FairseqLanguageModel, register_model, register_model_architecture,
)

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


# @register_model('ASRModel')
class ASRModel(nn.Module):
# class ASRModel(FairseqModel):
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
        # self.encoder = self.load_wav2vec_encoder().to(self.device)
        # self.decoder = self.load_bart_decoder().to(self.device)


    def set_num_updates(self, num_updates):
        # self.wav2vec_encoder.set_num_updates(num_updates)
        # self.bart_decoder.set_num_updates(num_updates)
        for m in self.modules():
            if hasattr(m, "set_num_updates") and m != self:
                m.set_num_updates(num_updates)
        self.num_updates = num_updates

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
        orig_embedding_dim = bart_model.embed_tokens.embedding_dim
        bart_model.embed_tokens = nn.Embedding(num_embeddings = self.vocab_size, embedding_dim = orig_embedding_dim,
                                                padding_idx = self.word_dictionary.pad_index, )
        nn.init.normal_(bart_model.embed_tokens.weight, mean = 0,std = 0.2)
        bart_model.dictionary = self.word_dictionary
    
        return bart_model


    def get_normalized_probs(
        self, 
        net_output,
        log_probs,
        sample = None,
    ):
        """
        Get normalized probabilities (or log probs) from a net's output.
        Pointer-generator network output is already normalized.
        """
        # probs = net_output
        # Make sure the probabilities are greater than zero when returning log
        # probabilities.
        if log_probs:
            return torch.nn.functional.log_softmax(net_output, dim = -1)
        else:
            return torch.nn.functional.softmax(net_output, dim = -1)
        return probs.clamp(1e-10, 1.0).log() if log_probs else probs
    
    def get_targets(self, sample, net_output):
        """Get targets from either the sample or the net's output."""
        return sample["target"]

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        # if self.embed_positions is None:
        #     return self.max_target_positions
        # return min(self.max_target_positions, self.embed_positions.max_positions())
        return self.decode_max_length

    def max_decoder_positions(self):
        """Maximum output length supported by the decoder."""
        # if self.embed_positions is None:
        #     return self.max_target_positions
        # return min(self.max_target_positions, self.embed_positions.max_positions())
        return self.decode_max_length



    def forward(self, **param_dict):
        '''
        batch_wav_input: batch_size * input_sequence_length
        padding_mask: batch_size * input_sequence_length, 0/1
        tgt_tokens: batch_size * target_sequence_length

        return: batch_size * target_sequence_length * vocab_size
        '''
        batch_wav_input = param_dict['src_tokens'].float()
        tgt_tokens = param_dict['prev_output_tokens']
        padding_mask = param_dict['pad_masks']
        wav2vec2_output = self.wav2vec_encoder(source = batch_wav_input, padding_mask = padding_mask, features_only = True)
        output_hidden_states = wav2vec2_output['x'].transpose(0,1)
        padding_mask = wav2vec2_output['padding_mask']
        encode_output = {
            'encoder_out':[output_hidden_states],
            'encoder_padding_mask': [padding_mask]
        }
        # print(encode_output['encoder_out'])
        # wav2vec2_output = self.wav2vec_encoder(batch_wav_input, padding_mask = padding_mask)['x']
        bart_output, _ = self.bart_decoder(prev_output_tokens = tgt_tokens,encoder_out = encode_output)
        return bart_output


    # def forward(self, batch_wav_input, padding_mask = None, tgt_tokens = None):
    #     '''
    #     batch_wav_input: batch_size * input_sequence_length
    #     padding_mask: batch_size * input_sequence_length, 0/1
    #     tgt_tokens: batch_size * target_sequence_length

    #     return: batch_size * target_sequence_length * vocab_size
    #     '''
    #     print("encode")
    #     wav2vec2_output = self.wav2vec_encoder(batch_wav_input, padding_mask = padding_mask)['x']
    #     print("decode")
    #     bart_output, _ = self.bart_decoder(prev_output_tokens = tgt_tokens,)
    #     return bart_output
