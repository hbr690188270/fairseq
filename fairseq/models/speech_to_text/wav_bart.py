from json import encoder
import pickle
from numpy.core.fromnumeric import argsort
import torch
import torch.nn as nn
from torch.nn.modules import padding
from fairseq.models import lstm, transformer
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
from fairseq.models.lstm import Embedding, LSTMDecoder
import numpy as np
import argparse

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

def base_architecture(args):
    # args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    # args.encoder_layers = getattr(args, "encoder_layers", 6)
    # args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    # args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    # args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6) ##1
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8) ## 2
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)



# @register_model('ASRModel')
class ASRModel(nn.Module):
    def __init__(self, wav2vec2_model_type = 'wav2vec2.base',bart_model_type = 'bart.base', 
                    wav2vec2_cache_dir = '/data1/private/houbairu/model_cache/wav2vec_model/',
                    bart_model_cache_dir = '/data1/private/houbairu/model_cache/bart_model/',
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

    # def load_wav2vec_encoder(self,):
    #     '''
    #     return: fairseq.models.wav2vec.wav2vec2.Wav2Vec2Model
    #     A bad implementation of model loading. Need further improvement
    #     '''
    #     task = None
    #     from fairseq import tasks
    #     archive_map = get_wav2vec2_hubs()
    #     model_url = archive_map[self.wav2vec2_model_type]
    #     resolved_archive_file = file_utils.cached_path(model_url, cache_dir = self.wav2vec2_cache_dir)
    #     # state = torch.load(resolved_archive_file)
    #     state = checkpoint_utils.load_checkpoint_to_cpu(resolved_archive_file)
    #     # print(state['model'])
    #     if "args" in state and state["args"] is not None:
    #         cfg = convert_namespace_to_omegaconf(state["args"])
    #     elif "cfg" in state and state["cfg"] is not None:
    #         cfg = state["cfg"]
    #     else:
    #         raise RuntimeError(
    #             f"Neither args nor cfg exist in state keys = {state.keys()}"
    #         )
    #     if task is None:
    #         task = tasks.setup_task(cfg.task)

    #     if "task_state" in state:
    #         task.load_state_dict(state["task_state"])
    #     model = task.build_model(cfg.model)
    #     model.load_state_dict(state_dict = state['model'],strict = True,model_cfg = cfg.model)
    #     return model

    def load_wav2vec_encoder(self,):
        x = torch.load(self.wav2vec2_cache_dir + "wav2vec_small.pt")
        cfg = x['args']
        cfg = convert_namespace_to_omegaconf(cfg)
        param_dict = x['model']
        model = Wav2Vec2Model(cfg.model)
        model.load_state_dict(param_dict)
        return model

    def load_bart_decoder(self):
        '''
        return: fairseq.models.TransformerDecoder
        '''
        bart = BARTModel.from_pretrained('/data1/private/houbairu/model_cache/bart_model/bart.base/', checkpoint_file='model.pt')
        # print(bart.model)
        # print(type(bart.model))
        bart_model = bart.model.decoder
        # print(bart_model)
        # print(type(bart_model))
        bart_model.output_projection = nn.Linear(self.bart_hidden_dim, self.vocab_size)
        nn.init.normal_(bart_model.output_projection.weight, mean = 0,std = 0.2)
        orig_embedding_dim = bart_model.embed_tokens.embedding_dim
        bart_model.embed_tokens = nn.Embedding(num_embeddings = self.vocab_size, embedding_dim = orig_embedding_dim,
                                                padding_idx = self.word_dictionary.pad_index, )
        # nn.init.normal_(bart_model.embed_tokens.weight, mean = 0,std = 0.2)
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
        # return probs.clamp(1e-10, 1.0).log() if log_probs else probs
    
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
        prev_output_tokens = param_dict['prev_output_tokens']
        padding_mask = param_dict['pad_masks']
        # wav2vec2_output = self.wav2vec_encoder(source = batch_wav_input, padding_mask = padding_mask, features_only = True)
        # output_hidden_states = wav2vec2_output['x'].transpose(0,1)
        # padding_mask = wav2vec2_output['padding_mask']
        # output_hidden_states = output_hidden_states.new_ones(output_hidden_states.size())
        # encode_output = {
        #     'encoder_out':[output_hidden_states],
        #     'encoder_padding_mask': [padding_mask]
        # }

        # bart_output, _ = self.bart_decoder(prev_output_tokens = prev_output_tokens,encoder_out = encode_output)
        bart_output, _ = self.bart_decoder(prev_output_tokens = prev_output_tokens,encoder_out = None, full_context_alignment =True)

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

class ASRModel_v2(nn.Module):
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

    def set_num_updates(self, num_updates):
        # self.wav2vec_encoder.set_num_updates(num_updates)
        # self.bart_decoder.set_num_updates(num_updates)
        for m in self.modules():
            if hasattr(m, "set_num_updates") and m != self:
                m.set_num_updates(num_updates)
        self.num_updates = num_updates

    def load_wav2vec_encoder(self,):
        x = torch.load(self.wav2vec2_cache_dir + "wav2vec_small.pt")
        cfg = x['args']
        cfg = convert_namespace_to_omegaconf(cfg)
        param_dict = x['model']
        model = Wav2Vec2Model(cfg.model)
        model.load_state_dict(param_dict)
        return model

    def load_bart_decoder(self):
        '''
        return: fairseq.models.TransformerDecoder
        '''
        bart = BARTModel.from_pretrained('/data/private/houbairu/model_cache/bart_model/bart.base/', checkpoint_file='model.pt')
        # print(bart.model)
        # print(type(bart.model))
        bart_model = bart.model.decoder
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
        # return probs.clamp(1e-10, 1.0).log() if log_probs else probs
    
    def get_targets(self, sample, net_output):
        """Get targets from either the sample or the net's output."""
        return sample["target"]

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.decode_max_length

    def max_decoder_positions(self):
        """Maximum output length supported by the decoder."""
        return self.decode_max_length



    def forward(self, **param_dict):
        '''
        batch_wav_input: batch_size * input_sequence_length
        padding_mask: batch_size * input_sequence_length, 0/1
        tgt_tokens: batch_size * target_sequence_length

        return: batch_size * target_sequence_length * vocab_size
        '''
        batch_wav_input = param_dict['src_tokens'].float().to('cuda')
        prev_output_tokens = param_dict['prev_output_tokens'].to('cuda')
        padding_mask = param_dict['pad_masks'].to('cuda')
        wav2vec2_output = self.wav2vec_encoder(source = batch_wav_input, padding_mask = padding_mask, features_only = True)
        output_hidden_states = wav2vec2_output['x'].transpose(0,1)
        padding_mask = wav2vec2_output['padding_mask']
        # output_hidden_states = output_hidden_states.new_ones(output_hidden_states.size())
        encode_output = {
            'encoder_out':[output_hidden_states],
            'encoder_padding_mask': [padding_mask]
        }

        # bart_output, _ = self.bart_decoder(prev_output_tokens = prev_output_tokens,encoder_out = encode_output)
        bart_output, _ = self.bart_decoder(prev_output_tokens = prev_output_tokens,encoder_out = encode_output)

        return bart_output

class BART_Tokenizer():
    def __init__(self, bpe, bart_dictionary,max_len = 512):
        self.bpe = bpe
        self.bart_dictionary = bart_dictionary
        self.max_positions = [max_len]

    def encode(self, sentence, add_special_tokens = True, only_add_eos = True):
        tokens = self.bpe.encode(sentence)
        if len(tokens.split(" ")) > min(self.max_positions) - 2:
            tokens = " ".join(tokens.split(" ")[: min(self.max_positions) - 2])
        if add_special_tokens:
            if only_add_eos:
                bpe_sentence = tokens + " </s>"
            else:
                bpe_sentence = "<s> " + tokens + " </s>"
        else:
            bpe_sentence = tokens
        tokens = self.bart_dictionary.encode_line(bpe_sentence, append_eos=False)
        return tokens.long()
    
    def decode(self, tokens:torch.LongTensor):
        assert tokens.dim() == 1
        tokens = tokens.cpu().numpy()
        if tokens[0] == self.bart_dictionary.bos():
            tokens = tokens[1:]  # remove <s>
        eos_mask = tokens == self.bart_dictionary.eos()
        doc_mask = eos_mask[1:] & eos_mask[:-1]
        sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)
        sentences = [
            self.bpe.decode(self.bart_dictionary.string(s)) for s in sentences
        ]
        if len(sentences) == 1:
            return sentences[0]
        return sentences        

class Self_LSTMDecoder(nn.Module):
    def __init__(self, embedding_dim = 300, hidden_dim = 768, vocab_size = 50000, max_len = 50, num_layers = 2,):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.max_len = max_len

        self.embedding = nn.Embedding(num_embeddings = self.vocab_size,embedding_dim = self.embedding_dim).to(self.device)
        self.lstm_cell = nn.LSTM(input_size = embedding_dim, hidden_size = hidden_dim,
                         num_layers = num_layers, bidirectional = True, dropout = 0.2, batch_first = True)
        self.dropout_layer = nn.Dropout(p = 0.2)
        self.dense_layer = nn.Linear(in_features = hidden_dim * 2, out_features = 2)



class ASRModel_lstm_decoder(nn.Module):
    def __init__(self, wav2vec2_model_type = 'wav2vec2.base',
                    wav2vec2_cache_dir = '/data1/private/houbairu/model_cache/wav2vec_model/',
                    word_dictionary = None,
                    wav2vec2_output_dim = 768, lstm_hidden_dim = 768,
                    decode_max_length = 50,
                    device = torch.device('cuda'),
                    vocab_size = 10000,
                    num_layers = 1
                    ):
        '''
        word_dictionary: fairseq.data.dictionary.Dictionary
        '''
        super().__init__()
        self.wav2vec2_model_type = wav2vec2_model_type
        self.wav2vec2_cache_dir = wav2vec2_cache_dir

        self.vocab_size = vocab_size
        self.wav2vec2_output_dim = wav2vec2_output_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_layers = num_layers

        self.word_dictionary = word_dictionary
        self.vocab_size = vocab_size
        self.decode_max_length = decode_max_length

        self.device = device

        # self.load_bart_decoder()
        self.wav2vec_encoder = self.load_encoder().to(self.device)
        self.bart_decoder = self.load_decoder().to(self.device)


    def set_num_updates(self, num_updates):
        # self.wav2vec_encoder.set_num_updates(num_updates)
        # self.bart_decoder.set_num_updates(num_updates)
        for m in self.modules():
            if hasattr(m, "set_num_updates") and m != self:
                m.set_num_updates(num_updates)
        self.num_updates = num_updates

    def load_encoder(self,):
        x = torch.load(self.wav2vec2_cache_dir + "wav2vec_small.pt")
        cfg = x['args']
        cfg = convert_namespace_to_omegaconf(cfg)
        param_dict = x['model']
        model = Wav2Vec2Model(cfg.model)
        model.load_state_dict(param_dict)
        return model

    def load_decoder(self, use_pretrain_embedding = False):
        if use_pretrain_embedding:
            emb_mat = np.load("/data1/private/houbairu/fairseq/embeddings_glove.npy")
            print(emb_mat.shape)
            emb_mat = nn.Embedding(num_embeddings = emb_mat.shape[1], padding_idx = self.word_dictionary.pad(), embedding_dim = emb_mat.shape[0], _weight = torch.tensor(emb_mat.T).float())

            decoder = LSTMDecoder(dictionary = self.word_dictionary, hidden_size = self.lstm_hidden_dim, 
                            attention = True, num_layers = self.num_layers, encoder_output_units = self.wav2vec2_output_dim,
                            pretrained_embed= emb_mat, embed_dim = 300)
        else:
            decoder = LSTMDecoder(dictionary = self.word_dictionary, hidden_size = self.lstm_hidden_dim, 
                            attention = True, num_layers = self.num_layers, encoder_output_units = self.wav2vec2_output_dim,
                            )            

            # decoder = LSTMDecoder(dictionary = self.word_dictionary, hidden_size = self.lstm_hidden_dim, 
            #             attention = False, num_layers = self.num_layers, encoder_output_units = 0)

        return decoder

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
        # return probs.clamp(1e-10, 1.0).log() if log_probs else probs
    
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
        prev_output_tokens = param_dict['prev_output_tokens']
        padding_mask = param_dict['pad_masks']
        wav2vec2_output = self.wav2vec_encoder(source = batch_wav_input, padding_mask = padding_mask, features_only = True)
        output_hidden_states = wav2vec2_output['x'].transpose(0,1)
        padding_mask = wav2vec2_output['padding_mask'].transpose(0,1)
        # print(padding_mask)

        bsz = prev_output_tokens.size(0)
        zero_state = output_hidden_states.new_zeros(bsz, self.lstm_hidden_dim)
        prev_hiddens = [zero_state for i in range(self.num_layers)]
        prev_cells = [zero_state for i in range(self.num_layers)]
        ones_hidden = output_hidden_states.new_ones(output_hidden_states.size())
        encode_output = [
            output_hidden_states,
            prev_hiddens,
            prev_cells,
            padding_mask,
        ]
        # encode_output = [
        #     ones_hidden,
        #     prev_hiddens,
        #     prev_cells,
        #     padding_mask,
        # ]
        # print("encoder output: ",encode_output['encoder_out'][0])
        # print("encoder output shape: ",encode_output['encoder_out'][0].size())

        bart_output, _ = self.bart_decoder(prev_output_tokens = prev_output_tokens,encoder_out = encode_output)
        # bart_output, _ = self.bart_decoder(prev_output_tokens = prev_output_tokens)

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

class ASRModel_transformer_decoder(nn.Module):
    def __init__(self, wav2vec2_model_type = 'wav2vec2.base',
                    wav2vec2_cache_dir = '/data1/private/houbairu/model_cache/wav2vec_model/',
                    word_dictionary = None,
                    wav2vec2_output_dim = 768, lstm_hidden_dim = 768,
                    decode_max_length = 50,
                    device = torch.device('cuda'),
                    vocab_size = 10000,
                    num_layers = 1,
                    args = None
                    ):
        '''
        word_dictionary: fairseq.data.dictionary.Dictionary
        '''
        super().__init__()
        self.wav2vec2_model_type = wav2vec2_model_type
        self.wav2vec2_cache_dir = wav2vec2_cache_dir

        self.vocab_size = vocab_size
        self.wav2vec2_output_dim = wav2vec2_output_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_layers = num_layers

        self.word_dictionary = word_dictionary
        self.vocab_size = vocab_size
        self.decode_max_length = decode_max_length

        self.device = device

        # self.load_bart_decoder()
        self.wav2vec_encoder = self.load_encoder().to(self.device)
        self.bart_decoder = self.load_decoder(args).to(self.device)



    def set_num_updates(self, num_updates):
        # self.wav2vec_encoder.set_num_updates(num_updates)
        # self.bart_decoder.set_num_updates(num_updates)
        for m in self.modules():
            if hasattr(m, "set_num_updates") and m != self:
                m.set_num_updates(num_updates)
        self.num_updates = num_updates

    def load_encoder(self,):
        x = torch.load(self.wav2vec2_cache_dir + "wav2vec_small.pt")
        cfg = x['args']
        cfg = convert_namespace_to_omegaconf(cfg)
        param_dict = x['model']
        model = Wav2Vec2Model(cfg.model)
        model.load_state_dict(param_dict)
        return model

    def load_decoder(self, args):

        base_architecture(args)
        vocab_num = len(self.word_dictionary)
        embed_tokens = nn.Embedding(num_embeddings = vocab_num, embedding_dim = 512, padding_idx = self.word_dictionary.pad())
        decoder = transformer.TransformerDecoder(args, dictionary = self.word_dictionary,embed_tokens = embed_tokens)
        # decoder = None
        return decoder

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
        # return probs.clamp(1e-10, 1.0).log() if log_probs else probs
    
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
        prev_output_tokens = param_dict['prev_output_tokens']
        padding_mask = param_dict['pad_masks']

        wav2vec2_output = self.wav2vec_encoder(source = batch_wav_input, padding_mask = padding_mask, features_only = True)
        output_hidden_states = wav2vec2_output['x'].transpose(0,1)
        padding_mask = wav2vec2_output['padding_mask']
        # output_hidden_states = output_hidden_states.new_ones(output_hidden_states.size())
        encode_output = {
            'encoder_out':[output_hidden_states],
            'encoder_padding_mask': [padding_mask]
        }
        bart_output, _ = self.bart_decoder(prev_output_tokens = prev_output_tokens,encoder_out = encode_output)
        # bart_output, _ = self.bart_decoder(prev_output_tokens = prev_output_tokens)

        return bart_output


