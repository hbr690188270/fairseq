import collections
import itertools
import os
import math
import torch
import copy
import numpy as np

# from fairseq import distributed_utils, options, progress_bar, tasks, utils
from fairseq import distributed_utils, options, progress_bar, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf

from fairseq.data import iterators
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter
from fairseq.tasks.speech_to_text import SpeechToTextTask2
from fairseq.data.encoders import gpt2_bpe

from fairseq.models.bart import BARTModel, BARTHubInterface
import edit_distance

def main(args):
    device = torch.device('cuda')    
    torch.cuda.set_device(0)

    torch.manual_seed(100)

    bpe_tokenizer = gpt2_bpe.GPT2BPE(gpt2_bpe.GPT2BPEConfig())

    # res1 = bpe_tokenizer.bpe.encode("hello world")
    # print(res1)
    # res2 = bpe_tokenizer.bpe.decode(res1)
    # print(res2)
    # print(type(res2))
    # pause = input("???")

    # Setup task, e.g., translation, language modeling, etc.
    tgt_dict_path = '/data/private/houbairu/audio_dataset/librispeech/aux_files/bart_decoder_dictionary.pkl'

    task = SpeechToTextTask2.setup_task(args, tgt_dict_path)

    if args.debug:
        debug = True
        print("debug... only read 1000 samples...")
    else:
        debug = False

    max_len = 50

    print("loading dataset ....")
    # task.load_dataset("test", max_len = max_len, debug = debug, bpe_tokenizer = bpe_tokenizer)
    task.load_dataset("test", max_len = max_len, debug = debug, bpe_tokenizer = bpe_tokenizer, max_frames = int(25*16000))

    # Build model and criterion
    tgt_dict = task.tgt_dict
    vocab_size = len(task.tgt_dict)
    print("loading model....")
    asr_model = task.build_model(args, vocab_size = len(tgt_dict))
    model_path = ['./bart_1e-4/checkpoint_best.pt']
    param_dict = torch.load(model_path[0])
    asr_model.load_state_dict(param_dict["model"])
    asr_model = asr_model.to('cuda')
    asr_model.eval()

    print("model loaded!  ")

    batch_size = args.batch_size
    beam_size = 1
    print(batch_size)
    epoch_itr = task.get_batch_iterator(
        # dataset=task.dataset("test"),
        dataset=task.dataset("test"),

        max_tokens=args.max_tokens,
        max_sentences=batch_size,
        ignore_invalid_inputs=True,
        required_batch_size_multiple=1,
        seed=args.seed,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
    )
    itr = epoch_itr.next_epoch_itr(fix_batches_to_gpus=args.fix_batches_to_gpus)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch, no_progress_bar='none',default='none'
        )

    acc = 0
    total = 0

    total_token = 0
    token_acc = 0
    for i, samples in enumerate(progress, start=epoch_itr.iterations_in_epoch):
        # bart_output = asr_model(**samples['net_input'])
        net_input = samples['net_input']
        target = samples["target"].to("cuda")
        tokens = torch.zeros(batch_size, max_len + 2).to("cuda").long().fill_(tgt_dict.eos_index)
        
        batch_wav_input = net_input['src_tokens'].float().to("cuda")
        padding_mask = net_input['pad_masks'].to("cuda")

        scores = torch.zeros(batch_size * beam_size, max_len + 1).to("cuda")
        for step in range(max_len + 1):

            wav2vec2_output = asr_model.wav2vec_encoder(source = batch_wav_input, padding_mask = padding_mask, features_only = True)
            output_hidden_states = wav2vec2_output['x'].transpose(0,1)
            padding_mask = wav2vec2_output['padding_mask']
            encode_output = {
                'encoder_out':[output_hidden_states],
                'encoder_padding_mask': [padding_mask]
            }        
            lprobs, _ = asr_model.bart_decoder(prev_output_tokens = tokens[:,:step+1],encoder_out = encode_output)
            lprobs = asr_model.get_normalized_probs(lprobs, log_probs=True, sample=None)
            lprobs = lprobs[:, -1, :]
            lprobs[:, tgt_dict.pad()] = -math.inf
            if step >= max_len:
                lprobs[:, : tgt_dict.eos()] = -math.inf
                lprobs[:, tgt_dict.eos() + 1 :] = -math.inf
            scores = scores.type_as(lprobs)
            pred = torch.argmax(lprobs, dim = -1)
            tokens[:, step] = pred
            if pred[0] == tgt_dict.eos():
                tokens[:, step+1:] = tgt_dict.pad()
                break
            
        # pred = torch.argmax(lprobs, dim = -1) 
        # mask = target.ne(tgt_dict.pad())
        # n_correct = torch.sum(
        #     pred.masked_select(mask).eq(target.masked_select(mask))
        # )
        # token_num = torch.sum(mask)
        # token_acc += n_correct
        # total_token += token_num

        # pred = pre
        # print(pred)
        eos_pos = -1
        tokens = tokens.view(-1)
        for pos in range(len(tokens)):
            if tokens[pos] == tgt_dict.eos():
                eos_pos = pos
                break
        pred = list(tokens.view(-1).detach().cpu().numpy())
        # print("valid: ", pred[:eos_pos + 1])
        pred_str = bpe_tokenizer.bpe.decode(pred[:eos_pos])
        target = list(target.view(-1).detach().cpu().numpy())
        target_str = bpe_tokenizer.bpe.decode(target[:-1])
        print("pred: ", pred_str)
        print("target: ", target_str)
        pred_list = pred_str.split()
        target_list = target_str.split()

        distance = edit_distance.edit_distance(pred_list, target_list)
        acc += distance[0]

        total += len(target_list)
    print("WER: ", acc)
    print("total: ", total)
    print("error: ", acc/total)

    print("correct: ", token_acc)
    print("total: ", total_token)
    print("accuracy: ", token_acc/total_token)



if __name__ == '__main__':
    parser = options.get_training_parser()
    parser.add_argument('--max_sentences', type = int, default = 4)
    parser.add_argument('--batch_size', type = int, default = 4)
    parser.add_argument('--update_freq', type = int, default = 1)
    parser.add_argument('--freeze_encoder', action='store_true')
    parser.add_argument('--freeze_decoder', action='store_true')

    parser.add_argument('--debug', action = 'store_true')
    parser.add_argument('--max_len', type = int, default = 0)
    parser.add_argument('--restore_file', type = str, default = "checkpoint_best.pt")

    # parser.add_argument('--lr', '--learning-rate', default=0.25,type = float)


    args = options.parse_args_and_arch(parser)


    main(args)
    # cfg = convert_namespace_to_omegaconf(args)
    # distributed_utils.call_main(cfg, main)
