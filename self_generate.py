import collections
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
from fairseq.tasks.speech_to_text import SpeechToTextTask2
from fairseq.data.encoders import gpt2_bpe

from fairseq.models.bart import BARTModel, BARTHubInterface

def main(args):
    device = torch.device('cuda')    
    torch.cuda.set_device(0)

    torch.manual_seed(100)

    bpe_tokenizer = gpt2_bpe.GPT2BPE(gpt2_bpe.GPT2BPEConfig())

    # Setup task, e.g., translation, language modeling, etc.
    tgt_dict_path = '/data1/private/houbairu/audio_dataset/librispeech/aux_files/bart_decoder_dictionary.pkl'

    task = SpeechToTextTask2.setup_task(args, tgt_dict_path)

    if args.debug:
        debug = True
        print("debug... only read 1000 samples...")
    else:
        debug = False

    max_len = 50

    print("loading dataset ....")
    # task.load_dataset("test", max_len = max_len, debug = debug, bpe_tokenizer = bpe_tokenizer)
    task.load_dataset("test", max_len = max_len, debug = debug, bpe_tokenizer = bpe_tokenizer, max_frames = int(15.6*16000))

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
    for i, samples in enumerate(progress, start=epoch_itr.iterations_in_epoch):
        # bart_output = asr_model(**samples['net_input'])
        net_input = samples['net_input']
        target = samples["target"].to("cuda")
        tokens = torch.zeros(batch_size, max_len + 2).to("cuda").long().fill_(tgt_dict.eos_index)
        
        batch_wav_input = net_input['src_tokens'].float().to("cuda")
        padding_mask = net_input['pad_masks'].to("cuda")
        wav2vec2_output = asr_model.wav2vec_encoder(source = batch_wav_input, padding_mask = padding_mask, features_only = True)
        output_hidden_states = wav2vec2_output['x'].transpose(0,1)
        padding_mask = wav2vec2_output['padding_mask']
        encode_output = {
            'encoder_out':[output_hidden_states],
            'encoder_padding_mask': [padding_mask]
        }        
        prev_output_tokens = net_input["prev_output_tokens"].to("cuda")
        bart_output, _ = asr_model.bart_decoder(prev_output_tokens = prev_output_tokens,encoder_out = encode_output)

        pred = torch.argmax(bart_output, dim = -1) 
        # mask = target.ne(self.padding_idx)
        # n_correct = torch.sum(
        #     lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        # )
        # total = torch.sum(mask)

        pred = bpe_tokenizer.bpe.decode(pred.data)
        target = bpe_tokenizer.bep.decode(target.data)
        print("pred: ", pred)
        print("target: ", samples['target'])
        acc += torch.sum(pred.eq(target))
        total += pred.view(-1).size(0)
    print("correct: ", acc)
    print("total: ", total)
    print("accuracy: ", acc/total)



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
