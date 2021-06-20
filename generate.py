#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

"""
Translate pre-processed data with a trained model.
"""
from ctypes import alignment
import torch
from torch import nn

from fairseq import bleu, checkpoint_utils, data, options, progress_bar, tasks, tokenizer, utils
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.sequence_generator import SequenceGenerator
from fairseq.sequence_scorer import SequenceScorer
from fairseq.tasks.multilingual_translation import MultilingualTranslationTask
from fairseq.tasks.speech_to_text import SpeechToTextTask2
from fairseq.data.encoders import gpt2_bpe
import edit_distance


def main(args):

    # model_path = ['./bart_1e-4/checkpoint_best.pt']
    # model_path = ['./bart_1e-5/checkpoint_best.pt']
    model_path = ['./enc_freeze_1e-4/checkpoint_best.pt']


    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.raw_text, \
        '--replace-unk requires a raw text dataset (--raw-text)'

    if args.max_tokens is None:
        args.max_tokens = 16000*300
    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu


    bpe_tokenizer = gpt2_bpe.GPT2BPE(gpt2_bpe.GPT2BPEConfig())

    # Load dataset splits
    # tgt_dict_path = '/data/private/houbairu/audio_dataset/librispeech/aux_files/fairseq_fr_dictionary.pkl'
    tgt_dict_path = '/data/private/houbairu/audio_dataset/librispeech/aux_files/bart_decoder_dictionary.pkl'
    task = SpeechToTextTask2.setup_task(args, tgt_dict_path)
    if args.debug:
        debug = True
    else:
        debug = False
    task.load_dataset_prev("test", bpe_tokenizer = bpe_tokenizer, debug = debug)
    # print('| {} {} {} examples'.format(args.data, "test", len(task.dataset("test"))))
    print('|{} {} examples'.format("test", len(task.dataset("test"))))

    # Set dictionaries
    tgt_dict = task.target_dictionary

    # Load ensemble
    asr_model = task.build_model(args, vocab_size = len(tgt_dict))
    param_dict = torch.load(model_path[0])
    asr_model.load_state_dict(param_dict["model"])
    asr_model = asr_model.to('cuda')
    # print(asr_model)
    asr_model.eval()
    models = [asr_model]



    itr = task.get_batch_iterator(
        dataset=task.dataset("test"),
        max_tokens=args.max_tokens,
        max_sentences=1,
        max_positions=None,
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=8,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
    ).next_epoch_itr(shuffle=False)

    # Initialize generator
    gen_timer = StopwatchMeter()
    if args.score_reference:
        translator = SequenceScorer(models, task.target_dictionary)
    else:
        translator = SequenceGenerator(
            models, task.target_dictionary, beam_size=args.beam, min_len=args.min_len,
            max_len = 100,
            normalize_scores=(not args.unnormalized),
            len_penalty=args.lenpen, unk_penalty=args.unkpen,
            temperature=args.sampling_temperature,
        )

    if use_cuda:
        translator.cuda()

    # Generate and compute BLEU score
    scorer = bleu.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())
    num_sentences = 0
    has_target = True
    align_dict = None
    acc = 0
    total = 0

    with progress_bar.build_progress_bar(args, itr) as t:
        if args.score_reference:
            translations = translator.score_batched_itr(t, cuda=use_cuda, timer=gen_timer)
        else:
            translations = translator.generate_batched_itr(
                t,
                cuda=use_cuda, timer=gen_timer, 
            )

        wps_meter = TimeMeter()
        for sample_id, src_tokens, target_tokens, hypos in translations:
            # Process input and ground truth
            has_target = target_tokens is not None
            target_tokens = target_tokens.int().cpu() if has_target else None

            # Either retrieve the original sentences or regenerate them from tokens.

            # src_str = src_dict.string(src_tokens, args.remove_bpe)
            if has_target:
                # print(target_tokens.view(-1).numpy())
                target_str = bpe_tokenizer.bpe.decode(list(target_tokens.view(-1).numpy())[:-1],)

            if not args.quiet:
                # print('S-{}\t{}'.format(sample_id, src_str))
                if has_target:
                    print('T-{}\t{}'.format(sample_id, target_str))

            # Process top predictions
            for i, hypo in enumerate(hypos[:min(len(hypos), args.nbest)]):
                # hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                #     hypo_tokens=hypo['tokens'].int().cpu(),
                #     src_str=None,
                #     alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                #     align_dict=None,
                #     tgt_dict=tgt_dict,
                #     remove_bpe=args.remove_bpe,
                # )
                hypo_tokens=hypo['tokens'].int().cpu()
                hypo_str = bpe_tokenizer.bpe.decode(list(hypo_tokens.view(-1).numpy())[:-1],)
                alignment = None
                if not args.quiet:
                    print('H-{}\t{}'.format(sample_id, hypo_str))
                    # print('H-{}\t{}\t{}'.format(sample_id, hypo['score'], hypo_str))
                    # print('P-{}\t{}'.format(
                    #     sample_id,
                    #     ' '.join(map(
                    #         lambda x: '{:.4f}'.format(x),
                    #         hypo['positional_scores'].tolist(),
                    #     ))
                    # ))

                    # if args.print_alignment:
                    #     print('A-{}\t{}'.format(
                    #         sample_id,
                    #         ' '.join(map(lambda x: str(checkpoint_utils.item(x)), alignment))
                    #     ))

                # Score only the top hypothesis
                if has_target and i == 0:
                    if align_dict is not None or args.remove_bpe is not None:
                        # Convert back to tokens for evaluation with unk replacement and/or without BPE
                        target_tokens = tokenizer.Tokenizer.tokenize(
                            target_str, tgt_dict, add_if_not_exist=True)
                    scorer.add(target_tokens, hypo_tokens)

            target_list = target_str.split()
            pred_list = hypo_str.split()
            distance = edit_distance.edit_distance(pred_list, target_list)
            acc += distance[0]

            total += len(target_list)


            wps_meter.update(src_tokens.size(0))
            t.log({'wps': round(wps_meter.avg)})
            num_sentences += 1

    print('| Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
        num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))
    if has_target:
        print('| Generate {} with beam={}: {}'.format("test", args.beam, scorer.result_string()))
    
    print("WER: ", acc)
    print("total: ", total)
    print("error: ", acc/total)

if __name__ == '__main__':
    parser = options.get_generation_parser()
    parser.add_argument('--debug', action = 'store_true')

    args = options.parse_args_and_arch(parser)

    main(args)
