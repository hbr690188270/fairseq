#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Train a new model on one or across multiple GPUs.
"""

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
import logging
from fairseq.models.speech_to_text.wav_bart import BART_Tokenizer
logger = logging.getLogger()
logger.setLevel(logging.INFO) 

# log_file = "bart_1e-4"
# fh = logging.FileHandler("logs/" + log_file + ".txt", mode='w')

# log_file = "bart_1e-5"
# fh = logging.FileHandler("logs/" + log_file + ".txt", mode='w')

# log_file = "enc_freeze_1e-4"
# fh = logging.FileHandler("logs/" + log_file + ".txt", mode='w')

# log_file = "enc_freeze_1e-5"
# fh = logging.FileHandler("logs/" + log_file + ".txt", mode='w')

# log_file = "pretrain_bart_enc_freeze_1e-4"
# fh = logging.FileHandler("logs/" + log_file + ".txt", mode='w')

# log_file = "pretrain_bart_1e-4"
# fh = logging.FileHandler("logs/" + log_file + ".txt", mode='w')

# log_file = "pretrain_bart_1e-5"
# fh = logging.FileHandler("logs/" + log_file + ".txt", mode='w')


# log_file = "enc_freeze_1e-5"
# fh = logging.FileHandler("logs/" + log_file + ".txt", mode='w')


log_file = "test"
fh = logging.FileHandler("logs/" + log_file + ".txt", mode='w')

logger.addHandler(fh)


from fairseq.models.bart import BARTModel, BARTHubInterface

def main(args):
    # if args.max_tokens is None:
    # args.max_tokens = 16000*300
    print(args)

    # if not torch.cuda.is_available():
        # raise NotImplementedError('Training on CPU is not supported')
    # if not torch.cuda.is_available():
    #     device = torch.device('cpu:0')
    # else:
    #     torch.cuda.set_device(args.device_id)
    device = torch.device('cuda')    
    torch.cuda.set_device(0)

    torch.manual_seed(100)

    # bart = BARTModel.from_pretrained('/data1/private/houbairu/model_cache/bart_model/bart.base/', checkpoint_file='model.pt')
    # bpe_tokenizer = bart.bpe
    bpe_tokenizer = gpt2_bpe.GPT2BPE(gpt2_bpe.GPT2BPEConfig())

    # Setup task, e.g., translation, language modeling, etc.
    # tgt_dict_path = '/data1/private/houbairu/audio_dataset/librispeech/aux_files/fairseq_fr_dictionary.pkl'
    # tgt_dict_path = '/data1/private/houbairu/audio_dataset/librispeech/aux_files/fairseq_en_dictionary.pkl'
    tgt_dict_path = '/data1/private/houbairu/audio_dataset/librispeech/aux_files/bart_decoder_dictionary.pkl'

    task = SpeechToTextTask2.setup_task(args, tgt_dict_path)
    tgt_dict = task.tgt_dict
    vocab_size = len(task.tgt_dict)

    bart_tokenizer = BART_Tokenizer(bpe_tokenizer, tgt_dict)

    # Load dataset splits
    if args.debug:
        debug = True
        print("debug... only read 1000 samples...")
    else:
        debug = False

    if args.max_len == 0:
        max_len = None
        print("do not set max length padding...")
    else:
        max_len = args.max_len
        print("set max length %d ..."%(max_len))

    print("loading dataset ....")
    logger.info("loading dataset ....")
    load_dataset_splits(task, ['train', 'dev', 'test'],max_len, debug, bart_tokenizer)
    print("dataset loaded! ")
    logger.info("dataset loaded! ")
    # Build model and criterion

    print("loading model....")
    logger.info("loading model....")
    model = task.build_model(args, vocab_size = vocab_size)
    print("vocab size: ", model.bart_decoder.embed_tokens.weight.size())

    print("model loaded!  ")
    logger.info("model loaded!  ")
    criterion = task.build_criterion(args)
    print('|criterion {}'.format(criterion.__class__.__name__))
    print('| num. model params: {}'.format(sum(p.numel() for p in model.parameters())))

    # if args['freeze_encoder']:

    if args.freeze_encoder:
        logger.info("freeze encoder...")
        print("freeze encoder parameters...")
        for p in model.wav2vec_encoder.parameters():
            p.requires_grad = False

    if args.freeze_decoder:
        logger.info("freeze decoder...")
        print("freeze encoder parameters...")
        for p in model.bart_decoder.parameters():
            p.requires_grad = False

    max_positions = None


    trainer = Trainer(args, task, model, criterion)
    trainer.consolidate_optimizer()

    print(trainer._optimizer._optimizer)
    print('| training on {} GPUs'.format(args.distributed_world_size))
    print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.batch_size,
    ))

    # Initialize dataloader
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = 5
    print(batch_size)
    logger.info("batch size: %d"%(batch_size))
    epoch_itr = task.get_batch_iterator(
        dataset=task.dataset(args.train_subset),
        max_tokens=args.max_tokens,
        max_sentences=batch_size,
        max_positions=max_positions,
        ignore_invalid_inputs=True,
        required_batch_size_multiple=1,
        seed=args.seed,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
    )

    # Load the latest checkpoint if one is available
    # load_checkpoint(args, trainer, epoch_itr)
    # if not load_checkpoint(args, trainer, epoch_itr):
    #     trainer.dummy_train_step([dummy_batch])

    #Freeze encoder weights if requested

    # Train until the learning rate gets too small
    # max_epoch = args.max_epoch or math.inf
    # max_update = args.max_update or math.inf
    max_epoch = 20
    # max_update = math.inf

    lr = trainer.get_lr()

    print("learning rate: ", lr)
    logger.info("learning rate: ")
    logger.info(lr)
    train_meter = StopwatchMeter()
    train_meter.start()
    valid_losses = [None]
    valid_subsets = args.valid_subset.split(',')
    acc_list = []
    loss_list = []
    # while lr > args.min_lr and epoch_itr.epoch < max_epoch and trainer.get_num_updates() < max_update:
    while epoch_itr.epoch < max_epoch:

        # train for one epoch
        print("epoch: ", epoch_itr.epoch)
        logger.info("epoch: ")
        logger.info(epoch_itr.epoch)
        print("lr: ", [x['lr'] for x in trainer.optimizer.param_groups])
        acc, loss = train(args, trainer, task, epoch_itr)
        acc_list.append(acc)
        loss_list.append(loss)

        if epoch_itr.epoch > 15:
            trainer.optimizer.param_groups[0]['lr'] = 1e-6
            trainer.optimizer.param_groups[1]['lr'] = 1e-5

        # if epoch_itr.epoch > 100:
        #     trainer.optimizer.param_groups[0]['lr'] = 1e-8
        #     trainer.optimizer.param_groups[1]['lr'] = 5e-7

        # if epoch_itr.epoch > 150:
        #     trainer.optimizer.param_groups[0]['lr'] = 1e-8
        #     trainer.optimizer.param_groups[1]['lr'] = 1e-7

        # if epoch_itr.epoch > 200:
        #     trainer.optimizer.param_groups[0]['lr'] = 1e-9
        #     trainer.optimizer.param_groups[1]['lr'] = 1e-8


        if epoch_itr.epoch % args.validate_interval == 0:
            print("validating...")
            logger.info("validating...")
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
            logger.info("valid loss: ")
            logger.info(valid_losses)
            print("valid loss: ", valid_losses)

        # save checkpoint
        if epoch_itr.epoch % 2 == 0:
            save_checkpoint(args, trainer, epoch_itr, valid_losses[0])
        print()
        logger.info("")
    train_meter.stop()
    print('| done training in {:.1f} seconds'.format(train_meter.sum))
    logger.info('| done training in {:.1f} seconds'.format(train_meter.sum))
    with open(log_file + "_acc.txt",'w', encoding = 'utf-8') as f:
        f.write(' '.join([str(x) for x in acc_list]))

    with open(log_file + "_loss.txt",'w', encoding = 'utf-8') as f:
        f.write(' '.join([str(x) for x in loss_list]))


def train(args, trainer, task, epoch_itr,):
    """Train the model for one epoch."""

    # Update parameters every N batches
    # if epoch_itr.epoch <= len(args.update_freq):
    #     update_freq = args.update_freq[epoch_itr.epoch - 1]
    # else:
    #     update_freq = args.update_freq[-1]
    update_freq = args.update_freq[0]
    # update_freq = 1
    # print("update freq: ", update_freq)
    # logger.info("update freq: ")
    # logger.info(update_freq)
    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(fix_batches_to_gpus=args.fix_batches_to_gpus)
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch, no_progress_bar='simple',
    )
    # progress = progress_bar.build_progress_bar(
        # args, itr, epoch_itr.epoch, no_progress_bar='none',default='none'
    # )

    extra_meters = collections.defaultdict(lambda: AverageMeter())
    first_valid = args.valid_subset.split(',')[0]
    max_update = args.max_update or math.inf

    epoch_acc = 0
    epoch_loss = 0
    total_tokens = 0
    total_samples = len(progress)
    print("total sample: ", total_samples)
    logger.info("total sample: ")
    logger.info(total_samples)
    for i, samples in enumerate(progress, start=epoch_itr.iterations_in_epoch):
        # if i % 1000 == 0:
        #     msg = "%d/%d"%(i, total_samples)
        #     logger.info(msg)
        log_output = trainer.train_step(samples)
        if log_output is None:
            continue
        acc = log_output['acc']
        nll_loss = log_output['nll_loss']
        valid_token_num = log_output['valid_token_num']
        epoch_acc += acc * valid_token_num
        total_tokens += valid_token_num
        epoch_loss += nll_loss
        if valid_token_num == 0:
            print(samples['src_tokens'])
        # print("train loss: ", log_output['nll_loss'])
        # print("train loss: ", log_output['loss'])

        # log mid-epoch stats
        stats = get_training_stats(trainer)
        # print(log_output)
        for k, v in log_output.items():
            if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                continue  # these are already logged above
            if 'loss' in k:
                extra_meters[k].update(v, log_output['sample_size'])
            else:
                extra_meters[k].update(v)
            stats[k] = extra_meters[k].avg
        progress.log(stats)

        # ignore the first mini-batch in words-per-second calculation
        if i == 0:
            trainer.get_meter('wps').reset()

        # num_updates = trainer.get_num_updates()
        # if args.save_interval_updates > 0 and num_updates % args.save_interval_updates == 0 and num_updates > 0:
        #     valid_losses = validate(args, trainer, task, epoch_itr, [first_valid])
        #     save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        # if num_updates >= max_update:
        #     break
    

    epoch_acc /= total_tokens
    epoch_loss /= len(progress)
    print("train acc: ", epoch_acc)
    print("train loss: ", epoch_loss)

    logger.info("train acc: ")
    logger.info(epoch_acc)
    logger.info("train loss: ")
    logger.info(epoch_loss)
    # log end-of-epoch stats
    stats = get_training_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
    progress.print(stats)

    # reset training meters
    for k in [
        'train_loss', 'train_nll_loss', 'wps', 'ups', 'wpb', 'bsz', 'gnorm', 'clip',
    ]:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()
    return epoch_acc, epoch_loss


def get_training_stats(trainer):
    stats = collections.OrderedDict()
    stats['loss'] = '{:.3f}'.format(trainer.get_meter('train_loss').avg)
    if trainer.get_meter('train_nll_loss').count > 0:
        nll_loss = trainer.get_meter('train_nll_loss').avg
        stats['nll_loss'] = '{:.3f}'.format(nll_loss)
    else:
        nll_loss = trainer.get_meter('train_loss').avg
    stats['ppl'] = get_perplexity(nll_loss)
    stats['wps'] = round(trainer.get_meter('wps').avg)
    stats['ups'] = '{:.1f}'.format(trainer.get_meter('ups').avg)
    stats['wpb'] = round(trainer.get_meter('wpb').avg)
    stats['bsz'] = round(trainer.get_meter('bsz').avg)
    stats['num_updates'] = trainer.get_num_updates()
    stats['lr'] = trainer.get_lr()
    stats['gnorm'] = '{:.3f}'.format(trainer.get_meter('gnorm').avg)
    stats['clip'] = '{:.0%}'.format(trainer.get_meter('clip').avg)
    stats['oom'] = trainer.get_meter('oom').avg
    if trainer.get_meter('loss_scale') is not None:
        stats['loss_scale'] = '{:.3f}'.format(trainer.get_meter('loss_scale').avg)
    stats['wall'] = round(trainer.get_meter('wall').elapsed_time)
    stats['train_wall'] = round(trainer.get_meter('train_wall').sum)
    return stats


def validate(args, trainer, task, epoch_itr, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""
    valid_losses = []
    epoch_acc = 0
    total_tokens = 0
    for subset in subsets:
        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=task.dataset(subset),
            max_tokens=args.max_tokens,
            max_sentences=5,
            max_positions=None,
            ignore_invalid_inputs=True,
            required_batch_size_multiple=1,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.build_progress_bar(
            args, itr, epoch_itr.epoch,
            prefix='valid on \'{}\' subset'.format(subset),
            no_progress_bar='none',default='none'
        )

        # reset validation loss meters
        for k in ['valid_loss', 'valid_nll_loss']:
            meter = trainer.get_meter(k)
            if meter is not None:
                meter.reset()
        extra_meters = collections.defaultdict(lambda: AverageMeter())

        for sample in progress:
            log_output = trainer.valid_step(sample)
            acc = log_output['acc']
            valid_token_num = log_output['valid_token_num']
            epoch_acc += acc * valid_token_num
            total_tokens += valid_token_num
            for k, v in log_output.items():
                if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                    continue
                if 'loss' in k:
                    extra_meters[k].update(v, log_output['sample_size'])
                else:
                    extra_meters[k].update(v)

        # log validation stats
        stats = get_valid_stats(trainer)
        for k, meter in extra_meters.items():
            stats[k] = meter.avg
        progress.print(stats)

        valid_losses.append(stats['valid_loss'])
    
    
    epoch_acc /= total_tokens
    print("valid acc: ", epoch_acc)
    logger.info("valid acc: ")
    logger.info(epoch_acc)
    return valid_losses


def get_valid_stats(trainer):
    stats = collections.OrderedDict()
    stats['valid_loss'] = trainer.get_meter('valid_loss').avg
    if trainer.get_meter('valid_nll_loss').count > 0:
        nll_loss = trainer.get_meter('valid_nll_loss').avg
        stats['valid_nll_loss'] = nll_loss
    else:
        nll_loss = trainer.get_meter('valid_loss').avg
    stats['valid_ppl'] = get_perplexity(nll_loss)
    stats['num_updates'] = trainer.get_num_updates()
    if hasattr(save_checkpoint, 'best'):
        stats['best'] = min(save_checkpoint.best, stats['valid_loss'])
    return stats


def get_perplexity(loss):
    try:
        return '{:.2f}'.format(math.pow(2, loss))
    except OverflowError:
        return float('inf')


def save_checkpoint(args, trainer, epoch_itr, val_loss):
    if args.no_save or not distributed_utils.is_master(args):
        return
    epoch = epoch_itr.epoch
    end_of_epoch = epoch_itr.end_of_epoch()
    updates = trainer.get_num_updates()

    checkpoint_conds = collections.OrderedDict()
    checkpoint_conds['checkpoint{}.pt'.format(epoch)] = (
            end_of_epoch and not args.no_epoch_checkpoints and
            epoch % args.save_interval == 0
    )
    checkpoint_conds['checkpoint_{}_{}.pt'.format(epoch, updates)] = (
            not end_of_epoch and args.save_interval_updates > 0 and
            updates % args.save_interval_updates == 0
    )
    checkpoint_conds['checkpoint_best.pt'] = (
            val_loss is not None and
            (not hasattr(save_checkpoint, 'best') or val_loss < save_checkpoint.best)
    )
    checkpoint_conds['checkpoint_last.pt'] = True  # keep this last so that it's a symlink

    prev_best = getattr(save_checkpoint, 'best', val_loss)
    if val_loss is not None:
        save_checkpoint.best = min(val_loss, prev_best)
    extra_state = {
        'train_iterator': epoch_itr.state_dict(),
        'val_loss': val_loss,
    }
    if hasattr(save_checkpoint, 'best'):
        extra_state.update({'best': save_checkpoint.best})

    checkpoints = [os.path.join(args.save_dir, fn) for fn, cond in checkpoint_conds.items() if cond]
    if len(checkpoints) > 0:
        for cp in checkpoints:
            trainer.save_checkpoint(cp, extra_state)

    if not end_of_epoch and args.keep_interval_updates > 0:
        # remove old checkpoints; checkpoints are sorted in descending order
        checkpoints = utils.checkpoint_paths(args.save_dir, pattern=r'checkpoint_\d+_(\d+)\.pt')
        for old_chk in checkpoints[args.keep_interval_updates:]:
            os.remove(old_chk)


def load_checkpoint(args, trainer, epoch_itr):
    """Load a checkpoint and replay dataloader to match."""
    # os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, args.restore_file)
    if os.path.isfile(checkpoint_path):
        extra_state = trainer.load_checkpoint(checkpoint_path, args.reset_optimizer, args.reset_lr_scheduler,
                                              )
        if extra_state is not None:
            # replay train iterator to match checkpoint
            epoch_itr.load_state_dict(extra_state['train_iterator'])

            print('| loaded checkpoint {} (epoch {} @ {} updates)'.format(
                checkpoint_path, epoch_itr.epoch, trainer.get_num_updates()))

            trainer.lr_step(epoch_itr.epoch)
            trainer.lr_step_update(trainer.get_num_updates())
            if 'best' in extra_state:
                save_checkpoint.best = extra_state['best']
        return True
    return False


def load_dataset_splits(task, splits,max_len = None, debug = False, bpe_tokenizer = None):
    for split in splits:
        task.load_dataset(split, max_len = max_len, debug = debug, bpe_tokenizer = bpe_tokenizer)




if __name__ == '__main__':
    parser = options.get_training_parser()
    # group = parser.add_argument_group("other")
    # group.add_argument('--max_sentences', type = int, default = 4)
    # group.add_argument('--batch_size', type = int, default = 4)
    # group.add_argument('--update_freq', type = int, default = 1)
    # group.add_argument('--freeze_encoder', action='store_true')
    # group.add_argument('--debug', action = 'store_true')
    # group.add_argument('--max_len', type = int, default = 0)
    # group.add_argument('--restore_file', type = str, default = "checkpoint_best.pt")

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
