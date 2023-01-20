# coding=utf-8

import os
import logging
import glob
import math
import json
import argparse
import random
from pathlib import Path
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import RandomSampler

from datasets import MyDataset
from tokenization_cmbert import CmbertTokenizer
from modeling_cmbert import CMBERT, CMBERTConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from collator import FakeAbstractCollator
from data_utils import load_stopwords
from torch.utils.data import DataLoader
ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys())
                  for conf in (CMBERTConfig,)), ())
MODEL_CLASSES = {
    'cmbert': (CMBERTConfig, CMBERT, CmbertTokenizer)
}

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_max_epoch_model(output_dir):
    fn_model_list = glob.glob(os.path.join(output_dir, "model.*.bin"))
    fn_optim_list = glob.glob(os.path.join(output_dir, "optim.*.bin"))
    if (not fn_model_list) or (not fn_optim_list):
        return None
    both_set = set([int(Path(fn).stem.split('.')[-1]) for fn in fn_model_list]
                   ) & set([int(Path(fn).stem.split('.')[-1]) for fn in fn_optim_list])
    if both_set:
        return max(both_set)
    else:
        return None


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default='./data', type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--src_file", default='train.txt', type=str,
                        help="The input data file name.")
    parser.add_argument("--self_generated", default=False,
                        help="Whether to run training.")
    parser.add_argument("--model_type", default='cmbert', type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default='pre_weight/bert-base', type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--output_dir", default='./ckpt', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--log_dir", default='./logs', type=str,
                        help="The output directory where the log will be written.")
    parser.add_argument("--model_recover_path", default=None, type=str,
                        help="The file of fine-tuned pretraining model.")
    parser.add_argument("--optim_recover_path", default=None, type=str,
                        help="The file of pretraining optimizer.")
    parser.add_argument("--config_name", default=None, type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default=None, type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")

    # Other parameters
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument('--max_position_embeddings', type=int, default=512,
                        help="max position embeddings")
    parser.add_argument("--do_train", default=True,
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size", default=2, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=2, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--label_smoothing", default=0, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="The weight decay rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float,
                        help="Dropout rate for hidden states.")
    parser.add_argument("--attention_probs_dropout_prob", default=0.1, type=float,
                        help="Dropout rate for attention probabilities.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--tokenized_input', action='store_true',
                        help="Whether the input is tokenized.")
    parser.add_argument('--max_len', default=512)
    parser.add_argument('--max_len_a', type=int, default=771,
                        help="Truncate_config: maximum l128ngth of segment A.")
    parser.add_argument('--max_len_b', type=int, default=256,
                        help="Truncate_config: maximum length of segment B.")
    parser.add_argument('--trunc_seg', default='',
                        help="Truncate_config: first truncate segment A/B (option: a, b).")
    parser.add_argument('--always_truncate_tail', action='store_true',
                        help="Truncate_config: Whether we should always truncate tail.")
    parser.add_argument("--mask_prob", default=0.20, type=float,
                        help="Number of prediction is sometimes less than max_pred when sequence is short.")
    parser.add_argument("--mask_prob_eos", default=0, type=float,
                        help="Number of prediction is sometimes less than max_pred when sequence is short.")
    parser.add_argument('--max_pred', type=int, default=256,
                        help="Max tokens of prediction.")
    parser.add_argument("--num_workers", default=0, type=int,
                        help="Number of workers for the data loader.")

    parser.add_argument('--mask_source_words', action='store_true',
                        help="Whether to mask source words for training")
    parser.add_argument('--skipgram_prb', type=float, default=0.0,
                        help='prob of ngram mask')
    parser.add_argument('--skipgram_size', type=int, default=1,
                        help='the max size of ngram mask')
    parser.add_argument('--mask_whole_word', action='store_true',
                        help="Whether masking a whole word.")

    args = parser.parse_args()

    if not(args.model_recover_path and Path(args.model_recover_path).exists()):
        args.model_recover_path = None

    args.output_dir = args.output_dir.replace(
        '[PT_OUTPUT_DIR]', os.getenv('PT_OUTPUT_DIR', ''))
    args.log_dir = args.log_dir.replace(
        '[PT_OUTPUT_DIR]', os.getenv('PT_OUTPUT_DIR', ''))

    os.makedirs(args.output_dir, exist_ok=True)
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(
        args.output_dir, 'opt.json'), 'w'), sort_keys=True, indent=2)


    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(
        args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError(
            "At least one of `do_train` or `do_eval` must be True.")



    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        max_position_embeddings=args.max_position_embeddings, label_smoothing=args.label_smoothing)
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    # data_tokenizer = WhitespaceTokenizer() if args.tokenized_input else tokenizer

    stopwords_dict = load_stopwords('baidu_stopwords.txt')
    collator = FakeAbstractCollator(args,tokenizer, stopwords_dict,
                                    args.max_seq_length,tokenizer.convert_tokens_to_ids)


    # test_dataloader=DataLoader(
    #         test_ds,
    #         batch_size=args.batch_size,
    #         num_workers=args.dataloader_workers,
    #         collate_fn=collator,
    #         pin_memory=False,
    #     )
    # val_dataloader = DataLoader(
    #     val_ds,
    #     batch_size=args.batch_size,
    #     num_workers=args.dataloader_workers,
    #     collate_fn=collator,
    #     pin_memory=False,
    # )

        # Prepare model
    recover_step = _get_max_epoch_model(args.output_dir)
    global_step = 0

    model_recover = None

    model = model_class.from_pretrained(
        args.model_name_or_path, state_dict=model_recover, config=config)

    model = model.to(device)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=10000)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level)

    if recover_step:
        logger.info("***** Recover optimizer: %d *****", recover_step)
        optim_recover = torch.load(os.path.join(
            args.output_dir, "optim.{0}.bin".format(recover_step)), map_location='cpu')
        if hasattr(optim_recover, 'state_dict'):
            optim_recover = optim_recover.state_dict()
        optimizer.load_state_dict(optim_recover)

        logger.info("***** Recover amp: %d *****", recover_step)
        model_recover = torch.load(os.path.join(
            args.output_dir, "model.{0}.bin".format(recover_step)), map_location='cpu')
        model.load_state_dict(model_recover)

        logger.info("***** Recover scheduler: %d *****", recover_step)
        scheduler_recover = torch.load(os.path.join(
            args.output_dir, "sched.{0}.bin".format(recover_step)), map_location='cpu')
        scheduler.load_state_dict(scheduler_recover)

    logger.info("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()


    logger.info("***** Running training *****")
    logger.info("  Batch size = %d", args.train_batch_size)


    model.train()
    if recover_step:
        start_epoch = recover_step+1
    else:
        start_epoch = 1
    for i_epoch in trange(start_epoch, int(args.num_train_epochs)+1, desc="Epoch"):
        # if args.do_train:
        if os.path.exists('data/prepared_text.txt'):
            print('存在无法提取伪摘要的句子，存储在data/prepared_text')
            os.remove('data/prepared_text.txt')
        if args.self_generated:
            train_ds = MyDataset(os.path.join(args.data_dir, args.src_file),os.path.join('output','output.txt'))
        # test_ds = MyDataset(args.test_file)
        # val_ds = MyDataset()
        else:
            train_ds = MyDataset(os.path.join(args.data_dir, args.src_file), None)
        train_dataloader = DataLoader(
            train_ds,
            batch_size=args.train_batch_size,
            num_workers=0,
            collate_fn=collator,
            pin_memory=False,
        )
        iter_bar = tqdm(train_dataloader, desc='Iter (loss=X.XXX)')
        for step, batch in enumerate(iter_bar):
            batch = [
                t.to(device) if t is not None else None for t in batch]
            input_ids, segment_ids, input_mask, lm_label_ids, masked_pos, masked_weights, _ = batch
            masked_lm_loss = model(input_ids, segment_ids, input_mask, lm_label_ids,
                                   masked_pos=masked_pos, masked_weights=masked_weights)
            if n_gpu > 1:    # mean() to average on multi-gpu.
                # loss = loss.mean()
                masked_lm_loss = masked_lm_loss.mean()
            loss = masked_lm_loss

            # logging for each step (i.e., before normalization by args.gradient_accumulation_steps)
            iter_bar.set_description('Iter (loss=%5.3f)' % loss.item())

            # ensure that accumlated gradients are normalized
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                optimizer.zero_grad()
                global_step += 1

        # Save a trained model
        logger.info(
            "** ** * Saving fine-tuned model and optimizer ** ** * ")
        model_to_save = model.module if hasattr(
            model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(
            args.output_dir, "model.{0}.bin".format(i_epoch))
        torch.save(model_to_save.state_dict(), output_model_file)
        output_optim_file = os.path.join(
            args.output_dir, "optim.{0}.bin".format(i_epoch))
        torch.save(optimizer.state_dict(), output_optim_file)
        if args.fp16:
            output_amp_file = os.path.join(
                args.output_dir, "amp.{0}.bin".format(i_epoch))
            torch.save(amp.state_dict(), output_amp_file)
        output_sched_file = os.path.join(
            args.output_dir, "sched.{0}.bin".format(i_epoch))
        torch.save(scheduler.state_dict(), output_sched_file)

        logger.info("***** CUDA.empty_cache() *****")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
