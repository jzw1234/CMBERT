# -*- coding=utf-8 -*-
from data_utils import (
    get_input_mask, pseudo_summary_f1, shift_tokens_right,
    padding_to_maxlength, load_stopwords, text_segmentate)
import sys
import torch
from random import randint, shuffle, choice
from random import random as rand

def batch_list_to_batch_tensors(batch):
    batch_tensors = []
    for x in zip(*batch):
        if x[0] is None:
            batch_tensors.append(None)
        elif isinstance(x[0], torch.Tensor):
            batch_tensors.append(torch.stack(x))
        else:
            try:
                batch_tensors.append(torch.tensor(x, dtype=torch.long))
            except:
                batch_tensors.append(None)
    return batch_tensors
def truncate_tokens_pair(tokens_a, tokens_b, max_len):
    if len(tokens_a) + len(tokens_b) > max_len-3:
        while len(tokens_a) + len(tokens_b) > max_len-3:
            if len(tokens_a) > len(tokens_b):
                tokens_a = tokens_a[:-1]
            else:
                tokens_b = tokens_b[:-1]
    return tokens_a, tokens_b
def get_random_word(vocab_words):
    i = randint(0, len(vocab_words)-1)
    return vocab_words[i]
class FakeAbstractCollator:

    def __init__(self,args, tokenizer, stopwords_dict, max_enc_length,indexer):
        self.args = args
        self.tokenizer = tokenizer
        self.max_seq_length = max_enc_length
        self.stopwords_dict = stopwords_dict
        self.vocab_words= list(self.tokenizer.vocab.keys())
        self._tril_matrix = torch.tril(torch.ones(
            (args.max_len, args.max_len), dtype=torch.long))

    def __call__(self, samples):
        # print("samples: ", samples)
        input_ids_all=[]
        segment_ids_all=[]
        input_mask_all=[]
        masked_ids_all=[]
        masked_pos_all=[]
        masked_weights_all=[]

        for text in samples:
            # print(text)
            text = text.split('：')
            if len(text)==1:
                text = text_segmentate(text[0])
                if len(text)==1:
                    text = text[0].split('\t')
            source, target = pseudo_summary_f1(
                text, self.stopwords_dict, self.tokenizer,
                "rouge-l")

            next_sentence_label = None
            tokens_a, tokens_b = source,target#instance[:2]
            tokens_a = self.tokenizer.tokenize(tokens_a)
            tokens_b = self.tokenizer.tokenize(tokens_b)
            # -3  for special tokens [CLS], [SEP], [SEP]
            tokens_a, tokens_b = truncate_tokens_pair(tokens_a, tokens_b, self.args.max_len)
            # Add Special Tokens
            tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
            segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
            # For masked Language Models
            # the number of prediction is sometimes less than max_pred when sequence is short
            effective_length = len(tokens_b)
            if self.args.mask_source_words:
                effective_length += len(tokens_a)
            n_pred = min(self.args.max_pred, max(1, int(round(effective_length * self.args.mask_prob))))
            # candidate positions of masked tokens
            cand_pos = []
            special_pos = set()
            for i, tk in enumerate(tokens):
                # only mask tokens_b (target sequence)
                # we will mask [SEP] as an ending symbol
                if (i >= len(tokens_a) + 2) and (tk != '[CLS]'):
                    cand_pos.append(i)
                elif self.args.mask_source_words and (i < len(tokens_a) + 2) and (tk != '[CLS]') and (
                not tk.startswith('[SEP')):
                    cand_pos.append(i)
                else:
                    special_pos.add(i)
            shuffle(cand_pos)

            masked_pos = set()
            max_cand_pos = max(cand_pos)
            for pos in cand_pos:
                if len(masked_pos) >= n_pred:
                    break
                if pos in masked_pos:
                    continue

                def _expand_whole_word(st, end):
                    new_st, new_end = st, end
                    while (new_st >= 0) and tokens[new_st].startswith('##'):
                        new_st -= 1
                    while (new_end < len(tokens)) and tokens[new_end].startswith('##'):
                        new_end += 1
                    return new_st, new_end

                if (self.args.skipgram_prb > 0) and (self.args.skipgram_size >= 2) and (rand() < self.args.skipgram_prb):
                    # ngram
                    cur_skipgram_size = randint(2, self.args.skipgram_size)
                    if self.args.mask_whole_word:
                        st_pos, end_pos = _expand_whole_word(
                            pos, pos + cur_skipgram_size)
                    else:
                        st_pos, end_pos = pos, pos + cur_skipgram_size
                else:
                    # directly mask
                    if self.args.mask_whole_word:
                        st_pos, end_pos = _expand_whole_word(pos, pos + 1)
                    else:
                        st_pos, end_pos = pos, pos + 1

                for mp in range(st_pos, end_pos):
                    if (0 < mp <= max_cand_pos) and (mp not in special_pos):
                        masked_pos.add(mp)
                    else:
                        break

            masked_pos = list(masked_pos)
            if len(masked_pos) > n_pred:
                shuffle(masked_pos)
                masked_pos = masked_pos[:n_pred]

            masked_tokens = [tokens[pos] for pos in masked_pos]
            for pos in masked_pos:
                if rand() < 0.8:  # 80%
                    tokens[pos] = '[MASK]'
                elif rand() < 0.5:  # 10%
                    tokens[pos] = get_random_word(self.vocab_words)
            # when n_pred < max_pred, we only calculate loss within n_pred
            masked_weights = [1] * len(masked_tokens)

            # Token Indexing[tokenizer._convert_id_to_token(token) for token in ids]
            masked_ids = [self.tokenizer._convert_token_to_id(token) for token in masked_tokens]#self.indexer(torch.tensor(masked_tokens))
            # Token Indexing
            input_ids = [self.tokenizer._convert_token_to_id(token) for token in tokens]#self.indexer(tokens)

            # Zero Padding
            n_pad = self.args.max_len - len(input_ids)
            input_ids.extend([0] * n_pad)
            input_ids_all.append(torch.tensor(input_ids))

            segment_ids.extend([0] * n_pad)
            segment_ids_all.append(torch.tensor(segment_ids))

            input_mask = torch.zeros(self.args.max_len, self.args.max_len, dtype=torch.long)
            input_mask[:, :len(tokens_a) + 2].fill_(1)
            second_st, second_end = len(
                tokens_a) + 2, len(tokens_a) + len(tokens_b) + 3
            input_mask[second_st:second_end, second_st:second_end].copy_(
                self._tril_matrix[:second_end - second_st, :second_end - second_st])
            input_mask_all.append(input_mask)
            # Zero Padding for masked target
            if self.args.max_pred > n_pred:
                n_pad = self.args.max_pred - n_pred
                if masked_ids is not None:
                    masked_ids.extend([0] * n_pad)
                    masked_ids_all.append(torch.tensor(masked_ids))
                if masked_pos is not None:
                    masked_pos.extend([0] * n_pad)
                    masked_pos_all.append(torch.tensor(masked_pos))
                if masked_weights is not None:
                    masked_weights.extend([0] * n_pad)
                    masked_weights_all.append(torch.tensor(masked_weights))

        return (torch.stack(input_ids_all), torch.stack(segment_ids_all), torch.stack(input_mask_all), torch.stack(masked_ids_all), torch.stack(masked_pos_all), torch.stack(masked_weights_all) , next_sentence_label)
