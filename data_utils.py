import os
import random
import sys
from transformers import BertForMaskedLM
import torch


import json
import pickle
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
from torch.utils.data import Dataset

sys.path.append(r'D:\file of he\model\IK-GCN\utils\LAL-Parser\src_joint')

def ParseData(data_path):
    with open(data_path) as infile:
        all_data = []
        data = json.load(infile)
        for d in data:
            for aspect in d['aspects']:
                text_list = list(d['token']) # wordlist
                tok = list(d['token'])       # word token
                tok = [t.lower() for t in tok]
                tok = ' '.join(tok)          #sentence
                label = aspect['polarity']   # label

                # position
                aspect_post = [aspect['from'], aspect['to']]
                sample = {'text': tok, 'label': label, 'aspect_post': aspect_post, 'text_list': text_list}
                all_data.append(sample)

    return all_data


def build_tokenizer(fnames, max_length, data_file):
    parse = ParseData
    if os.path.exists(data_file):
        print('loading tokenizer:', data_file)
        tokenizer = pickle.load(open(data_file, 'rb'))
    else:
        tokenizer = Tokenizer.from_files(fnames=fnames, max_length=max_length, parse=parse)
        pickle.dump(tokenizer, open(data_file, 'wb'))
    return tokenizer


class Vocab(object):
    ''' vocabulary of dataset '''
    def __init__(self, vocab_list, add_pad, add_unk):
        self._vocab_dict = dict()
        self._reverse_vocab_dict = dict()
        self._length = 0
        if add_pad:
            self.pad_word = '<pad>'
            self.pad_id = self._length
            self._length += 1
            self._vocab_dict[self.pad_word] = self.pad_id
        if add_unk:
            self.unk_word = '<unk>'
            self.unk_id = self._length
            self._length += 1
            self._vocab_dict[self.unk_word] = self.unk_id
        for w in vocab_list:
            self._vocab_dict[w] = self._length
            self._length += 1
        for w, i in self._vocab_dict.items():   
            self._reverse_vocab_dict[i] = w  
    
    def word_to_id(self, word):  
        if hasattr(self, 'unk_id'):
            return self._vocab_dict.get(word, self.unk_id)
        return self._vocab_dict[word]
    
    def id_to_word(self, id_):   
        if hasattr(self, 'unk_word'):
            return self._reverse_vocab_dict.get(id_, self.unk_word)
        return self._reverse_vocab_dict[id_]
    
    def has_word(self, word):
        return word in self._vocab_dict
    
    def __len__(self):
        return self._length
    
    @staticmethod
    def load_vocab(vocab_path: str):
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)


class Tokenizer(object):
    ''' transform text to indices '''
    def __init__(self, vocab, max_length, lower, pos_char_to_int, pos_int_to_char):
        self.vocab = vocab
        self.max_length = max_length
        self.lower = lower

        self.pos_char_to_int = pos_char_to_int
        self.pos_int_to_char = pos_int_to_char
    
    @classmethod
    def from_files(cls, fnames, max_length, parse, lower=True):
        corpus = set()
        pos_char_to_int, pos_int_to_char = {}, {}
        for fname in fnames:
            for obj in parse(fname):
                text_raw = obj['text']
                if lower:
                    text_raw = text_raw.lower()
                corpus.update(Tokenizer.split_text(text_raw)) 
        return cls(vocab=Vocab(corpus, add_pad=True, add_unk=True), max_length=max_length, lower=lower, pos_char_to_int=pos_char_to_int, pos_int_to_char=pos_int_to_char)
    
    @staticmethod
    def pad_sequence(sequence, pad_id, maxlen, dtype='int64', padding='post', truncating='post'):
        x = (np.zeros(maxlen) + pad_id).astype(dtype)
        if truncating == 'pre':
            trunc = sequence[-maxlen:] 
        else:
            trunc = sequence[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == 'post':
            x[:len(trunc)] = trunc 
        else:
            x[-len(trunc):] = trunc
        return x
    
    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        words = Tokenizer.split_text(text)
        sequence = [self.vocab.word_to_id(w) for w in words] 
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence.reverse()  
        return Tokenizer.pad_sequence(sequence, pad_id=self.vocab.pad_id, maxlen=self.max_length, 
                                      padding=padding, truncating=truncating)
    
    @staticmethod
    def split_text(text):
        # for ch in ["\'s", "\'ve", "n\'t", "\'re", "\'m", "\'d", "\'ll", ",", ".", "!", "*", "/", "?", "(", ")", "\"", "-", ":"]:
        #     text = text.replace(ch, " "+ch+" ")
        return text.strip().split()


class SentenceDataset(Dataset):
    ''' PyTorch standard dataset class '''
    def __init__(self, fname, tokenizer, opt, vocab_help):

        parse = ParseData
        post_vocab, pos_vocab, dep_vocab, pol_vocab = vocab_help
        data = list()
        polarity_dict = {'positive':0, 'negative':1, 'neutral':2}
        for obj in tqdm(parse(fname), total=len(parse(fname)), desc="Training examples"):
            text = tokenizer.text_to_sequence(obj['text'])
            aspect = tokenizer.text_to_sequence(obj['aspect'])  # max_length=10
            post = [post_vocab.stoi.get(t, post_vocab.unk_index) for t in obj['post']]
            post = tokenizer.pad_sequence(post, pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post', truncating='post')
            pos = [pos_vocab.stoi.get(t, pos_vocab.unk_index) for t in obj['pos']]
            pos = tokenizer.pad_sequence(pos, pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post', truncating='post')
            deprel = [dep_vocab.stoi.get(t, dep_vocab.unk_index) for t in obj['deprel']]
            deprel = tokenizer.pad_sequence(deprel, pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post', truncating='post')
            mask = tokenizer.pad_sequence(obj['mask'], pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post', truncating='post')

            adj = np.ones(opt.max_length) * opt.pad_id
            if opt.parseadj:
                from absa_parser import headparser
                # * adj
                headp, syntree = headparser.parse_heads(obj['text'])
                adj = softmax(headp[0])
                adj = np.delete(adj, 0, axis=0)
                adj = np.delete(adj, 0, axis=1)
                adj -= np.diag(np.diag(adj))
                if not opt.direct:
                    adj = adj + adj.T
                adj = adj + np.eye(adj.shape[0])
                adj = np.pad(adj, (0, opt.max_length - adj.shape[0]), 'constant')
            
            if opt.parsehead:
                from absa_parser import headparser
                headp, syntree = headparser.parse_heads(obj['text'])
                syntree2head = [[leaf.father for leaf in tree.leaves()] for tree in syntree]
                head = tokenizer.pad_sequence(syntree2head[0], pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post', truncating='post')
            else:
                head = tokenizer.pad_sequence(obj['head'], pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post', truncating='post')
            length = obj['length']
            polarity = polarity_dict[obj['label']]
            data.append({
                'text': text, 
                'aspect': aspect, 
                'post': post,
                'pos': pos,
                'deprel': deprel,
                'head': head,
                'adj': adj,
                'mask': mask,
                'length': length,
                'polarity': polarity
            })

        self._data = data

    def __getitem__(self, index):
        return self._data[index]
    
    def __len__(self):
        return len(self._data)

def _load_wordvec(data_path, embed_dim, vocab=None):
    with open(data_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        word_vec = dict()
        if embed_dim == 200:
            for line in f:
                tokens = line.rstrip().split()
                if tokens[0] == '<pad>' or tokens[0] == '<unk>': # avoid them
                    continue
                if vocab is None or vocab.has_word(tokens[0]):
                    word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
        elif embed_dim == 300:
            for line in f:
                tokens = line.rstrip().split()
                if tokens[0] == '<pad>': # avoid them
                    continue
                elif tokens[0] == '<unk>':
                    word_vec['<unk>'] = np.random.uniform(-0.25, 0.25, 300)
                word = ''.join((tokens[:-300]))
                if vocab is None or vocab.has_word(tokens[0]):
                    word_vec[word] = np.asarray(tokens[-300:], dtype='float32')
        else:
            print("embed_dim error!!!")
            exit()
            
        return word_vec

def build_embedding_matrix(vocab, embed_dim, data_file):
    if os.path.exists(data_file):
        print('loading embedding matrix:', data_file)
        embedding_matrix = pickle.load(open(data_file, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(vocab), embed_dim))
        fname = 'glove/glove.840B.300d.txt'
        word_vec = _load_wordvec(fname, embed_dim, vocab)
        for i in range(len(vocab)):
            vec = word_vec.get(vocab.id_to_word(i))
            if vec is not None:
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(data_file, 'wb'))
    return embedding_matrix


def softmax(x):
    if len(x.shape) > 1:
        # matrix
        tmp = np.max(x, axis=1)
        x -= tmp.reshape((x.shape[0], 1))
        x = np.exp(x)
        tmp = np.sum(x, axis=1)
        x /= tmp.reshape((x.shape[0], 1))
    else:
        # vector
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp
    return x


class Tokenizer4BertGCN:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.max_seq_len = max_seq_len
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.mask_token_id = self.tokenizer.mask_token_id
    def tokenize(self, s):
        return self.tokenizer.tokenize(s)
    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)
    def convert_ids_to_tokens(self, ids):
        return self.tokenizer.convert_ids_to_tokens(ids)

# ori:原数据集
# aug:原数据集和增强数据集
# aug_pure:仅含增强数据集
class ABSAGCNData(Dataset):
    def __init__(self, fname, tokenizer, opt, istrain=False, do_augment=False):
        self.data = []
        if opt.dataset == 'laptop':
            if istrain:

                data_path = 'E:\codehe\clpt-GCN\dataset/Laptops_allennlp/{}_train_sep.npz'.format(opt.dataset)
            else:
                data_path = 'E:\codehe\clpt-GCN\dataset/Laptops_allennlp/{}_test_sep.npz'.format(opt.dataset)

        elif opt.dataset == 'twitter':
            if istrain:

                data_path = 'D:\codehe\clpt-GCN\dataset/Tweets_allennlp/{}_train_sep.npz'.format(opt.dataset)
            else:
                data_path = 'D:\codehe\clpt-GCN\dataset/Tweets_allennlp/{}_test_sep.npz'.format(opt.dataset)

        elif opt.dataset == 'restaurant':
            # if istrain:
            #
            #     data_path = 'D:\\file of he\model\IK-GCN\dataset/Restaurants_allennlp/{}_train_sep_softmax.npz'.format(opt.dataset)
            # else:
            #     data_path = 'D:\\file of he\model\IK-GCN\dataset/Restaurants_allennlp/{}_test_sep_softmax.npz'.format(opt.dataset)
            if istrain:

                data_path = 'D:\codehe\clpt-GCN\dataset/Restaurants_allennlp/{}_train_sep_softmax.npz'.format(opt.dataset)
            else:
                data_path = 'D:\codehe\clpt-GCN\dataset/Restaurants_allennlp/{}_test_sep_softmax.npz'.format(opt.dataset)

        else:
            if istrain:
                data_path = 'D:\codehe\clpt-GCN\dataset/MAMS/{}_train_sep_softmax.npz'.format(opt.dataset)
            else:
                data_path = 'D:\codehe\clpt-GCN\dataset/MAMS/{}_test_sep_softmax.npz'.format(opt.dataset)

        if not os.path.exists(data_path):
            parse = ParseData
            polarity_dict = {'positive': 0, 'negative': 1, 'neutral': 2}
            for obj in tqdm(parse(fname), total=len(parse(fname)), desc="Training examples"):
                polarity = polarity_dict[obj['label']]
                text = obj['text']
                term_start = obj['aspect_post'][0]
                term_end = obj['aspect_post'][1]
                text_list = obj['text_list']
                left, term, right = text_list[: term_start], text_list[term_start: term_end], text_list[term_end:]

                from absa_parser import headparser
                headp, syntree, tlst = headparser.parse_heads(text)
                ori_adj = softmax(headp[0])
                ori_adj = np.delete(ori_adj, 0, axis=0)
                ori_adj = np.delete(ori_adj, 0, axis=1)
                ori_adj -= np.diag(np.diag(ori_adj))
                if not opt.direct:
                    ori_adj = (ori_adj + ori_adj.T)/2
                ori_adj = ori_adj + np.eye(ori_adj.shape[0])
                assert len(text_list) == ori_adj.shape[0] == ori_adj.shape[1], '{}-{}-{}'.format(len(text_list),
                                                                                                 text_list,
                                                                                                 ori_adj.shape)

                left_tokens, term_tokens, right_tokens = [], [], []
                left_tok2ori_map, term_tok2ori_map, right_tok2ori_map = [], [], []

                for ori_i, w in enumerate(left):
                    for t in tokenizer.tokenize(w):
                        left_tokens.append(t)  # * ['expand', '##able', 'highly', 'like', '##ing']
                        left_tok2ori_map.append(ori_i)  # * [0, 0, 1, 2, 2]
                asp_start = len(left_tokens)
                offset = len(left)
                for ori_i, w in enumerate(term):
                    for t in tokenizer.tokenize(w):
                        term_tokens.append(t)
                        # term_tok2ori_map.append(ori_i)
                        term_tok2ori_map.append(ori_i + offset)
                asp_end = asp_start + len(term_tokens)
                asp_idx = list(range(asp_start, asp_end))

                offset += len(term)
                for ori_i, w in enumerate(right):
                    for t in tokenizer.tokenize(w):
                        right_tokens.append(t)
                        right_tok2ori_map.append(ori_i + offset)

                while len(left_tokens) + len(right_tokens) > tokenizer.max_seq_len - 2 * len(
                        term_tokens) - 3:  # 测试发现该循环在laptop中没有任何用途
                    if len(left_tokens) > len(right_tokens):
                        left_tokens.pop(0)
                        left_tok2ori_map.pop(0)
                    else:
                        right_tokens.pop()
                        right_tok2ori_map.pop()
                # if len(l_ori) > len(left_tokens):
                #     print(l_ori)
                #     print(left_tokens)
                # if len(r_ori) > len(right_tokens):
                #     print(r_ori)
                #     print(right_tokens)
                bert_tokens = left_tokens + term_tokens + right_tokens
                # prompt_token = left_tokens + ['[MASK]'] + right_tokens

                if istrain:
                    if polarity == 0:
                        sentiment = 'negative'
                        neg_sent_1 = 'negative'
                        neg_sent_2 = 'neutral'
                    elif polarity == 1:
                        sentiment = 'neutral'
                        neg_sent_1 = 'positive'
                        neg_sent_2 = 'neutral'
                    else:
                        sentiment = 'positive'
                        neg_sent_1 = 'positive'
                        neg_sent_2 = 'negative'
                else:
                    sentiment = 'positive'
                    neg_sent_1 = 'negative'
                    neg_sent_2 = 'neutral'

                aspect = ' '.join(term)
                auxiliary_sent_pos = 'the sentiment of  ' + aspect + ' is ' + sentiment
                mask_sent = 'the sentiment of  ' + aspect + ' is [MASK]'
                auxiliary_sent_neg_1 = 'the sentiment of  ' + aspect + ' is ' + neg_sent_1
                auxiliary_sent_neg_2 = 'the sentiment of  ' + aspect + ' is '  + neg_sent_2

                auxiliary_tokens_pos = tokenizer.tokenize(auxiliary_sent_pos)
                auxiliary_tokens_neg_1 = tokenizer.tokenize(auxiliary_sent_neg_1)
                auxiliary_tokens_neg_2 = tokenizer.tokenize(auxiliary_sent_neg_2)
                mask_tokens = tokenizer.tokenize(mask_sent)


                tok2ori_map = left_tok2ori_map + term_tok2ori_map + right_tok2ori_map
                truncate_tok_len = len(bert_tokens)
                tok_adj = np.zeros(
                    (truncate_tok_len, truncate_tok_len), dtype='float32')
                for i in range(truncate_tok_len):
                    for j in range(truncate_tok_len):
                        tok_adj[i][j] = ori_adj[tok2ori_map[i]][tok2ori_map[j]]

                context_asp_ids = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(
                    bert_tokens) + [tokenizer.sep_token_id]
                context_asp_ids_mask = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(
                    bert_tokens) + [tokenizer.sep_token_id] + tokenizer.convert_tokens_to_ids(mask_tokens) + [
                                      tokenizer.sep_token_id]

                context_asp_ids_pos = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(
                    bert_tokens) + [tokenizer.sep_token_id]+ tokenizer.convert_tokens_to_ids(auxiliary_tokens_pos) + [
                                      tokenizer.sep_token_id]

                context_asp_ids_neg_1 = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(
                    bert_tokens) + [tokenizer.sep_token_id]+ tokenizer.convert_tokens_to_ids(auxiliary_tokens_neg_1) + [
                                      tokenizer.sep_token_id]

                context_asp_ids_neg_2 = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(
                    bert_tokens) + [tokenizer.sep_token_id]+ tokenizer.convert_tokens_to_ids(auxiliary_tokens_neg_2) + [
                                      tokenizer.sep_token_id]
                context_asp_len = len(context_asp_ids)
                context_asp_len_mask = len(context_asp_ids_mask)
                paddings = [0] * (opt.max_length - context_asp_len)
                paddings_mask = [0] * (opt.max_length - context_asp_len_mask)
                context_len = len(bert_tokens)
                loc_mask = [0] * opt.max_length
                loc_mask[context_asp_len_mask - 2] = 1
                context_asp_seg_ids = [0] * (1 + context_len + 1) + [1] * (len(term) + 1)
                context_asp_seg_ids += [0] * (opt.max_length - len(context_asp_seg_ids))
                # context_asp_seg_ids_mask = [0] * (1 + context_len_mask + 1) + [1] * (len(bert_tokens) + 1) + paddings_mask
                context_asp_seg_ids_mask = [0] * (1 + context_len + 1) + [1] * (
                        len(mask_tokens) + 1) + paddings_mask
                context_asp_attention_mask = [1] * context_asp_len + paddings
                context_asp_attention_mask_mask = [1] * context_asp_len_mask + paddings_mask
                context_asp_ids += paddings
                context_asp_ids_mask += paddings_mask
                context_asp_ids_pos += paddings_mask
                context_asp_ids_neg_1 += paddings_mask
                context_asp_ids_neg_2 += paddings_mask


                src_mask = [0] + [1] * context_len + [0] * (opt.max_length - context_len - 1)
                polarity_label = tokenizer.convert_tokens_to_ids([sentiment])[0]
                label_id = [-1] * opt.max_length
                label_id[context_asp_len_mask - 2] = polarity_label
                aspect_mask = [0] + [0] * asp_start + [1] * (asp_end - asp_start) #加上辅助句子
                aspect_mask = aspect_mask + (opt.max_length - len(aspect_mask)) * [0]

                # if len(context_asp_ids) == opt.max_length:
                #     pass
                # else:print("error_1")
                #
                # if len(context_asp_ids_mask) == opt.max_length:
                #     pass
                # else:print("error_2")
                # if len(context_asp_ids_pos) == opt.max_length:
                #     pass
                # else:print("error_3")
                # if len(context_asp_ids_neg_1) == opt.max_length:
                #     pass
                # else:print("error_4")
                # if len(context_asp_ids_neg_2) == opt.max_length:
                #     pass
                # else:print("error_5")
                # if len(src_mask) == opt.max_length:
                #     pass
                # else:print("error_6")
                # if len(label_id) == opt.max_length:
                #     pass
                # else:print("error_7")
                # if len(aspect_mask) == opt.max_length:
                #     pass
                # else:print("error_8")
                # if len(context_asp_attention_mask) == opt.max_length:
                #     pass
                # else:print("error_9")
                # if len(context_asp_attention_mask_mask) == opt.max_length:
                #     pass
                # else:
                #     print("error_10")
                # if len(context_asp_seg_ids) == opt.max_length:
                #     pass
                # else:
                #     print("error_11")
                # if len(context_asp_seg_ids_mask) == opt.max_length:
                #     pass
                # else:
                #     print("error_12")





                context_asp_ids_np = np.asarray(context_asp_ids, dtype='int64')
                context_asp_ids_mask_np = np.asarray(context_asp_ids_mask, dtype='int64')
                context_asp_ids_pos_np = np.asarray(context_asp_ids_pos, dtype='int64')
                context_asp_ids_neg_1_np = np.asarray(context_asp_ids_neg_1, dtype='int64')
                context_asp_ids_neg_2_np = np.asarray(context_asp_ids_neg_2, dtype='int64')
                context_asp_seg_ids_np = np.asarray(context_asp_seg_ids, dtype='int64')
                context_asp_seg_ids_mask_np = np.asarray(context_asp_seg_ids_mask, dtype='int64')
                context_asp_attention_mask_np = np.asarray(context_asp_attention_mask, dtype='int64')
                context_asp_attention_mask_mask_np = np.asarray(context_asp_attention_mask_mask, dtype='int64')
                loc_mask = np.asarray(loc_mask, dtype='int64')
                src_mask = np.asarray(src_mask, dtype='int64')
                loc_mask = np.asarray(loc_mask, dtype='int64')
                aspect_mask = np.asarray(aspect_mask, dtype='int64')
                # pad adj
                label_id = np.asarray(label_id, dtype='int64')
                context_asp_adj_matrix = np.ones((tokenizer.max_seq_len, tokenizer.max_seq_len)).astype('float32')
                pad_adj = np.ones((context_asp_len, context_asp_len)).astype('float32')
                pad_adj[1:context_len + 1, 1:context_len + 1] = tok_adj
                context_asp_adj_matrix[:context_asp_len, :context_asp_len] = pad_adj


                data = {
                     'text': text,
                    'text_bert_indices_pos': context_asp_ids_pos_np,
                    'text_bert_indices': context_asp_ids_np,
                    'text_bert_indices_mask': context_asp_ids_mask_np,
                    'text_bert_indices_neg_1': context_asp_ids_neg_1_np,
                    'text_bert_indices_neg_2': context_asp_ids_neg_2_np,
                    'bert_segments_ids_aul': context_asp_seg_ids_mask_np,

                    'attention_mask_aul': context_asp_attention_mask_mask_np,
                    'bert_segments_ids': context_asp_seg_ids_np,
                    'attention_mask': context_asp_attention_mask_np,
                    'adj_matrix': context_asp_adj_matrix,
                    'src_mask': src_mask,
                    'loc_mask':loc_mask,
                    'aspect_mask': aspect_mask,
                    'polarity': polarity,
                    'label_id':label_id,
                }

                self.data.append(data)
            np.savez(data_path, processed_data=self.data)

        else:
            data_set = np.load(data_path, allow_pickle=True)
            self.data = data_set['processed_data']


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



def dist(x, y):
    # inner product or cos similarity
    return np.sqrt(((x - y)**2).sum())

def calculate_threshold(distance, std_strength):
    return distance.mean() + distance.std() * std_strength






