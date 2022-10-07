import itertools
import re
from collections import Counter
import gensim
import numpy as np
import scipy.sparse as sp
import pickle
import os
from torch_geometric.data import Data
import jieba
import torch

jieba.set_dictionary('dict.txt.big')

w2v_dim = 300
max_len = 50

dic = {
    'non-rumor': 0,  # Non-rumor   NR
    'false': 1,  # false rumor    FR
    'unverified': 2,  # unverified tweet  UR
    'true': 3,  # debunk rumor  TR
}


def clean_str_cut(string, task='twitter'):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    if task != "weibo":
        string = re.sub(r"[^A-Za-z0-9(),!?#@\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)

    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    words = list(jieba.cut(string.strip().lower())) if task == "weibo" else string.strip().lower().split()
    return words


def read_replies(filepath, tweet_id, task, max_replies=30):
    filepath1 = filepath + "replies/" + tweet_id + ".txt"
    replies = []
    if os.path.exists(filepath1):
        with open(filepath1, 'r', encoding='utf-8') as fin:
            for line in fin:
                replies.append(clean_str_cut(line, task)[:max_len])
    return replies[:max_replies]


def read_train_dev_test(root_path, file_name, appendix, X_all_tids):
    filepath = root_path + file_name + appendix
    with open(filepath, 'r', encoding='utf-8') as fin:
        X_tid, X_content, X_replies, y_ = [], [], [], []
        for line in fin.readlines():
            tid, content, label = line.strip().split("\t")
            X_all_tids.append(tid)
            X_tid.append(tid)
            replies = read_replies(root_path, tid, file_name)
            X_replies.append(replies)
            X_content.append(clean_str_cut(content, file_name)[:max_len])
            y_.append(dic[label])
    return X_tid, X_content, X_replies, y_


def read_dataset(root_path, file_name, time_delay):
    X_all_tids = []
    X_all_uids = []

    X_train_tid, X_train_content, X_train_replies, y_train = read_train_dev_test(root_path, file_name, ".train",
                                                                                 X_all_tids)
    X_dev_tid, X_dev_content, X_dev_replies, y_dev = read_train_dev_test(root_path, file_name, ".dev", X_all_tids)
    X_test_tid, X_test_content, X_test_replies, y_test = read_train_dev_test(root_path, file_name, ".test", X_all_tids)

    with open(root_path + file_name + "_graph2.txt", 'r', encoding='utf-8') as input:
        edge_index, edges_weight = [], []
        for line in input.readlines():
            tmp = line.strip().split()
            src = tmp[0]

            ### Add padding to the graph.txt, for 240 480 720
            for dst_ids_ws in tmp[1:]:
                dst, w = dst_ids_ws.split(":")
                if (time_delay == 240 and float(w) <= 0.004140272429925889) or \
                        (time_delay == 480 and float(w) <= 0.002080472683393667) or \
                        (time_delay == 720 and float(w) <= 0.0013869625520110957):
                    # X_all_uids.append(dst)    
                    # # edge_index.append([src, dst])
                    # edge_index.append([dst, src])     
                    # # edges_weight.append(float(w))
                    # edges_weight.append(-1)   
                    continue
                else:
                    X_all_uids.append(dst)
                    # edge_index.append([src, dst])
                    edge_index.append([dst, src])
                    # edges_weight.append(float(w))
                    edges_weight.append(float(w))

    X_id = list(set(X_all_tids + X_all_uids))
    num_node = len(X_id)
    print(num_node)
    X_id_dic = {id: i + 1 for i, id in enumerate(X_id)}

    edges_list = [[X_id_dic[tup[0]], X_id_dic[tup[1]]] for tup in edge_index]
    edges_list = torch.LongTensor(edges_list).t()
    edges_weight = torch.FloatTensor(edges_weight)
    # data = Data(edge_index=edges_list, edge_weight=edges_weight)

    X_train_tid = np.array([X_id_dic[tid] for tid in X_train_tid])
    X_dev_tid = np.array([X_id_dic[tid] for tid in X_dev_tid])
    X_test_tid = np.array([X_id_dic[tid] for tid in X_test_tid])

    return X_train_tid, X_train_content, X_train_replies, y_train, \
           X_dev_tid, X_dev_content, X_dev_replies, y_dev, \
           X_test_tid, X_test_content, X_test_replies, y_test, edges_list, edges_weight


def vocab_to_word2vec(fname, vocab):
    """
    Load word2vec from Mikolov
    """
    word_vecs = {}
    model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)
    count_missing = 0
    for word in vocab:
        if model.__contains__(word):
            word_vecs[word] = model[word]
        else:
            # add unknown words by generating random word vectors
            count_missing += 1
            word_vecs[word] = np.random.uniform(-0.25, 0.25, w2v_dim)

    print(str(len(word_vecs) - count_missing) + " words found in word2vec.")
    print(str(count_missing) + " words not found, generated by random.")
    return word_vecs


def build_vocab_word2vec(sentences, w2v_path='numberbatch-en.txt'):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common() if x[1] >= 2]
    vocabulary_inv = vocabulary_inv[1:]  # don't need <PAD>
    # Mapping from word to index
    word_to_ix = {x: i + 1 for i, x in enumerate(vocabulary_inv)}

    print("embedding_weights generation.......")
    word2vec = vocab_to_word2vec(w2v_path, word_to_ix)  #
    embedding_weights = build_word_embedding_weights(word2vec, vocabulary_inv)
    return word_to_ix, embedding_weights


def build_word_embedding_weights(word_vecs, vocabulary_inv):
    """
    Get the word embedding matrix, of size(vocabulary_size, word_vector_size)
    ith row is the embedding of ith word in vocabulary
    """
    vocab_size = len(vocabulary_inv)
    embedding_weights = np.zeros(shape=(vocab_size + 1, w2v_dim), dtype='float32')
    # initialize the first row
    embedding_weights[0] = np.zeros(shape=(w2v_dim,))

    for idx in range(1, vocab_size):
        embedding_weights[idx] = word_vecs[vocabulary_inv[idx]]
    print("Embedding matrix of size " + str(np.shape(embedding_weights)))
    return embedding_weights


def build_input_data(X, word_to_ix, is_replies=False, max_replies=30):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    if not is_replies:
        X = [[0] * (max_len - len(sentence)) + [word_to_ix[word] if word in word_to_ix else 0 for word in sentence] for
             sentence in X]
    else:
        X = [[[0] * max_len] * (max_replies - len(replies)) + [
            [0] * (max_len - len(doc)) + [word_to_ix[word] if word in word_to_ix else 0 for word in doc] for doc in
            replies] for replies in X]
    return X


def w2v_feature_extract(root_path, out_path, w2v_path, time_delay=-1):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    X_train_tid_16, X_train_content_16, X_train_replies_16, y_train_16, \
    X_dev_tid_16, X_dev_content_16, X_dev_replies_16, y_dev_16, \
    X_test_tid_16, X_test_content_16, X_test_replies_16, y_test_16, edges_list_16, edges_weight_16 = read_dataset(
        './twitter16/', 'twitter16', time_delay)

    X_train_tid_15, X_train_content_15, X_train_replies_15, y_train_15, \
    X_dev_tid_15, X_dev_content_15, X_dev_replies_15, y_dev_15, \
    X_test_tid_15, X_test_content_15, X_test_replies_15, y_test_15, edges_list_15, edges_weight_15 = read_dataset(
        './twitter15/', 'twitter15', time_delay)

    print("text word2vec generation.......")
    text_data = X_train_content_16 + X_dev_content_16 + X_test_content_16 + list(
        itertools.chain(*X_train_replies_16)) + list(itertools.chain(*X_dev_replies_16)) + list(
        itertools.chain(*X_test_replies_16)) + \
                X_train_content_15 + X_dev_content_15 + X_test_content_15 + list(
        itertools.chain(*X_train_replies_15)) + list(itertools.chain(*X_dev_replies_15)) + list(
        itertools.chain(*X_test_replies_15))

    # vocabulary, word_embeddings = build_vocab_word2vec(text_data, w2v_path=w2v_path)
    # pickle.dump(vocabulary, open(out_path + "/vocab.pkl", 'wb'))
    # pickle.dump(word_embeddings, open(out_path + "/word_embeddings.pkl", 'wb'))
    vocabulary = pickle.load(open("twitter_replies_all/vocab.pkl", 'rb'))
    word_embeddings = pickle.load(open("twitter_replies_all/word_embeddings.pkl", 'rb'))
    print("Vocabulary size: " + str(len(vocabulary)))
    print("build input data.......")
    X_train_content_16 = build_input_data(X_train_content_16, vocabulary)
    X_dev_content_16 = build_input_data(X_dev_content_16, vocabulary)
    X_test_content_16 = build_input_data(X_test_content_16, vocabulary)

    X_train_replies_16 = build_input_data(X_train_replies_16, vocabulary, True)
    X_dev_replies_16 = build_input_data(X_dev_replies_16, vocabulary, True)
    X_test_replies_16 = build_input_data(X_test_replies_16, vocabulary, True)

    X_train_content_15 = build_input_data(X_train_content_15, vocabulary)
    X_dev_content_15 = build_input_data(X_dev_content_15, vocabulary)
    X_test_content_15 = build_input_data(X_test_content_15, vocabulary)

    X_train_replies_15 = build_input_data(X_train_replies_15, vocabulary, True)
    X_dev_replies_15 = build_input_data(X_dev_replies_15, vocabulary, True)
    X_test_replies_15 = build_input_data(X_test_replies_15, vocabulary, True)
    data = Data(
        edge_index=torch.concat([edges_list_16, edges_list_15], dim=1),
        edge_weight=torch.concat([edges_weight_16, edges_weight_15]))
    pickle.dump(
        [np.concatenate([X_train_tid_16, X_train_tid_15]),
         np.concatenate([X_train_content_16, X_train_content_15]),
         np.concatenate([X_train_replies_16, X_train_replies_15]),
         np.concatenate([y_train_16, y_train_15]),
         word_embeddings,
         data
         ], open(out_path + "/train.pkl", 'wb'))
    pickle.dump([
        np.concatenate([X_dev_tid_16, X_dev_tid_15]),
        np.concatenate([X_dev_content_16, X_dev_content_15]),
        np.concatenate([X_dev_replies_16, X_dev_replies_15]),
        np.concatenate([y_dev_16, y_dev_15])
    ], open(out_path + "/dev.pkl", 'wb'))
    pickle.dump([
        np.concatenate([X_test_tid_16, X_test_tid_15]),
        np.concatenate([X_test_content_16, X_test_content_15]),
        np.concatenate([X_test_replies_16, X_test_replies_15]),
        np.concatenate([y_test_16, y_test_15])], open(out_path + "/test.pkl", 'wb'))


if __name__ == "__main__":
    w2v_feature_extract('./twitter/', "./twitter_replies_all", "twitter_w2v.bin", time_delay=-1)
    # w2v_feature_extract('./twitter/', "./twitter_replies_no", "twitter_w2v.bin", time_delay=-1)
    # w2v_feature_extract('./twitter/', "./twitter_time_240", "twitter_w2v.bin", time_delay=240)
    # w2v_feature_extract('./twitter/', "./twitter_time_480", "twitter_w2v.bin", time_delay=480)
    # w2v_feature_extract('./twitter/', "./twitter_time_720", "twitter_w2v.bin", time_delay=720)
