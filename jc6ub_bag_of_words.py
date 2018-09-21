import nltk
import numpy as np
import collections


def bag_of_word(vocabulary_size, input_texts, word_to_idx_dict):
    trn_data_list = []
    for i in range(len(input_texts)):
        temp_vec = np.zeros(vocabulary_size)
        words = nltk.wordpunct_tokenize(input_texts[i])
        for word in words:
            try:
                temp_idx = word_to_idx_dict[word]
                temp_vec[temp_idx] += 1
            except KeyError:
                pass
        trn_data_list.append(temp_vec)
    return trn_data_list

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def load_data(frequency_threshold=2):
    trn_texts = open("data/trn.data.txt").read().strip().split("\n")
    trn_labels = list(map(str, open("data/trn.label.txt").read().strip().split("\n")))
    trn_size = len(trn_labels)

    dev_texts = open("data/dev.data.txt").read().strip().split("\n")
    dev_labels = list(map(str, open("data/dev.label.txt").read().strip().split("\n")))

    tst_texts = open("data/tst.data.txt").read().strip().split("\n")

    trn_texts = to_lowercase(trn_texts)
    dev_texts = to_lowercase(dev_texts)
    tst_texts = to_lowercase(tst_texts)
    word_frequency = collections.defaultdict(int)

    for i in range(trn_size):
        words = nltk.wordpunct_tokenize(trn_texts[i])
        for word in words:
            word_frequency[word] += 1
    # sort according to word frequency
    word_frequency = sorted(word_frequency.items(), key=lambda kv: kv[1], reverse=False)

    # eliminate low frequency word
    idx = 0
    word_to_idx = {}
    for word, count in word_frequency:
        if count >= frequency_threshold:
            word_to_idx.update({word: idx})
            idx = idx + 1
    vocab_size = len(word_to_idx)
    print('vocab_size: %d' % vocab_size)
    trn_data_features = bag_of_word(vocab_size, trn_texts, word_to_idx)
    dev_data_features = bag_of_word(vocab_size, dev_texts, word_to_idx)
    tst_data_features = bag_of_word(vocab_size, tst_texts, word_to_idx)

    return np.array(trn_data_features), trn_labels, np.array(dev_data_features), dev_labels, np.array(tst_data_features)


if __name__ == '__main__':
    trn_data_features, trn_labels, dev_data_features, dev_labels, tst_data_features = load_data()
    
    print(trn_data_features.shape, len(trn_labels), dev_data_features.shape, len(dev_labels), tst_data_features.shape)