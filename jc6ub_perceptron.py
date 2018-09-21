import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
from bag_of_words import load_data

def to_feature(data, label):
    # data = data.toarray()
    features = np.repeat(data, 2)
    if label == '0':
        features[::2] = 0 
    elif label == '1':
        features[1::2] = 0 
    else:
        raise NotImplementedError
    return features

def train(trn_data, trn_labels, delta, label_set=['0', '1'], is_shuffle=False):
    '''
    :rtype, np.array with shape (feature_size,)
    '''
    # shuffle
    if is_shuffle:
        trn_data, trn_labels = shuffle(trn_data, trn_labels)
    for i in range(trn_data.shape[0]):
        data, label = trn_data[i], trn_labels[i]
        scores = [np.sum(delta * to_feature(data, label)) for label in label_set]
        if np.argmax(scores) != int(label):
            inv_label = '1' if label == '0' else '0'
            delta = delta + to_feature(data, label) - to_feature(data, inv_label)
    return delta 

def eval(dataset, labels, delta, label_set=['0', '1']):
    acc = 0.
    for i in range(dataset.shape[0]):
        data, label = dataset[i], labels[i]
        scores = [np.sum(delta * to_feature(data, label)) for label in label_set]
        if np.argmax(scores) == int(label):
            acc += 1.
    acc = acc / dataset.shape[0]
    return acc

def main():
    # load data
    trn_data, trn_labels, dev_data, dev_labels, tst_data = load_data(frequency_threshold=3)
    # train model
    epochs = 10
    best_delta = None
    delta = np.zeros(trn_data.shape[1]*2)
    print('delta.shape', delta.shape)
    for epoch in range(epochs):
        print('In epoch %d:' % epoch)
        delta = train(trn_data, trn_labels, delta, is_shuffle=True)
        train_acc = eval(trn_data, trn_labels, delta)
        dev_acc = eval(dev_data, dev_labels, delta)
        print('train acc: %.4f, dev acc: %.4f'% (train_acc, dev_acc))

if __name__ == '__main__':
    main()