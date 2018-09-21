import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle

def load_data():
    # Load data
    with open("data/trn.data.txt", 'r') as f:
        trn_texts = f.read().splitlines() 
    with open("data/trn.label.txt", 'r') as f:
        trn_labels = f.read().splitlines()
    print("Training data ...")
    print(len(trn_texts), len(trn_labels))

    with open("data/dev.data.txt", 'r') as f:
        dev_texts = f.read().splitlines() 
    with open("data/dev.label.txt", 'r') as f:
        dev_labels = f.read().splitlines()
    print("Development data ...")
    print(len(dev_texts), len(dev_labels))

    with open("data/tst.data.txt", 'r') as f:
        tst_texts = f.read().splitlines()
    print("Test data ...")
    print(len(tst_texts))
    return trn_texts, trn_labels, dev_texts, dev_labels, tst_texts

def to_feature(data, label):
    data = data.toarray()
    features = np.repeat(data, 2)
    if label == '0':
        features[::2] = 0 
    elif label == '1':
        features[1::2] = 0 
    else:
        raise NotImplementedError
    return features

def train(trn_data, trn_labels, delta, steps, label_set=['0', '1'], shuffle=False):
    '''
    :rtype, np.array with shape (feature_size,)
    '''
    # shuffle
    if shuffle:
        trn_data, trn_labels = shuffle(trn_data, trn_labels)
    for i in range(trn_data.shape[0]):
        steps += 1
        data, label = trn_data[i], trn_labels[i]
        scores = [np.sum(delta * to_feature(data, label)) for label in label_set]
        if np.argmax(scores) != int(label):
            inv_label = '1' if label == '0' else '0'
            delta = delta + to_feature(data, label) - to_feature(data, inv_label)
    return delta, steps

def eval(dataset, labels, delta, label_set=['0', '1']):
    acc = 0.
    for i in range(dataset.shape[0]):
        data, label = dataset[i], labels[i]
        scores = [np.sum(delta * to_feature(data, label)) for label in label_set]
        if np.argmax(scores) == int(label):
            acc += 1.
    acc = acc / dataset.shape[0]
    return acc

def predict(dataset, delta, label_set=['0', '1']):
    preds = []
    for i in range(dataset.shape[0]):
        data = dataset[i]
        scores = [np.sum(delta * to_feature(data, label)) for label in label_set]
        preds.append(str(np.argmax(scores)))
    return np.array(preds)

def main():
    # Load data
    trn_texts, trn_labels, dev_texts, dev_labels, tst_texts = load_data()
    # vectorize
    choice = 3
    if choice == 1:
        print("Preprocessing without any feature selection")
        vectorizer = CountVectorizer(lowercase=False)
        # vocab size 77166
    elif choice == 2:
        print("Lowercasing all the tokens")
        vectorizer = CountVectorizer(lowercase=True)
        # vocab size 60610
    elif choice == 3:
        print("Lowercasing and filtering out low-frequency words")
        vectorizer = CountVectorizer(lowercase=True, min_df=2)
        # vocab size 31218
    elif choice == 4:
        print("Lowercasing and filtering out low-frequency words, uni- and bi-gram")
        vectorizer = CountVectorizer(lowercase=True, min_df=2, ngram_range=(1,2))
        # vocab size 323167
    elif choice == 5:
        print("Uni- and bi-gram")
        vectorizer = CountVectorizer(ngram_range=(1,2))
        # vocab 1048596
    elif choice == 6:
        print("Lowercasing and filtering out high-frequency words")
        vectorizer = CountVectorizer(lowercase=True, max_df=0.5)
        # vocab size 60610

    trn_data = vectorizer.fit_transform(trn_texts)
    print('trn_data.shape', trn_data.shape)
    dev_data = vectorizer.transform(dev_texts)
    print('dev_data.shape', dev_data.shape)
    tst_data = vectorizer.transform(tst_texts)
    print('tst_data.shape', tst_data.shape)
    # train model
    epochs = 10
    steps = 0
    delta = np.zeros(trn_data.shape[1]*2)
    m = np.zeros(trn_data.shape[1]*2)
    print('delta.shape', delta.shape)
    for epoch in range(epochs):
        print('In epoch %d:' % epoch)
        delta, steps = train(trn_data, trn_labels, delta, steps)
        train_acc = eval(trn_data, trn_labels, delta/steps)
        dev_acc = eval(dev_data, dev_labels, delta/steps)
        print('at steps %d train acc: %.4f, dev acc: %.4f'% (steps, train_acc, dev_acc))

    best_delta = delta/steps
    tst_pred = predict(tst_data, best_delta)
    print(tst_pred.shape)
    np.savetxt('jc6ub-averaged-perceptron-test.pred', tst_pred, fmt='%s')

if __name__ == '__main__':
    main()