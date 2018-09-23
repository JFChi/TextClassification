import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
from jc6ub_bag_of_words import load_data

def to_feature(data, label):
    features = np.repeat(data, 2)
    if label == '0':
        features[::2] = 0 
    elif label == '1':
        features[1::2] = 0 
    else:
        raise NotImplementedError
    return features

def train(trn_data, trn_labels, theta, steps, label_set=['0', '1'], shuffle=False):
    '''
    :rtype, np.array with shape (feature_size,)
    '''
    # shuffle
    if shuffle:
        trn_data, trn_labels = shuffle(trn_data, trn_labels)
    for i in range(trn_data.shape[0]):
        steps += 1
        data, label = trn_data[i], trn_labels[i]
        scores = [np.sum(theta * to_feature(data, label)) for label in label_set]
        if np.argmax(scores) != int(label):
            inv_label = '1' if label == '0' else '0'
            theta = theta + to_feature(data, label) - to_feature(data, inv_label)
    return theta, steps

def eval(dataset, labels, theta, label_set=['0', '1']):
    acc = 0.
    for i in range(dataset.shape[0]):
        data, label = dataset[i], labels[i]
        scores = [np.sum(theta * to_feature(data, label)) for label in label_set]
        if np.argmax(scores) == int(label):
            acc += 1.
    acc = acc / dataset.shape[0]
    return acc

def predict(dataset, theta, label_set=['0', '1']):
    preds = []
    for i in range(dataset.shape[0]):
        data = dataset[i]
        scores = [np.sum(theta * to_feature(data, label)) for label in label_set]
        preds.append(str(np.argmax(scores)))
    return np.array(preds)

def main():
    # load data
    trn_data, trn_labels, dev_data, dev_labels, tst_data = load_data(frequency_threshold=2)
    # train model
    epochs = 40
    steps = 0
    theta = np.zeros(trn_data.shape[1]*2)
    m = np.zeros(trn_data.shape[1]*2)
    best_theta = np.zeros(trn_data.shape[1]*2)
    best_dev_acc = 0.0
    print('theta.shape', theta.shape)
    for epoch in range(epochs):
        print('In epoch %d:' % epoch)
        theta, steps = train(trn_data, trn_labels, theta, steps)
        m = theta + m # update
        train_acc = eval(trn_data, trn_labels, m/steps)
        dev_acc = eval(dev_data, dev_labels, m/steps)
        print('at steps %d train acc: %.4f, dev acc: %.4f'% (steps, train_acc, dev_acc))
        if dev_acc >= best_dev_acc:
            best_theta = m/steps
            best_dev_acc = dev_acc
            print('Achieve best dev acc: %.4f' % best_dev_acc)

    tst_pred = predict(tst_data, best_theta)
    print(tst_pred.shape)
    np.savetxt('jc6ub-averaged-perceptron-test.pred', tst_pred, fmt='%s')

if __name__ == '__main__':
    main()