import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression as LR

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

def main():
    # Load data
    trn_texts, trn_labels, dev_texts, dev_labels, tst_texts = load_data()
    # vectorize
    choice = 0
    if choice == 0:
        print("Preprocessing in default setting")
        vectorizer = CountVectorizer()
    elif choice == 1:
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
    # train
    lamda = 1. # 1e-3, 1e-2, 1e-1, 1., 10., 100.
    penalty = 'l1' # 'l1'
    print('lambda:', lamda, ' penalty: ', penalty)
    classifier = LR(C=1./lamda, penalty=penalty)
    classifier.fit(trn_data, trn_labels)
    print("Training accuracy =", classifier.score(trn_data, trn_labels))
    print("Dev accuracy =", classifier.score(dev_data, dev_labels))

    # give prediction on test set
    tst_pred = classifier.predict(tst_data) # numpy array
    np.savetxt('jc6ub-lr-test.pred', tst_pred, fmt='%s')

if __name__ == '__main__':
    main()