import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def main():
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
    print(trn_data.shape)
    print(type(trn_data))
    print(trn_data[0])
    print(trn_data[0].toarray().shape)
    # for value in trn_data[0]:
    #     print(value, type(value))

if __name__ == '__main__':
    main()