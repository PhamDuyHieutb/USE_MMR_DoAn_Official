import nltk
import operator
import os
import json
import math
import numpy as np
import re
import string
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer

porter = nltk.PorterStemmer()

SPECICAL_CHARACTER = {'(', ')', '[', ']', '"', '”', '“', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
QUOTE = {'"'}

extra_stopwords = ["''", "``", "'s"]

class sentence(object):

    def __init__(self, docName, stemmedWords, OGwords):

        self.stemmedWords = stemmedWords
        self.docName = docName
        self.OGwords = OGwords
        self.wordFrequencies = self.sentenceWordFreqs()
        self.lexRankScore = None

    def getStemmedWords(self):
        return self.stemmedWords

    def getDocName(self):
        return self.docName

    def getOGwords(self):
        return self.OGwords

    def getWordFreqs(self):
        return self.wordFrequencies

    def getLexRankScore(self):
        return self.LexRankScore

    def setLexRankScore(self, score):
        self.LexRankScore = score

    def sentenceWordFreqs(self):
        wordFreqs = {}
        for word in self.stemmedWords:
            if word not in wordFreqs.keys():
                wordFreqs[word] = 1
            else:
                wordFreqs[word] = wordFreqs[word] + 1

        return wordFreqs


def split_doc(file_name):
    with open(file_name, 'r') as file:
        docs = file.read().strip().split('\n####\n')
    # sentences = []
    # for doc in docs:
    #     sentences += doc.split('\n')

    return docs


def split_sentences(file_name):
    sentences = []
    with open(file_name, 'r') as file:
        data = file.read().strip().split('\n')
        for sent in data:
            sentences.append(sent)

    return sentences


def split_sentences_from_text(text):
    sentence_token = nltk.data.load('tokenizers/punkt/english.pickle')
    tmp = sentence_token.tokenize(text)

    sentences = []
    for item in tmp:
        if "…" in item:
            b = item.split("…")
            for i in b:
                sentences.append(i)
        else:
            sentences.append(item)

    return sentences


def separate_label_sent(sentences):
    arr_labels = []
    arr_sents = []
    for i in sentences:
        arr_labels.append(i[0])
        arr_sents.append(i[2:])
    return arr_labels, arr_sents


def remove_short_sents(old_sentences):
    new_sentences_stem = []
    new_sentences_origin = []
    for i in range(len(old_sentences)):
        line = old_sentences[i]

        # chuyển đối tất cả chữ hoa trong chuỗi sang kiểu chữ thường "Good Mike" => "good mike"
        stemmedSent = line.strip().lower().split()  # remove label => line[2:]

        stemmedSent = list(
            filter(lambda x: x != '.' and x != '`' and x != ',' and x != '?' and x != "'" and x != ":"
                             and x != '!' and x != '''"''' and x != "''" and x != '-',
                   stemmedSent))

        if ((i + 1) == len(old_sentences)) and (len(stemmedSent) < 5):
            break
        if stemmedSent != []:
            new_sentences_stem.append(stemmedSent)
            new_sentences_origin.append(old_sentences[i])

    return new_sentences_stem, new_sentences_origin


def text_process(sentences, stop_words):
    new_sentences = []

    for item in sentences:
        tmp = item.lower()
        text_tmp = []
        for word in tmp.split(' '):
            if word not in stop_words:  # and (len(word) != 1 or word in SPECICAL_CHARACTER)
                text_tmp.append(word)

        new_sentences.append(' '.join(text_tmp))

    return new_sentences


def text_process_all(docs, stop_words):

    # remove stopwords, stem word

    new_docs = []
    for doc in docs:
        new_sents = []
        sents = doc.split('\n')
        for sent in sents:
            tmp = sent[2:].lower()  # remove label from data => [2:]
            text_tmp = []
            for word in tmp.split(' '):
                if word not in stop_words:
                    text_tmp.append(porter.stem(word))

            if len(text_tmp) > 0:
                new_sents.append(' '.join(text_tmp))

        new_docs.append(' '.join(new_sents))
    return new_docs




def get_sentence_first_paragraph(file_name, stop_words):
    with open(file_name, 'r') as file:
        text = file.readlines()

    contain = []
    for item in text:
        contain.append(split_sentences_from_text(item)[0])

    contain = text_process(contain, stop_words)

    contain = set(contain)

    return contain


def get_doc_from_sentences(sentences):
    doc = []
    for sent in sentences:
        doc.append(sent)

    return ' '.join(doc)


def get_freq_words_from_doc(doc):
    words = {}

    for word in doc.split(' '):
        if word not in words:
            words[word] = 1
        else:
            words[word] += 1

    return words


def get_word_freq(word, dict_words):
    if word not in dict_words:
        return 0

    return dict_words[word]


def tf(word, document):
    dict_words = get_freq_words_from_doc(document)
    count_words = 0

    for item in dict_words:
        count_words += dict_words[item]

    return dict_words[word] * 1.0 / count_words


def idf(word, documents):
    N = len(documents)

    contain_word = 0
    for doc in documents:

        if word in doc:
            contain_word += 1

    return math.log(1.0 * N / contain_word)


def save_idf(idf_dict, output):
    with open(output, 'w') as fp:
        json.dump(idf_dict, fp)


def read_json_file(path):
    with open(path, 'r') as fp:
        dicti = json.load(fp)
    return dicti


def get_idf_sklearn(documents):

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(documents)
    idf = vectorizer.idf_
    all_idf = dict(zip(vectorizer.get_feature_names(), idf))

    return all_idf

def get_all_idf(documents):
    words = {}

    for item in documents:
        for word in item.split(' '):
            if word not in words:
                words[word] = 0.0
    print('compute idf')
    count = 0
    for item in words:
        count += 1
        if count % 1000 == 0:
            print(count)

        words[item] = idf(item, documents)

    return words


def get_freq_word_uni(document):
    tf_words = {}
    for item in document.split(" "):
        if item not in QUOTE and item not in tf_words:
            tf_words[item] = tf(item, document)

    number_freq_word = int(0.3 * len(tf_words))
    freq_words = {}

    i = 0
    for key, value in sorted(tf_words.items(), key=operator.itemgetter(1), reverse=True):
        freq_words[key] = value

        i += 1
        if i == number_freq_word:
            break

    return freq_words


def get_centroid_uni(document, all_idf):
    words = {}

    for word in document.split(' '):
        if word not in QUOTE and word not in words:
            words[word] = 0.0

    for item in words:
        if item in all_idf:
            words[item] = tf(item, document) * all_idf[item]
        else:
            words[item] = tf(item, document) * 0.1

    number_centroid_uni = int(0.3 * len(words))

    centroid_uni = {}
    i = 0
    for key, value in sorted(words.items(), key=operator.itemgetter(1), reverse=True):
        centroid_uni[key] = value

        i += 1
        if i == number_centroid_uni:
            break

    return centroid_uni


def read_all_train_documents(file_names, stop_words):
    documents = []
    for filename in file_names:
        print(filename)
        docs = split_doc(filename)

        docs_clean = text_process_all(docs, stop_words)  # remove stopwords

        documents += docs_clean
    print('num doc ', len(documents))
    return documents


def convert_uni_to_bi(documents):
    bi_documents = []

    for item in documents:
        words = []
        for word in item.split(' '):
            if word not in QUOTE:
                words.append(word)

        bi_document = []
        for item in range(1, len(words)):
            bi_document.append(words[item - 1] + "__" + words[item])

        tmp = ""
        for item in bi_document:
            tmp += item + " "

        bi_documents.append(tmp[:-1])

    return bi_documents


from definitions import ROOT_DIR


def get_position_train_features(vectors, predict_labels, id):
    docs = open(ROOT_DIR + '/Data/CNN/data_labels_clus/train/train_' + id, 'r').read().strip().split('\n####\n')

    human_labels = []
    count = 0
    vectors_pos = np.zeros(21)
    for doc in docs:
        arr_pos_feas = []
        sents = doc.split('\n')
        numsent = len(sents)

        for i in range(numsent):
            arr_pos_feas.append([1 / math.log1p(i + 2)])

        vec_pos = np.concatenate((np.array(arr_pos_feas), vectors[count:(count + numsent)]), axis=1)
        vec_pos_norm2 = normalize(vec_pos, axis=1, norm='l2')
        vectors_pos = np.vstack((vectors_pos, vec_pos_norm2))

        human_labels += [int(s[0]) for s in doc.split('\n')]

        count += numsent


    if np.array_equal(human_labels, predict_labels):
        print('match')
    return vectors_pos[1:]


def get_position_features_by_file(vectors):
    arr_pos_feas = []
    numsent = len(vectors)

    for i in range(numsent):
        arr_pos_feas.append([1 / math.log1p(i + 2)])

    vectors_pos = np.concatenate((np.array(arr_pos_feas), vectors), axis=1)
    vectors_pos = normalize(vectors_pos, axis=1, norm='l2')

    return vectors_pos

def cos_similarity(s1, idf, sentences):
    '''
    compute cosine similarity of any sentence with first sentence of once document
    :param s1:
    :param s2:
    :param list_sent:
    :return:
    '''

    doc = get_doc_from_sentences(sentences)
    numerator = 0
    denom1 = 0
    denom2 = 0
    list_word_s1_tmp = s1.split(' ')
    list_word_s2_tmp = sentences[0].split(' ')

    list_word_s1 = []
    for item in list_word_s1_tmp:
        if item not in QUOTE:
            list_word_s1.append(item)

    list_word_s2 = []
    for item in list_word_s2_tmp:
        if item not in QUOTE:
            list_word_s2.append(item)

    all_words = set(list_word_s1 + list_word_s2)

    tf_arr = {}
    for word in all_words:
        tf_arr[word] = tf(word, doc)
    for word in list_word_s1:
        if word in idf:
            denom1 += (tf_arr[word] * idf[word]) ** 2
        else:
            denom1 += (tf_arr[word] * 0.1) ** 2

    for word in list_word_s2:
        tf_w = tf_arr[word]
        idf_w = 0
        if word in idf:
            idf_w = idf[word]
            denom2 += (tf_w * idf_w) ** 2
        if word in list_word_s1 and word in idf:
            numerator += (tf_w * idf_w) ** 2
        else:
            numerator += (tf_w * 0.1) ** 2
            denom2 += (tf_w * 0.1) ** 2
    sim = 0.0
    try:
        sim = numerator / (math.sqrt(denom1) * math.sqrt(denom2))
    except Exception:
        pass

    return sim


def concate_features(cnn_features, svm_features_path, option):
    all_svm_features = []

    list_files = []
    if option == 'train':
        list_files = open('/home/hieupd/PycharmProjects/CNN_SVM_summarization/Models/CNN_eng/train_files.txt',
                          'r').read().split('\n')
    else:
        list_files = open('/home/hieupd/PycharmProjects/CNN_SVM_summarization/Models/CNN_eng/test_files.txt',
                          'r').read().split('\n')

    for file in list_files:
        path_file = svm_features_path + '/' + file
        features = open(path_file, 'r').read().split('\n')
        all_svm_features += [fea[2:].strip().split(' ') for fea in features]

    return np.concatenate((np.array(all_svm_features), cnn_features), axis=1)


def prepare_data_svm(ar_labels, ar_svm, output):
    all_features = []
    for i in range(len(ar_labels)):
        feature = ar_labels[i] + ' ' + ' '.join(list(map(str, ar_svm[i])))
        all_features.append(feature)

    with open(output, 'w') as f:
        f.write('\n'.join(all_features))
        f.close()


def write_file_text(data, path_file):
    out = open(path_file, 'w')
    out.write(data)
    out.close()


def read_file_text(path_file):
    with open(path_file, 'r') as content:
        data = content.read()
    content.close()
    return data


def convert_features_svm(path):
    train = []
    test = []
    for clus in os.listdir(path + '/' + 'train'):
        f = open(path + '/train/' + clus, 'r')
        train += f.read().split('\n')
        f.close()

    for clus in os.listdir(path + '/' + 'test'):
        t = open(path + '/test/' + clus, 'r')
        test += t.read().split('\n')
        t.close()
    # train, test = train_test_split(all_features, test_size=0.2,random_state= 42)

    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    for vec in train:
        X_train.append(list(map(float, vec[2:].split(' '))))  # [:9]
        Y_train.append(int(vec[0]))

    for v in test:
        X_test.append(list(map(float, v[2:].split(' '))))  # [:9]
        Y_test.append(int(v[0]))

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    return X_train, Y_train, X_test, Y_test

def remove_stopwords(sent, stopwords):
    new_token = []
    sent = sent.strip().lower()
    for w in sent.split(' '):
        if w not in stopwords:
            new_token.append(w)

    return new_token

# clean and stemmer
def clean_stem_sent(sent, list_stopwords):
    token = []

    token += remove_stopwords(sent, list_stopwords)

    words = [w for w in token if w not in string.punctuation]
    words = [w for w in words if w not in extra_stopwords]
    stemmedSent = [porter.stem(word) for word in words]
    if len(words) > 0:
        return ' '.join(stemmedSent)
    else:
        return ''