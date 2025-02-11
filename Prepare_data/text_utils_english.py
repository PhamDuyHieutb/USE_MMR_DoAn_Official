import nltk
from nltk.tokenize import sent_tokenize
import re
import os

SPECICAL_CHARACTER = {'(', ')', '[', ']', '"', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
QUOTE = {'(', ')', '[', ']', '"'}

porter = nltk.PorterStemmer()


def text_process_english(sentences, options):
    new_sentences = []

    for item in sentences:
        item = item.strip()
        if item[-1] == '.':
            item = item[:-1]
        tmp = re.sub(r'\s+', ' ', item)
        tmp = tmp.split(' ')

        if options == 'refer':
            tmp = [porter.stem(word) for word in tmp]

        if len(tmp) > 4:
            new_sentences.append(' '.join(tmp))

    return new_sentences


def split_sentences(file_name):
    new_sents = []

    with open(file_name, 'r') as file:
        text_system = file.read().strip()

    tmp = text_system.split(' . ')

    preprocess_sents = text_process_english(tmp, '')  # comment because text were preprocessed
    sentences = []
    for item in preprocess_sents:
        sentences.append(item)

    return sentences


def get_all_sentences(file_system, file_reference):
    sentences_origin_system = []
    for filename in file_system:
        sentences_origin_system.append((filename, split_sentences(filename)))

    reference_docs = []
    for filename in file_reference:
        with open(filename, 'r') as file:
            refe_sents = text_process_english(sent_tokenize(file.read()), 'refer')
            reference_docs.append(' '.join(refe_sents))

    return sentences_origin_system, reference_docs


def write_text(text, path):
    f = open(path, 'w')
    f.write(text)
    f.close()


# clean a list of lines
def clean_lines(lines):
    cleaned = list()

    for i in range(len(lines)):
        line = lines[i]

        # strip source cnn office if it exists in first sentence
        if i == 0:
            index2 = line.find('RRB-')
            if index2 > -1:
                line = line[index2 + len('RRB-'):]

        line = re.sub(r"[^A-Za-z0-9.,']", " ", line)
        line = re.sub(r'\s+', ' ', line)
        line = line.strip()

        # if line has more 4 words => choose
        if len(line.split(' ')) > 6:  # and 'EST' not in line and 'BST' not in line:
            cleaned.append(line)

    # remove empty strings
    cleaned = [c for c in cleaned if len(c) > 0]
    return cleaned


def separate_CNN_Data(path_in, out):
    id = 0

    if not os.path.exists(out + '/documents'): os.mkdir(out + '/documents')
    if not os.path.exists(out + '/summaries'): os.mkdir(out + '/summaries')

    for file in os.listdir(path_in):

        try:
            data = open(path_in + '/' + file, 'r').read().strip()
            data = data.replace("`", "'")

            data = data.replace(" \'re", " are")
            data = data.replace(" \'m", " am")
            data = data.replace(" n\'t", " not")
            data = data.replace(" \'ll", " will")

            data = data.split('@highlight\n\n')

            doc = data[0].strip().split('\n\n')

            doc = [sent.strip() for sent in doc if sent != '']

            if len(doc) > 0:

                doc_punc = []
                for i in doc:
                    if i[-1] == '.':
                        doc_punc.append(i)
                    else:
                        doc_punc.append(i + ' .')

                doc = ' '.join(doc_punc)

                list_sents = clean_lines(doc.split(' . '))

                document = ' . '.join(list_sents)

                summari = [s.replace('\n', '').strip() for s in data[1:]]
                summari = ' . '.join(summari)
                summari = re.sub(r"[^A-Za-z0-9.,']", " ", summari)
                summari = re.sub(r"\s+", " ", summari)

                # print('+++++++++++++++')
                # print(summari)
                # exit()

                write_text(document, out + '/documents/doc_' + str(id))
                write_text(summari, out + '/summaries/summari_' + str(id))

                id += 1
                if id % 5000 == 0:
                    print(id)
            else:
                print(file, 'has nothing')

        except Exception as e:
            print(e, file)
