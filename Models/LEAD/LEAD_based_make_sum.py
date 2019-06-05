import shutil
import os
import nltk
from definitions import ROOT_DIR


if __name__ == '__main__':

    cnn_root = ROOT_DIR + 'Data/cnn/test'

    path_data = cnn_root + '/documents'
    path_results = cnn_root + '/summaries_lead3'


    if os.path.exists(path_results):
        shutil.rmtree(path_results)

    os.mkdir(path_results)

    count = 0
    list_files = []
    for name_file in os.listdir(path_data):
        id = name_file.split('_')[1]
        doc = open(path_data + '/' + name_file, 'r').read().strip()

        arr_all_sents = nltk.sent_tokenize(doc)

        f = open(path_results + '/system_' + id, 'w')
        f.write(' '.join(arr_all_sents[:3]))
