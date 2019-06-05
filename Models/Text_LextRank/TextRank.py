import os
import nltk
import numpy
from definitions import ROOT_DIR

porter = nltk.PorterStemmer()


class TextRank(object):
    def __init__(self):
        self.text = Preprocessing()

    def PageRank(self, graph, node_weights, d=.85, iter=20):
        weight_sum = numpy.sum(graph, axis=0)
        while iter > 0:
            for i in range(len(node_weights)):
                temp = 0.0
                for j in range(len(node_weights)):
                    temp += graph[i, j] * node_weights[j] / weight_sum[j]
                node_weights[i] = 1 - d + (d * temp)
            iter -= 1

    def buildSummary(self, sentences, node_weights, n):

        top_index = [i for i, j in sorted(enumerate(node_weights), key=lambda x: x[1], reverse=True)]
        summary = []
        # keeping adding sentences until number of words exceeds summary length
        for i in top_index[:n]:
            summary.append(sentences[i])

        return summary

    def main(self, n, path):
        sentences = self.text.processFile(path)

        num_nodes = len(sentences)
        graph = numpy.zeros((num_nodes, num_nodes))

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):  # tinh toan độ trùng lặp giữa 2 sentences
                graph[i, j] = float(len(set(sentences[i].getStemmedWords()) & set(sentences[j].getStemmedWords()))) / (
                        len(sentences[i].getStemmedWords()) + len(sentences[j].getStemmedWords()))
                graph[j, i] = graph[i, j]

        node_weights = numpy.ones(num_nodes)
        self.PageRank(graph, node_weights)
        result = self.buildSummary(sentences, node_weights, n)

        return result


class sentence(object):

    def __init__(self, stemmedWords, OGwords):
        self.stemmedWords = stemmedWords
        self.OGwords = OGwords
        self.lexRankScore = None

    def getStemmedWords(self):
        return self.stemmedWords

    def getOGwords(self):
        return self.OGwords


class Preprocessing(object):

    def processFile(self, file_path_and_name):
        try:
            f = open(file_path_and_name, 'r')
            text_1 = f.read()
            sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
            lines = sent_tokenizer.tokenize(text_1.strip())
            tem_sen = []

            for i in range(len(lines)):
                sent = lines[i].strip()
                OG_sent = sent[:]
                sent = sent.lower()
                line = nltk.word_tokenize(sent)

                stemmed_sentence = [porter.stem(word) for word in line]
                stemmed_sentence = list(filter(lambda x: x != '.' and x != '`' and x != ',' and x != '_' and x != ';'
                                                         and x != '(' and x != ')' and x.find('&') == -1
                                                         and x != '?' and x != "'" and x != '!' and x != '''"'''
                                                         and x != '``' and x != '--' and x != ':'
                                                         and x != "''" and x != "'s", stemmed_sentence))
                if len(stemmed_sentence) <= 4:
                    continue
                if stemmed_sentence:
                    tem_sen.append(sentence(stemmed_sentence, OG_sent))

            return tem_sen

        except IOError:
            print('Oops! File not found', file_path_and_name)
            return [sentence([], "origin")]


if __name__ == '__main__':
    textRank = TextRank()
    path_data = '/home/hieupd/PycharmProjects/Data_DOAN/token_data/cnn/test'
    doc_folders = path_data + "/documents/"
    total_summary = []

    for file in os.listdir(doc_folders):
        path = doc_folders + file
        print("Running TextRank Summarizer for files in folder: ", file)
        summary = textRank.main(3, path)
        final_summary = ""
        for sent in summary:
            final_summary = final_summary + sent.getOGwords() + "\n"
        final_summary = final_summary[:-1]
        results_folder = path_data + "/systems_textRank"
        with open(os.path.join(results_folder, (str(file) + ".TextRank")), "w") as fileOut:
            fileOut.write(final_summary)
