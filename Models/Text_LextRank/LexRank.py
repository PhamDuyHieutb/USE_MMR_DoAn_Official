import math
import os
import nltk
import numpy


porter = nltk.PorterStemmer()


class LexRank(object):
    def __init__(self):
        self.text = Preprocessing()
        self.sim = DocumentSim()

    def score(self, sentences, idfs, CM, t):

        Degree = [0 for i in sentences]
        n = len(sentences)

        for i in range(n):
            for j in range(n):
                CM[i][j] = self.sim.sim(sentences[i], sentences[j], idfs)
                Degree[i] += CM[i][j]

        for i in range(n):
            for j in range(n):
                CM[i][j] = CM[i][j] / float(Degree[i])

        L = self.PageRank(CM, n)
        normalizedL = self.normalize(L)

        for i in range(len(normalizedL)):
            score = normalizedL[i]
            sentence = sentences[i]
            sentence.setLexRankScore(score)

        return sentences

    def PageRank(self, CM, n, maxerr=.0001):
        Po = numpy.zeros(n)
        P1 = numpy.ones(n)
        M = numpy.array(CM)
        t = 0
        while (numpy.sum(numpy.abs(P1 - Po)) > maxerr) and (t < 100):
            Po = numpy.copy(P1)
            t = t + 1
            P1 = numpy.matmul(Po, M)
        # 	print(numpy.sum(numpy.abs(P1-Po)))
        # print(t)
        return list(Po)

    def buildMatrix(self, sentences):

        # build our matrix
        CM = [[0 for s in sentences] for s in sentences]

        for i in range(len(sentences)):
            for j in range(len(sentences)):
                CM[i][j] = 0
        return CM

    def buildSummary(self, sentences, n):
        sentences = sorted(sentences, key=lambda x: x.getLexRankScore(), reverse=True)

        summary = [sentences[0]]
        # keeping adding sentences until number of words exceeds summary length
        i = 0
        while i < (n - 1):
            i += 1
            flag = True
            for sen_sum in summary:
                if sentences[i].getStemmedWords() == sen_sum.getStemmedWords():
                    flag = False
            if flag:
                summary.append(sentences[i])
        return summary

    def normalize(self, numbers):
        max_number = max(numbers)
        normalized_numbers = []

        for number in numbers:
            normalized_numbers.append(number / max_number)

        return normalized_numbers

    def main(self, n, path):
        sentences = self.text.processFile(path)
        idfs = self.sim.IDFs(sentences)
        CM = self.buildMatrix(sentences)

        sentences = self.score(sentences, idfs, CM, 0.1)

        summary = self.buildSummary(sentences, n)

        return summary


class sentence(object):

    def __init__(self, stemmedWords, OGwords):

        self.stemmedWords = stemmedWords
        self.OGwords = OGwords
        self.wordFrequencies = self.sentenceWordFreqs()
        self.lexRankScore = None

    def getStemmedWords(self):
        return self.stemmedWords

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


class DocumentSim(object):
    def __init__(self):
        self.text = Preprocessing()

    def TFs(self, sentences):

        tfs = {}
        for sent in sentences:
            wordFreqs = sent.getWordFreqs()

            for word in wordFreqs.keys():
                if tfs.get(word, 0) != 0:
                    tfs[word] = tfs[word] + wordFreqs[word]
                else:
                    tfs[word] = wordFreqs[word]
        return tfs

    def TFw(self, word, sentence):
        return sentence.getWordFreqs().get(word, 0)

    def IDFs(self, sentences):

        N = len(sentences)
        idfs = {}
        words = {}
        w2 = []

        for sent in sentences:
            for word in sent.getStemmedWords():
                if sent.getWordFreqs().get(word, 0) != 0:
                    words[word] = words.get(word, 0) + 1

        for word in words:
            n = words[word]
            try:
                w2.append(n)
                idf = math.log10(float(N) / n)
            except ZeroDivisionError:
                idf = 0

            idfs[word] = idf

        return idfs

    def IDF(self, word, idfs):
        return idfs[word]

    def sim(self, sentence1, sentence2, idfs):

        numerator = 0
        denom1 = 0
        denom2 = 0

        for word in sentence2.getStemmedWords():
            numerator += self.TFw(word, sentence2) * self.TFw(word, sentence1) * self.IDF(word, idfs) ** 2

        for word in sentence1.getStemmedWords():
            denom2 += (self.TFw(word, sentence1) * self.IDF(word, idfs)) ** 2

        for word in sentence2.getStemmedWords():
            denom1 += (self.TFw(word, sentence2) * self.IDF(word, idfs)) ** 2

        try:
            return numerator / (math.sqrt(denom1) * math.sqrt(denom2))

        except ZeroDivisionError:
            return float("-inf")


if __name__ == '__main__':
    lexRank = LexRank()
    path_data = '/home/hieupd/PycharmProjects/Data_DOAN/token_data/cnn/test'
    doc_folders = path_data + "/documents/"
    total_summary = []

    for file in os.listdir(doc_folders):
        path = doc_folders + file
        print("Running LexRank Summarizer for files in folder: ", file)
        summary = lexRank.main(3, path)
        final_summary = ""
        for sent in summary:
            final_summary = final_summary + sent.getOGwords() + "\n"
        final_summary = final_summary[:-1]
        results_folder = path_data + "/systems_lexRank"
        with open(os.path.join(results_folder, (str(file) + ".LexRank")), "w") as fileOut:
            fileOut.write(final_summary)
