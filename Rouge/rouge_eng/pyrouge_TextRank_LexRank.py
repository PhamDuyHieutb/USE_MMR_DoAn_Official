#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pyrouge import Rouge155
import time
from definitions import ROOT_DIR

start_time = time.time()

if __name__ == "__main__":

    rouge_dir = ROOT_DIR + '/Rouge/rouge_eng/ROUGE-1.5.5'


    rouge_args = '-e ROUGE-1.5.5/data -n 2 -m -u -c 95 -f A -p 0.5 -t 0 -a' #'-e ROUGE-1.5.5/data -n 2 -m -u -c 95 -r 1000 -f A -p 0.5 -t 0 -a -x -l 100'

    rouge = Rouge155(rouge_dir, rouge_args)

    # 'model' refers to the human summaries
    rouge.model_dir = ROOT_DIR + "/Data/cnn/test/summaries"
    rouge.model_filename_pattern = 'summari_#ID#'

    rouge.system_dir = ROOT_DIR + "/Data/summaries/systems_textRank"
    rouge.system_filename_pattern = 'doc_(\d+).TextRank'

    print ("-----------------MMR  TExtRank--------------------------")

    rouge_output = rouge.convert_and_evaluate()
    print(rouge_output)

    print ("Execution time: " + str(time.time() - start_time) )
