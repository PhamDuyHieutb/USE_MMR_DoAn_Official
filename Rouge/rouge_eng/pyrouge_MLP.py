from pyrouge import Rouge155
import time
from definitions import ROOT_DIR

start_time = time.time()

if __name__ == "__main__":

    rouge_dir = ROOT_DIR + '/Rouge/rouge_eng/ROUGE-1.5.5'
    data = '/home/hieupd/PycharmProjects/Data_DOAN/token_data/cnn'


    # '-e ROUGE-1.5.5/data -n 2 -m -u -c 95 -f A -p 0.5 -t 0 -a'
    rouge_args = '-e ROUGE-1.5.5/data -a -c 95 -m -n 2'

    # '-e', self._rouge_data,                           # '-a',  # evaluate all systems
    # '-n', 4,  # max-ngram                             # '-x',  # do not calculate ROUGE-L
    # '-2', 4,  # max-gap-length                        # '-u',  # include unigram in skip-bigram
    # '-c', 95,  # confidence interval                  # '-r', 1000,  # number-of-samples (for resampling)
    # '-f', 'A',  # scoring formula                     # '-p', 0.5,  # 0 <= alpha <=1
    # '-t', 0,  # count by token instead of sentence    # '-d',  # print per evaluation scores

    rouge = Rouge155(rouge_dir, rouge_args)
    # rouge = Rouge155()

    #rouge.model_dir = data + '/sum_test'
    rouge.model_dir = data + '/test/summaries'
    # rouge.model_dir = data +
    # '/summaries_for_lead'
    rouge.model_filename_pattern = 'summari_#ID#'

    #rouge.system_dir = data + '/test/summaries_lead3'
    rouge.system_dir = ROOT_DIR + '/Data_Progress/summaries/universal_19_stand_cnn'
    rouge.system_filename_pattern = 'system_(\d+)'

    print("-------------------------------------------")

    rouge_output = rouge.convert_and_evaluate()
    print(rouge_output)

    print("Execution time: " + str(time.time() - start_time))
