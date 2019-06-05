"""
tokenize data, split train test validate follow list url respectively

"""

import os
import hashlib
import subprocess
import shutil
from definitions import ROOT_DIR

all_train_urls = ROOT_DIR + "/Data/url_lists/cnn_wayback_training_urls.txt"
all_val_urls = ROOT_DIR + "/Data/url_lists/cnn_wayback_validation_urls.txt"
all_test_urls = ROOT_DIR + "/Data/url_lists/cnn_wayback_test_urls.txt"

cnn_stories_dir = ROOT_DIR + '/Data/CNN_raw'  # folder for data CNN raw
cnn_tokenized_stories_dir = ROOT_DIR + "/Data/cnn_stories_tokenized"  # folder for data CNN tokenized
finished_files_dir = ROOT_DIR + "/cnn"  # folder for train, test, valid tokenized

# These are the number of .story files we expect there to be in cnn_stories_dir
num_expected_cnn_stories = 92579


def tokenize_stories(stories_dir, tokenized_stories_dir):
    """Maps a whole directory of .story files to a tokenized version using Stanford CoreNLP Tokenizer"""
    print("Preparing to tokenize %s to %s..." % (stories_dir, tokenized_stories_dir))
    stories = os.listdir(stories_dir)
    # make IO list file
    print("Making list of files to tokenize...")
    with open("mapping.txt", "w") as f:
        for s in stories:
            f.write("%s \t %s\n" % (os.path.join(stories_dir, s), os.path.join(tokenized_stories_dir, s)))
    command = ['java', 'edu.stanford.nlp.process.PTBTokenizer', '-ioFileList', '-preserveLines', 'mapping.txt']
    print("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tokenized_stories_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping.txt")

    # Check that the tokenized stories directory contains the same number of files as the original directory
    num_orig = len(os.listdir(stories_dir))
    num_tokenized = len(os.listdir(tokenized_stories_dir))
    if num_orig != num_tokenized:
        raise Exception(
            "The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
                tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
    print("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tokenized_stories_dir))


def read_text_file(text_file):
    lines = []
    with open(text_file, "r") as f:
        for line in f:
            lines.append(line.strip())
    return lines


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode())
    return h.hexdigest()


def get_url_hashes(url_list):
    return [hashhex(url) for url in url_list]


def split_train_test_valid(url_file, out_file):
    """Reads the tokenized .story files corresponding to the urls listed in the url_
    file and writes them to a out_file."""

    print("Making bin file for URLs listed in %s..." % url_file)
    url_list = read_text_file(url_file)

    url_hashes = get_url_hashes(url_list)
    story_fnames = [s + ".story" for s in url_hashes]
    num_stories = len(story_fnames)
    print(num_stories)

    for file in story_fnames:
        path_file = cnn_tokenized_stories_dir + '/' + file
        path_des = out_file + '/' + file
        shutil.copy(path_file, path_des)


def check_num_stories(stories_dir, num_expected):
    num_stories = len(os.listdir(stories_dir))
    if num_stories != num_expected:
        raise Exception(
            "stories directory %s contains %i files but should contain %i" % (stories_dir, num_stories, num_expected))


if __name__ == '__main__':

    # Check the stories directories contain the correct number of .story files
    check_num_stories(cnn_tokenized_stories_dir, num_expected_cnn_stories)

    # Create some new directories
    if not os.path.exists(cnn_tokenized_stories_dir): os.makedirs(cnn_tokenized_stories_dir)
    if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)

    # Run stanford tokenizer on both stories dirs, outputting to tokenized stories directories
    tokenize_stories(cnn_stories_dir, cnn_tokenized_stories_dir)

    # Read the tokenized stories, do a little postprocessing then write to bin files
    split_train_test_valid(all_test_urls, os.path.join(finished_files_dir, "test"))
    split_train_test_valid(all_val_urls, os.path.join(finished_files_dir, "valid"))
    split_train_test_valid(all_train_urls, os.path.join(finished_files_dir, "train"))
