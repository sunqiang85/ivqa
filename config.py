# process the vocabulary
import os

# basedir
basedir = os.environ['HOME'] + '/data/vqa2-sample'

# question
train_question_file = basedir + '/v2_OpenEnded_mscoco_train2014_questions.json'
val_question_file = basedir + '/v2_OpenEnded_mscoco_val2014_questions.json'
test_dev_question_file = basedir + '/v2_OpenEnded_mscoco_test-dev2015_questions.json'
test_question_file = basedir + '/v2_OpenEnded_mscoco_test2015_questions.json'

# annotation
train_ann_file = basedir + '/v2_mscoco_train2014_annotations.json'
val_ann_file = basedir + '/v2_mscoco_val2014_annotations.json'

# vocabulary_path
max_answers = 3000  # vocab size for answers
max_question_vocab = 5000  # vocab size for questions
vocabulary_path = '/home/sq/git/ivqa/vocab.json'  # path where the used vocabularies for question and answers are saved to

# data
batch_size = 128
data_workers = 4

# train
epochs = 2
device_id = 'cuda:3'
