# process the vocabulary
import os

# basedir
# txt_dir = os.environ['HOME'] + '/data/vqa2'     # for full
# coco_dir = os.environ['HOME'] + '/data/coco'    # for full
# img_dir = os.environ['HOME'] + '/data/coco/images'    # for full
#
txt_dir = os.environ['HOME'] + '/data/vqa2-sample' # for sample
coco_dir = os.environ['HOME'] + '/data/coco-sample' # for sample
img_dir = os.environ['HOME'] + '/data/coco-sample/images' # for sample

# question
train_question_file = txt_dir + '/v2_OpenEnded_mscoco_train2014_questions.json'
val_question_file = txt_dir + '/v2_OpenEnded_mscoco_val2014_questions.json'
test_dev_question_file = txt_dir + '/v2_OpenEnded_mscoco_test-dev2015_questions.json'
test_question_file = txt_dir + '/v2_OpenEnded_mscoco_test2015_questions.json'

# annotation
train_ann_file = txt_dir + '/v2_mscoco_train2014_annotations.json'
val_ann_file = txt_dir + '/v2_mscoco_val2014_annotations.json'

# vocabulary_path
vocabulary_path = txt_dir+'/vocab.json'  # path where the used vocabularies for question and answers are saved to
max_answers = 3000  # vocab size for answers
max_question_vocab = 5000  # vocab size for questions


# image
image_size = 224
output_features = 2048  # number of feature maps thereof
central_fraction = 0.875  # only take this much of the centre when scaling and centre cropping

# preprocess image
preprocess_batch_size = 32
data_workers = 2

# data
batch_size = 128
data_workers = 1

# train
epochs = 20
device_id = 'cuda:3'
learning_rate = 1e-3
weight_decay = 1e-3

# log
logdir = "logs"
