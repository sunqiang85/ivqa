import json

# process the vocabulary
import os

# basedir
basedir = os.environ['HOME'] + '/data'
imagedir = os.environ['HOME'] + '/data/coco/images'

# question
train_question_file = basedir + '/vqa2/v2_OpenEnded_mscoco_train2014_questions.json'
val_question_file = basedir + '/vqa2/v2_OpenEnded_mscoco_val2014_questions.json'
test_dev_question_file = basedir + '/vqa2/v2_OpenEnded_mscoco_test-dev2015_questions.json'
test_question_file = basedir + '/vqa2/v2_OpenEnded_mscoco_test2015_questions.json'

# annotation
train_ann_file = basedir + '/vqa2/v2_mscoco_train2014_annotations.json'
val_ann_file = basedir + '/vqa2/v2_mscoco_val2014_annotations.json'

# smaple output path
sampledir = os.environ['HOME'] + '/data/vqa2-sample'
sampleimagedir = os.environ['HOME'] + '/data/coco-sample/images'

# question
sample_train_question_file = sampledir + '/v2_OpenEnded_mscoco_train2014_questions.json'
sample_val_question_file = sampledir + '/v2_OpenEnded_mscoco_val2014_questions.json'
sample_test_dev_question_file = sampledir + '/v2_OpenEnded_mscoco_test-dev2015_questions.json'
sample_test_question_file = sampledir + '/v2_OpenEnded_mscoco_test2015_questions.json'

# annotation
sample_train_ann_file = sampledir + '/v2_mscoco_train2014_annotations.json'
sample_val_ann_file = sampledir + '/v2_mscoco_val2014_annotations.json'


def sample_question(in_path, out_path, percent=0.01):
    with open(in_path,'r') as f:
        questions_json = json.load(f)
    questions = questions_json['questions']
    sample_len = int(len(questions) * percent)
    questions_json['questions'] = questions[:sample_len]
    with open(out_path, 'w') as fd:
        json.dump(questions_json, fd)


def sample_annotation(in_path, out_path, percent=0.01):
    with open(in_path,'r') as f:
        ann_json = json.load(f)
    annotations = ann_json['annotations']
    sample_len = int(len(annotations) * percent)
    ann_json['annotations'] = annotations[:sample_len]
    with open(out_path, 'w') as fd:
        json.dump(ann_json, fd)




def copy(image_id,set_name):
    filename = "COCO_"+set_name+"_"+"{:012d}".format(image_id)+".jpg"
    source_name = os.path.join(imagedir,set_name,filename)
    target_name = os.path.join(sampleimagedir,set_name,filename)
    return "cp {:s} {:s}\n".format(source_name, target_name)

def sample_images(question_path,set_name):
    with open(question_path,'r') as f:
        questions_json= json.load(f)
    image_ids = [c['image_id'] for c in questions_json['questions']]
    image_ids = list(set(image_ids))
    image_ids.sort()
    cmd_str = ""
    for c in image_ids:
        str = copy(c,set_name)
        cmd_str=cmd_str+str
    return cmd_str




if __name__ == "__main__":
    sample_question(train_question_file, sample_train_question_file)
    sample_question(val_question_file, sample_val_question_file)
    sample_question(test_dev_question_file, sample_test_dev_question_file)
    sample_question(test_question_file, sample_test_question_file)

    sample_annotation(train_ann_file, sample_train_ann_file)
    sample_annotation(val_ann_file, sample_val_ann_file)
    cmd_str = sample_images(sample_train_question_file, "train2014")
    cmd_str = cmd_str + sample_images(sample_val_question_file,"val2014")
    cmd_str = cmd_str + sample_images(sample_test_question_file,"test2015")
    with open("sample_images.sh","w") as f:
        f.write(cmd_str)
