# ivqa
This my own visual question answering lab.


# project structure
## data
```
+--$home/data/coco
    +--images
        +--/train2014
        +--/val2014
        +--/test
    +--res152fc
        +--*image_feature.h5
+--$home/data/vqa2
    +--*annotations.json
    +--*questions.json
    +--vocab.json
    +--*answer_feature.h5
    +--*question_feature.h5              
```

## code
```
+--$repo/
    +-- config.py   # configuration
    +-- utils.py            
    +-- CocoDataset.py      # build dataset on specific folder
    +-- process
        +-- process_vocab*.py   # process voabulary on various method, default vqal evaluation standard
        +-- process_bow_answer.py
        +-- process_bow_question.py
        +-- process_resnetfc_image.py
    +--BoW
        +-- BoWDataset.py # specific vqa dataset return without image feature
        +-- model.py 
        +-- train.py 
        +-- test.py
     
```


# Envrionment
## Environment value
```
export PYTHONPATH=$repo
# eg
# export PYTHONPATH=/home/sq/git/ivqa
```


# workflow
- process_vocab_vqaeval.py: construct the vocabulary of answer and quesitons
```
process_vocab_vqaeveal.py: default, base on the official evalation 
process_vocab_showask.py: based on the paper show ask attend
```
- process_bow_answer.py: save answer to the h5py file in the format of numpy array
- process_bow_question.py: save question to the h5py file in the format of numpy array
- process_image.py: save image  to the h5py file in the format of numpy array
- train.py: train the model
- validate.py: evaluate the model on the validation set
- test.py: generate the test result
- evaluation.py: evaluate the reslut

# log
tensorboard --logdir=./log

# data structure
## vocab.json
dict:{question:{word:idx},answer:{word:idx}}

# Reference
- https://github.com/Cyanogenoid/pytorch-vqa
