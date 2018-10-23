# ivqa
This my own visual question answering lab.

# workflow
- process_vocab_*.py: construct the vocabulary of answer and quesitons
```
process_vocab_vqaeveal.py: default, base on the official evalation 
process_vocab_showask.py: based on the paper show ask attend
```
- process_answer.py: save answer to the h5py file in the format of numpy array
- process_question.py: save question to the h5py file in the format of numpy array
- process_image.py: save image  to the h5py file in the format of numpy array
- train.py: train the model
- validate.py: evaluate the model on the validation set
- test.py: generate the test result
- evaluation.py: evaluate the reslut

# data structure
## vocab.json
dict:{question:{word:idx},answer:{word:idx}}

# Reference
- https://github.com/Cyanogenoid/pytorch-vqa
