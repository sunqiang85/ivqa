import sys

sys.path.append("..")
import config
import json
import numpy as np
from process_vocab_vqaeval import process_punctuation
from tqdm import tqdm
import h5py


class AnswerEncoder():

    def __init__(self, vocab_path, ann_path, outfile_path):
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
        with open(ann_path, 'r') as f:
            self.answers_json = json.load(f)
        self.token_to_index = self.vocab['question']
        self.answer_to_index = self.vocab['answer']

    def _encode_answers(self, answers):
        """ Turn an answer into a vector """
        # answer vec will be a vector of answer counts to determine which answers will contribute to the loss.
        # this should be multiplied with 0.1 * negative log-likelihoods that a model produces and then summed up
        # to get the loss that is weighted by how many humans gave that answer
        answer_vec = np.zeros(len(self.answer_to_index))
        for answer in answers:
            index = self.answer_to_index.get(answer)
            if index is not None:
                answer_vec[index] += 1
        return answer_vec

    def process(self):

        answers = [[a['answer'] for a in ans_dict['answers']] for ans_dict in self.answers_json['annotations']]
        questions_idxs = [ann['question_id'] for ann in self.answers_json['annotations']]
        format_answers = []
        for answer_list in tqdm(answers, desc="format answer:"):
            format_answers.append(list(map(process_punctuation, answer_list)))

        encoded_answers = []
        for answer in tqdm(format_answers, desc="encode answer:"):
            encoded_answers.append(self._encode_answers(answer))
        return encoded_answers, questions_idxs




def econde_answer(input_path, out_path):
    answer_encoder = AnswerEncoder(config.vocabulary_path, input_path, out_path)
    encoded_answers, questions_idxs = answer_encoder.process()
    encoded_answers = np.vstack(encoded_answers)
    h5f = h5py.File(out_path, 'w')
    h5f.create_dataset('answers', data=encoded_answers, dtype="float16")
    h5f.create_dataset('questions_idxs', data=questions_idxs, dtype="long")
    h5f.close()

def main():
    # train
    out_path = "train_answer_feature.h5"
    econde_answer(config.train_ann_file, out_path)


    # val
    out_path = "val_answer_feature.h5"
    econde_answer(config.val_ann_file, out_path)



if __name__ == '__main__':
    main()
