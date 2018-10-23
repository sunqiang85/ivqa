import sys

sys.path.append("..")
import config
import json
import numpy as np
from process_vocab_vqaeval import process_punctuation
from tqdm import tqdm
import h5py


class QuestionEncoder():

    def __init__(self, vocab_path, question_path, outfile_path):
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
        with open(question_path, 'r') as f:
            self.quesiton_json = json.load(f)
        self.token_to_index = self.vocab['question_topk']
        self.answer_to_index = self.vocab['answer']

    def _encode_questions(self, question):
        question_vec = np.zeros(len(self.token_to_index))
        for word in question:
            index = self.token_to_index.get(word)
            if index is not None:
                question_vec[index] += 1
        return question_vec

    def filter_question(self, question):
        filter_question = []
        for word in question:
            index = self.token_to_index.get(word)
            if index is not None:
                filter_question.append(word)
        return filter_question

    def process(self):

        questions = [q['question'] for q in self.quesiton_json['questions']]
        questions_idxs = [q['question_id'] for q in self.quesiton_json['questions']]
        questions = [c.lower()[:-1].split(' ') for c in questions]
        encoded_questions = []
        for question in tqdm(questions, desc="encode question:"):
            encoded_questions.append(self._encode_questions(question))
        return encoded_questions, questions_idxs


def encode_question(input_path, out_path):
    question_encoder = QuestionEncoder(config.vocabulary_path, input_path, out_path)
    encoded_questions, questions_idxs = question_encoder.process()
    encoded_questions = np.vstack(encoded_questions)
    h5f = h5py.File(out_path, 'w')
    h5f.create_dataset('questions', data=encoded_questions, dtype="float16")
    h5f.create_dataset('questions_idxs', data=questions_idxs, dtype="long")
    h5f.close()


def main():
    # train
    out_path = "train_question_feature.h5"
    encode_question(config.train_question_file, out_path)

    # val
    out_path = "val_question_feature.h5"
    encode_question(config.val_question_file, out_path)

    # test
    out_path = "test_question_feature.h5"
    encode_question(config.test_question_file, out_path)


if __name__ == '__main__':
    main()
