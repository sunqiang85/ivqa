import sys
import config
import json
import h5py
import torch.utils.data as data


# for train only, to translate float16 to float32, because torch.stack does not support half_tensor(float16)
def collate_fn(batch):
    # question featrue, answer featrue, question id
    batch = [(b[0].astype("float32"), b[1].astype("float32"), b[2].astype("long")) for b in batch]
    return data.dataloader.default_collate(batch)


# for test only, to translate float16 to float32
def collate_fn_for_test(batch):
    # question featrue, answer featrue, question id
    batch = [(b[0].astype("float32"), b[1].astype("long")) for b in batch]
    return data.dataloader.default_collate(batch)


class BoWDataset(data.Dataset):
    """ VQA dataset, open-ended """

    # when train is False, answerable_only can not be true
    def __init__(self, questions_path, answers_path=None, image_features_path=None, train=True, answerable_only=False):
        super(BoWDataset, self).__init__()
        self.is_train = train
        self.q_h5 = h5py.File(questions_path, 'r')
        if train:
            self.a_h5 = h5py.File(answers_path, 'r')
        with open(config.vocabulary_path, 'r') as fd:
            vocab_json = json.load(fd)

        # vocab
        self.vocab = vocab_json
        self.token_to_index = self.vocab['question']
        self.answer_to_index = self.vocab['answer']
        self.questions_idxs = self.q_h5['questions_idxs'][:]

        # q and a

        self.questions = self.q_h5['questions']

        # only use questions that have at least one answer?
        self.answerable_only = answerable_only
        if train:
            self.answers = self.a_h5['answers']
            if self.answerable_only:
                self.answerable = self._find_answerable()

    def _check_integrity(self):
        if self.is_train:
            a_idxs = self.q_h5['questions_idxs'][:]
            q_idxs = self.a_h5['questions_idxs'][:]
            assert (all(a_idxs == q_idxs)), "question_id not match between questions and answers"

    def _find_answerable(self):
        """ Create a list of indices into questions that will have at least one answer that is in the vocab """
        answerable = []
        for i, answers in enumerate(self.answers):
            answer_has_index = len(answers.nonzero()) > 0
            # store the indices of anything that is answerable
            if answer_has_index:
                answerable.append(i)
        return answerable

    def __getitem__(self, item):
        if self.answerable_only:
            # change of indices to only address answerable questions
            item = self.answerable[item]

        q = self.questions[item]

        idx = self.questions_idxs[item]
        v = None
        if self.is_train:
            a = self.answers[item]
            # we return question_feature, answer_featrue, question_id
            return q, a, idx
        else:
            return q, idx

    def __len__(self):
        if self.answerable_only:
            return len(self.answerable)
        else:
            return len(self.questions)
