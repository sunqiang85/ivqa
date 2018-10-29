# coding=utf-8

# process the annotation.json to vocab.json
# This code is based on the code of VQA
# (https://github.com/GT-Vision-Lab/VQA/tree/master/PythonEvaluationTools/vqaEvaluation)
# This code is based on the code written by Tsung-Yi Lin for MSCOCO Python API available at the following link:
# (https://github.com/tylin/coco-caption/blob/master/pycocoevalcap/eval.py).
import json
from collections import Counter
import itertools
import re
import config

contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
                "couldnt": "couldn't",
                "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't",
                "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
                "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't",
                "hed": "he'd", "hed've": "he'd've",
                "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
                "Id've": "I'd've", "I'dve": "I'd've",
                "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've",
                "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
                "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've",
                "mightn'tve": "mightn't've", "mightve": "might've",
                "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've",
                "oclock": "o'clock", "oughtnt": "oughtn't",
                "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't",
                "shed've": "she'd've", "she'dve": "she'd've",
                "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't",
                "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
                "somebody'd": "somebodyd", "somebodyd've": "somebody'd've",
                "somebody'dve": "somebody'd've", "somebodyll": "somebody'll",
                "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've",
                "someone'dve": "someone'd've",
                "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd",
                "somethingd've": "something'd've",
                "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's",
                "thered": "there'd", "thered've": "there'd've",
                "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd",
                "theyd've": "they'd've",
                "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've",
                "twas": "'twas", "wasnt": "wasn't",
                "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't",
                "whatll": "what'll", "whatre": "what're",
                "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd",
                "wheres": "where's", "whereve": "where've",
                "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll",
                "whos": "who's", "whove": "who've", "whyll": "why'll",
                "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've",
                "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
                "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
                "yall'd've": "y'all'd've",
                "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd",
                "youd've": "you'd've", "you'dve": "you'd've",
                "youll": "you'll", "youre": "you're", "youve": "you've"}
manualMap = {'none': '0',
             'zero': '0',
             'one': '1',
             'two': '2',
             'three': '3',
             'four': '4',
             'five': '5',
             'six': '6',
             'seven': '7',
             'eight': '8',
             'nine': '9',
             'ten': '10'
             }
articles = ['a',
            'an',
            'the'
            ]

periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
commaStrip = re.compile("(\d)(\,)(\d)")
punct = [';', r"/", '[', ']', '"', '{', '}',
         '(', ')', '=', '+', '\\', '_', '-',
         '>', '<', '@', '`', ',', '?', '!']


# main text processer
def process_punctuation(intext):
    resAns = intext
    resAns = resAns.replace('\n', ' ')
    resAns = resAns.replace('\t', ' ')
    resAns = resAns.strip()
    resAns = processPunctuation(resAns)
    resAns = processDigitArticle(resAns)
    return resAns


# sub
def processPunctuation(inText):
    outText = inText
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) or (re.search(commaStrip, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = periodStrip.sub("",
                              outText,
                              re.UNICODE)
    return outText


# sub
def processDigitArticle(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manualMap.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText


def prepare_questions(questions_json):
    """ Tokenize and normalize questions from a given question json in the usual VQA format. """
    questions = [q['question'] for q in questions_json['questions']]
    for question in questions:
        question = question.lower()[:-1]
        yield question.split(' ')


def prepare_answers(answers_json):
    """ Normalize answers from a given answer json in the usual VQA format. """
    answers = [[a['answer'] for a in ans_dict['answers']] for ans_dict in answers_json['annotations']]
    # The only normalization that is applied to both machine generated answers as well as
    # ground truth answers is replacing most punctuation with space (see [0] and [1]).
    # Since potential machine generated answers are just taken from most common answers, applying the other
    # normalizations is not needed, assuming that the human answers are already normalized.
    # [0]: http://visualqa.org/evaluation.html
    # [1]: https://github.com/VT-vision-lab/VQA/blob/3849b1eae04a0ffd83f56ad6f70ebd0767e09e0f/PythonEvaluationTools/vqaEvaluation/vqaEval.py#L96

    for answer_list in answers:
        yield list(map(process_punctuation, answer_list))


def extract_vocab(iterable, top_k=None, start=0):
    """ Turns an iterable of list of tokens into a vocabulary.
        These tokens could be single answers or word tokens in questions.
    """
    all_tokens = itertools.chain.from_iterable(iterable)
    counter = Counter(all_tokens)
    if top_k:
        most_common = counter.most_common(top_k)
        most_common = (t for t, c in most_common)
    else:
        most_common = counter.keys()
    # descending in count, then lexicographical order
    tokens = sorted(most_common, key=lambda x: (counter[x], x), reverse=True)
    vocab = {t: i for i, t in enumerate(tokens, start=start)}
    return vocab


def main():
    questions = config.train_question_file
    answers = config.train_ann_file

    with open(questions, 'r') as fd:
        questions = json.load(fd)
    with open(answers, 'r') as fd:
        answers = json.load(fd)

    answers = prepare_answers(answers)

    vocab_questions = prepare_questions(questions)
    question_vocab = extract_vocab(vocab_questions, start=1)
    vocab_questions = prepare_questions(questions)
    question_vocab_topk = extract_vocab(vocab_questions, top_k=config.max_question_vocab)
    answer_vocab = extract_vocab(answers, top_k=config.max_answers)

    vocabs = {
        'question': question_vocab,
        'question_topk': question_vocab_topk,
        'answer': answer_vocab,
    }
    with open(config.vocabulary_path, 'w') as fd:
        json.dump(vocabs, fd)


if __name__ == '__main__':
    main()
