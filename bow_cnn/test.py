import sys
import json
import torch
from tqdm import tqdm
from VQADataset import VQADataset
from VQADataset import collate_fn_for_test
from bow_cnn.model import Net
import config


def main():
    if len(sys.argv) > 1:
        train_results_filename = sys.argv[1]
    else:
        train_results_filename = u'logs/test.pth'

    # Load the train result parameters to model
    train_results = torch.load(train_results_filename)
    train_model_state_dict = train_results['weights']

    val_dataset = VQADataset(
        questions_path=config.txt_dir + "/test_bow_question_feature.h5",
        answers_path=None,
        image_features_path=config.coco_dir + "/res152fc/test2015_image_feature.h5",
        train=False,
        answerable_only=False,
    )

    loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,  # only shuffle the data in training
        pin_memory=True,
        num_workers=config.data_workers,
        collate_fn=collate_fn_for_test,
    )

    # load model
    net = Net(config.output_features, config.max_question_vocab, config.max_answers)
    net.load_state_dict(train_model_state_dict)
    net.eval()
    device = torch.device(config.device_id)
    net.to(device)

    # start test loop
    answ = []
    idxs = []
    i = 0
    tq = tqdm(loader, desc='{} E{:03d}'.format("test", 1), ncols=0)
    for v, q, idx in tq:
        q = q.to(device)
        v = v.to(device)
        i = i + 1
        with torch.no_grad():
            out = net(v, q)  # out:batch_size * answer_size
            _, answer = out.data.cpu().max(dim=1)
            answ.append(answer)
            idxs.append(idx)
    answ = torch.cat(answ, dim=0)
    idxs = torch.cat(idxs, dim=0)

    # prepare answer word list
    answers = []
    index_to_answers = {}
    for k in loader.dataset.answer_to_index:
        index_to_answers[loader.dataset.answer_to_index[k]] = k
    for c in answ:
        answers.append(index_to_answers[int(c)])

    # prepare answer file with question id and corresponding answer
    question_ids = idxs.cpu().numpy()
    results = []
    for q, a in zip(question_ids, answers):
        results.append({"question_id": int(q), "answer": a})
    with open('results/test2015_result.json', 'w') as fd:
        json.dump(results, fd)


if __name__ == '__main__':
    main()
