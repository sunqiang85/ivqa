import os
import config
from bow.BoWDataset import BoWDataset
from bow.BoWDataset import collate_fn
from bow.model import Net
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import utils
import h5py
from tensorboardX import SummaryWriter
from argparse import ArgumentParser
from datetime import datetime


# train
def train(loader, net, writer, epoch, epochs):
    net.train()
    device = torch.device(config.device_id)
    log_softmax = nn.LogSoftmax(dim=1).to(device)
    tq = tqdm(loader, desc='{} E{:03d}/{:03d}'.format("train", epoch + 1, epochs), ncols=0)
    #
    accs = []
    i = 0
    for q, a, idx in tq:
        q = q.to(device)
        a = a.to(device)
        i = i + 1
        out = net(q)  # out:batch_size * answer_size
        nll = -log_softmax(out)
        loss = (nll * a / 10).sum(dim=1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = utils.batch_accuracy(out.data, a.data).cpu()
        fmt = '{:.4f}'.format
        tq.set_postfix(loss=fmt(loss.data.mean()), acc=fmt(acc.data.mean()))
        accs.append(acc)
        if i % 10 == 0:
            niter = epoch * len(loader) + i
            writer.add_scalar('Train/Loss', loss.data.mean(), niter)
            writer.add_scalar('Train/acc', acc.mean(), niter)
    accs = torch.cat(accs)
    mean_acc = accs.mean()
    print("[E{:03d}][mean acc:{:.4f}]".format(epoch + 1, mean_acc))


# validation
def validate(loader, net, writer, epoch, epochs):
    net.eval()
    device = torch.device(config.device_id)
    log_softmax = nn.LogSoftmax(dim=1).to(device)
    tq = tqdm(loader, desc='{} E{:03d}/{:03d}'.format("val", epoch + 1, epochs), ncols=0)
    #
    answ = []
    idxs = []
    accs = []
    i = 0
    for q, a, idx in tq:
        q = q.to(device)
        a = a.to(device)
        i = i + 1
        with torch.no_grad():
            out = net(q)  # out:batch_size * answer_size
            nll = -log_softmax(out)
            loss = (nll * a / 10).sum(dim=1).mean()
            acc = utils.batch_accuracy(out.data, a.data).cpu()
            fmt = '{:.4f}'.format
            tq.set_postfix(loss=fmt(loss.data.mean()), acc=fmt(acc.data.mean()))
            _, answer = out.data.cpu().max(dim=1)
        answ.append(answer)
        accs.append(acc)
        idxs.append(idx)
        if i % 10 == 0:
            niter = epoch * len(loader) + i
            writer.add_scalar('Val/Loss', loss.data.mean(), niter)
            writer.add_scalar('Val/acc', acc.mean(), niter)

    mean_acc = (torch.cat(accs, dim=0)).mean()
    print("[E{:03d}][mean acc:{:.4f}]".format(epoch + 1, mean_acc))

    answ = torch.cat(answ, dim=0).numpy()
    accs = torch.cat(accs, dim=0).numpy()
    idxs = torch.cat(idxs, dim=0).numpy()
    return answ, accs, idxs


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-t", "--target-file", help="target file for save, including log and model", dest="target",
                        default=None)
    parser.add_argument("-f", "--from-model-file", help="from the source model file for load", dest="srcmodel",
                        default=None)
    parser.add_argument("-s", "--start-echo", help="offset echo for start", dest="startecho", default=0)
    args = parser.parse_args()
    if args.target:
        name = args.target
    else:
        name = "bow"
        name = name + "echo" + str(config.epochs) + "_"
        name = name + datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    target_name = os.path.join('logs', '{}.pth'.format(name))
    target_name_result = os.path.join('logs', '{}.h5'.format(name))
    print('will save to {}'.format(target_name))

    train_dataset = BoWDataset(
        questions_path=config.txt_dir + "/train_bow_question_feature.h5",
        answers_path=config.txt_dir + "/train_bow_answer_feature.h5",
        image_features_path=None,
        train=True,
        answerable_only=True,

    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,  # only shuffle the data in training
        pin_memory=True,
        num_workers=config.data_workers,
        collate_fn=collate_fn,
    )

    val_dataset = BoWDataset(
        questions_path=config.txt_dir + "/val_bow_question_feature.h5",
        answers_path=config.txt_dir + "/val_bow_answer_feature.h5",
        image_features_path=None,
        train=True,
        answerable_only=False,

    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,  # only shuffle the data in training
        pin_memory=True,
        num_workers=config.data_workers,
        collate_fn=collate_fn,
    )

    # log
    writer = SummaryWriter('log')

    # model
    net = Net(config.max_question_vocab, config.max_answers)
    device = torch.device(config.device_id)
    net.to(device)

    weight_p, bias_p = [], []
    for name, p in net.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]
    # 这里的model中每个参数的名字都是系统自动命名的，只要是权值都是带有weight，偏置都带有bias，
    # 因此可以通过名字判断属性，这个和tensorflow不同，tensorflow是可以用户自己定义名字的，当然也会系统自己定义。
    optimizer = optim.Adam([
        {'params': weight_p, 'weight_decay': 1e-3},
        {'params': bias_p, 'weight_decay': 0}
    ], lr=1e-3)

    # train and validate
    epochs = config.epochs
    for epoch in range(epochs):
        train(train_loader, net, writer, epoch, epochs)
        r = validate(val_loader, net, writer, epoch, epochs)

    # save model and config
    config_as_dict = {k: v for k, v in vars(config).items() if not k.startswith('__') and not k == 'os'}
    results = {
        'name': name,
        'config': config_as_dict,
        'weights': net.state_dict(),
        'eval': {
            'answers': r[0],
            'accuracies': r[1],
            'idx': r[2],
        },
        'vocab': train_loader.dataset.vocab,
    }
    torch.save(results, target_name)

    # save validation answers, accuracies, and idx
    res_h5 = h5py.File(target_name_result, 'w')
    res_h5.create_dataset('answers', data=r[0])
    res_h5.create_dataset('accuracies', data=r[1])
    res_h5.create_dataset('idx', data=r[2])
    res_h5.close()
