import os
from VQADataset import VQADataset
from VQADataset import collate_fn
from bow_cnn.model import Net
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import utils
import h5py
from tensorboardX import SummaryWriter
from argparse import ArgumentParser
from datetime import datetime


# train
def train(loader, net, writer, epoch, epochs, optimizer, confdict):
    net.train()
    device = torch.device(confdict['device_id'])
    log_softmax = nn.LogSoftmax(dim=1).to(device)
    tq = tqdm(loader, desc='{} E{:03d}/{:03d}'.format("train", epoch + 1, epochs), ncols=0)
    #
    accs = []
    i = 0
    for v, q, a, idx in tq:
        v = v.to(device)
        q = q.to(device)
        a = a.to(device)
        i = i + 1
        out = net(v, q)  # out:batch_size * answer_size
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
            writer.add_scalar("parameter/lr", confdict['learning_rate'], niter)
            writer.add_scalar("parameter/weight_decay", confdict['weight_decay'], niter)
    accs = torch.cat(accs)
    mean_acc = accs.mean()
    print("[E{:03d}][mean acc:{:.4f}]".format(epoch + 1, mean_acc))


# validation
def validate(loader, net, writer, epoch, epochs, confdict):
    net.eval()
    device = torch.device(confdict['device_id'])
    log_softmax = nn.LogSoftmax(dim=1).to(device)
    tq = tqdm(loader, desc='{} E{:03d}/{:03d}'.format("val", epoch + 1, epochs), ncols=0)
    #
    answ = []
    idxs = []
    accs = []
    i = 0
    for v, q, a, idx in tq:
        q = q.to(device)
        a = a.to(device)
        v = v.to(device)
        i = i + 1
        with torch.no_grad():
            out = net(v, q)  # out:batch_size * answer_size
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


def train_val(train_loader, val_loader, confdict):
    # log
    writer = SummaryWriter(confdict['logdir'])
    # model
    net = Net(confdict['output_features'], confdict['max_question_vocab'], confdict['max_answers'])
    device = torch.device(confdict['device_id'])
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
    epochs = confdict['epochs']
    best_epoch = 0
    best_val_acc = 0
    for epoch in range(epochs):
        train(train_loader, net, writer, epoch, epochs, optimizer, confdict)
        r = validate(val_loader, net, writer, epoch, epochs, confdict)
        tmp_val_acc = r[1].mean()
        if tmp_val_acc > best_val_acc:
            best_val_acc = tmp_val_acc
            best_epoch = epoch

    # save model and config
    results = {
        'name': confdict['name'],
        'config': confdict,
        'weights': net.state_dict(),
        'eval': {
            'answers': r[0],
            'accuracies': r[1],
            'idx': r[2],
        },
        'vocab': train_loader.dataset.vocab,
    }
    torch.save(results, confdict['target_name'])

    # save validation answers, accuracies, and idx
    res_h5 = h5py.File(confdict['target_name_result'], 'w')
    res_h5.create_dataset('answers', data=r[0])
    res_h5.create_dataset('accuracies', data=r[1])
    res_h5.create_dataset('idx', data=r[2])
    res_h5.close()
    print("best_epoch:{:3d}, best_val_acc:{:.4f}".format(best_epoch,best_val_acc))
    return best_epoch, best_val_acc


if __name__ == "__main__":
    confdict = utils.get_config()
    parser = ArgumentParser()
    parser.add_argument("-target", "--target-file", help="target file for save, including log and model", dest="target",
                        default=None)
    parser.add_argument("-srcmodel", "--from-model-file", help="from the source model file for load", dest="srcmodel",
                        default=None)
    parser.add_argument("-startepoch", "--start-epoch", type=int, help="offset epoch for start", dest="startepoch",
                        default=0)
    parser.add_argument("-epochs", "--train-epochs", type=int, help="total train epochs", dest="epochs")
    parser.add_argument("-learning_rate", "--learning_rate", type=float, help="learning rate", dest="learning_rate")
    parser.add_argument("-weight_decay", "--weight_decay", type=float, help="weight decay", dest="weight_decay")
    parser.add_argument("-logdir", "--log-directory", type=str, help="log directory", dest="logdir")

    args = parser.parse_args()

    if args.epochs:
        confdict['epochs'] = args.epochs
    if args.learning_rate:
        confdict['learning_rate'] = args.learning_rate
    if args.weight_decay:
        confdict['weight_decay'] = args.weight_decay

    time_suffix = datetime.now().strftime("%m%d%H%M")
    if args.target:
        name = args.target
    else:
        name = "bowcnn"
        name = name + "epoch" + str(confdict['epochs']) + "_"
        name = name + time_suffix
    confdict['name'] = name
    target_name = os.path.join('logs', '{}.pth'.format(name))
    confdict['target_name'] = target_name
    target_name_result = os.path.join('logs', '{}.h5'.format(name))
    confdict['target_name_result'] = target_name_result
    print('will save to {}'.format(target_name))

    if args.logdir:
        confdict['logdir'] = args.logdir
    else:
        confdict['logdir'] = "logs/" + time_suffix

    train_dataset = VQADataset(
        questions_path=confdict['txt_dir'] + "/train_bow_question_feature.h5",
        answers_path=confdict['txt_dir'] + "/train_bow_answer_feature.h5",
        image_features_path=confdict['coco_dir'] + "/res152fc/train2014_image_feature.h5",
        train=True,
        answerable_only=True,

    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=confdict['batch_size'],
        shuffle=True,  # only shuffle the data in training
        pin_memory=True,
        num_workers=confdict['data_workers'],
        collate_fn=collate_fn,
    )

    val_dataset = VQADataset(
        questions_path=confdict['txt_dir'] + "/val_bow_question_feature.h5",
        answers_path=confdict['txt_dir'] + "/val_bow_answer_feature.h5",
        image_features_path=confdict['coco_dir'] + "/res152fc/val2014_image_feature.h5",
        train=True,
        answerable_only=False,

    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=confdict['batch_size'],
        shuffle=False,  # only shuffle the data in training
        pin_memory=True,
        num_workers=confdict['data_workers'],
        collate_fn=collate_fn,
    )
    train_val(train_loader, val_loader, confdict)
