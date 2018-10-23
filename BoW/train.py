import sys

sys.path.append('..')
import config
from BoW.BoWDataset import BoWDataset
from BoW.model import Net
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import utils
from tensorboardX import SummaryWriter

# to translate float16 to float32, because torch.stack does not support half_tensor(float16)
def collate_fn(batch):
    # question featrue, answer featrue, question id
    batch = [(b[0].astype("float32"),b[1].astype("float32"),b[2].astype("long")) for b in batch]
    return data.dataloader.default_collate(batch)




# train
def train(loader, net, writer, epoch, epochs):
    device = torch.device(config.device_id)
    log_softmax = nn.LogSoftmax(dim=1).to(device)
    tq = tqdm(loader, desc='{} E{:03d}/{:03d}'.format("train", epoch+1, epochs), ncols=0)
    #
    accs=[]
    i=0
    for q, a, idx in tq:
        q=q.to(device)
        a=a.to(device)
        i=i+1
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
        if i%10 ==0:
            niter = epoch * len(loader) + i
            writer.add_scalar('Train/Loss', loss.data.mean(), niter)
            writer.add_scalar('Train/acc', acc.mean(),niter)
    accs = torch.cat(accs)
    mean_acc = accs.mean()
    print("[E{:03d}][mean acc:{:.4f}]".format(epoch+1,mean_acc))


# validation
def validate(loader, net, writer, epoch, epochs):
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
    answ = torch.cat(answ, dim=0)
    accs = torch.cat(accs, dim=0)
    idxs = torch.cat(idxs, dim=0)
    mean_acc = accs.mean()
    print("[E{:03d}][mean acc:{:.4f}]".format(epoch + 1, mean_acc))
    return answ, accs, idxs


if __name__ == "__main__":
    train_dataset = BoWDataset(
        questions_path="train_question_feature.h5",
        answers_path="train_answer_feature.h5",
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
        questions_path="val_question_feature.h5",
        answers_path="val_answer_feature.h5",
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
    net= Net(config.max_question_vocab,config.max_answers)
    device = torch.device(config.device_id)
    net.to(device)
    optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad])

    # train and validate
    epochs= config.epochs
    for epoch in range(epochs):
        train(train_loader, net, writer, epoch, epochs)
        r = validate(val_loader, net, writer, epoch, epochs)


    