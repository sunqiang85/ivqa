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
from bow_cnn.train import train_val
import pandas as pd


if __name__ == "__main__":
    confdict = utils.get_config()


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

    search_result=[]
    for lr in [0.001, 0.003, 0.01]:
        for wd in [0.001,0.003, 0.01]:
            confdict['learning_rate'] = lr
            confdict['weight_decay'] = wd
            time_suffix = datetime.now().strftime("%m%d%H%M")
            name = "bowcnn"
            name = name + "epoch" + str(confdict['epochs']) + "_"
            name = name + time_suffix
            confdict['name'] = name
            target_name = os.path.join('logs', '{}.pth'.format(name))
            confdict['target_name'] = target_name
            target_name_result = os.path.join('logs', '{}.h5'.format(name))
            confdict['target_name_result'] = target_name_result
            print('will save to {}'.format(target_name))
            confdict['logdir'] = "logs/" + time_suffix
            best_epoch, best_val_acc = train_val(train_loader, val_loader, confdict)
            search_result.append([lr,wd,best_epoch,best_val_acc])
    colnames=['lr','weight_decay', 'best_epoch', 'best_val_acc']
    result_df = pd.DataFrame(search_result,columns=colnames)
    print(result_df)