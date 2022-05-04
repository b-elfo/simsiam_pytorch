import torch
from torch.optim.lr_scheduler import OneCycleLR

from data import dataloader
from model import SimSiamNet
from loss import SimSiamLoss
from train import do_n_epochs

import wandb

###

def train():
    max_lr = 5e-4
    batch_size = 32
    shuffle = True
    num_workers = 4
    num_epochs = 30

    dataset_path = 'CIFAR10'
    size = 32
    # dataset_path = '/media/elfo/Elfo Bkup/Seagate Portable/bird_identifier/train' # needs to be fixed, add 'glob.glob' to dataset imag dir collection
    # dataset_path = '/home/elfo/side_projects/simsiam/feature_extractor/birds_unlablled_noframes'
    # size = 256

    train_dataloader, valid_dataloader = dataloader(dataset_path,
                                                    batch_size=batch_size,
                                                    shuffle=shuffle,
                                                    num_workers=num_workers,
                                                    size=size,
                                                    )

    model = SimSiamNet(model_name='swsl_resnet50',
                       pretrained=True)
    loss_func = SimSiamLoss()
    optim = torch.optim.Adam(params=model.parameters(), 
                             lr=max_lr)

    lr_sched = OneCycleLR(optimizer=optim,
                          max_lr=max_lr,
                          epochs=num_epochs,
                          steps_per_epoch=len(train_dataloader),
                          )

    wandb.init(entity='elfo',
               project='simsiam_birds', 
               name='CIFAR10_pretraining'
               )

    do_n_epochs(train_dataloader=train_dataloader,
                valid_dataloader=valid_dataloader,
                model=model,
                loss_func=loss_func,
                optim=optim,
                lr_sched=lr_sched,
                num_epochs=num_epochs,
                dataset_path=dataset_path,
                )

if __name__=='__main__':
    train()
