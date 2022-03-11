import torch
from torch.optim.lr_scheduler import OneCycleLR

from data import dataloader
from model import SimSiamNet
from loss import SimSiamLoss
from train import do_n_epochs

import wandb

###

def train():
    init_lr = 5e-2
    batch_size = 16
    shuffle = True
    num_workers = 4
    num_epochs = 20

    train_dataloader, valid_dataloader = dataloader('/media/elfo/Elfo Bkup/Seagate Portable/bird_identifier/train', #'/home/elfo/side_projects/simsiam/feature_extractor/birds_unlablled_noframes',
                                                    batch_size=batch_size,
                                                    shuffle=shuffle,
                                                    num_workers=num_workers,
                                                    )

    model = SimSiamNet(model_name='swsl_resnet50',
                       pretrained=True)
    loss_func = SimSiamLoss()
    optim = torch.optim.Adam(params=model.parameters(), 
                             lr=init_lr)

    lr_sched = OneCycleLR(optimizer=optim,
                          max_lr=init_lr,
                          epochs=num_epochs,
                          steps_per_epoch=len(train_dataloader),
                          )

    wandb.init(project='simsiam_birds')

    do_n_epochs(train_dataloader=train_dataloader,
                valid_dataloader=valid_dataloader,
                model=model,
                loss_func=loss_func,
                optim=optim,
                lr_sched=lr_sched,
                num_epochs=num_epochs,
                )

if __name__=='__main__':
    train()
