import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import CIFAR10

import optuna
from optuna.trial import TrialState
from optuna.integration.wandb import WeightsAndBiasesCallback

from data import dataloader
from loss import SimSiamLoss
from model import SimSiamNet, DownStreamNet
import train, downstream

###

DEVICE = torch.device("cuda")
EPOCHS = 15

DEFAULT_HYPERPARAMS = {
    "policy": SimSiamNet,
    "verbose": 0,
    "device": 'cuda',
}

###

def simsiam_params(trial):
    encoder_name = trial.suggest_categorical("model_name", ["resnet50","swsl_resnet50"])
    max_lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])

    return {
        "encoder_name": encoder_name,
        "max_lr": max_lr,
        "batch_size": batch_size,
        "optimizer_name": optimizer_name,
    }

def objective_pretrain(trial):
    kwargs = DEFAULT_HYPERPARAMS.copy()
    kwargs.update(simsiam_params(trial))
    
    shuffle = True
    num_workers = 4
    
    dataset_path = 'CIFAR10'
    size = 32
    
    train_dataloader, valid_dataloader = dataloader(dataset_path,
                                                    batch_size=kwargs['batch_size'],
                                                    shuffle=shuffle,
                                                    num_workers=num_workers,
                                                    size=size,
                                                    )

    model = SimSiamNet(model_name=kwargs['encoder_name'],
                       pretrained=True)
    loss_func = SimSiamLoss()
    optimizer = getattr(optim, kwargs["optimizer_name"])(params=model.parameters(), 
                                                         lr=kwargs["max_lr"], 
                                                         )
    lr_sched = OneCycleLR(optimizer=optimizer,
                          max_lr=kwargs["max_lr"],
                          epochs=EPOCHS,
                          steps_per_epoch=len(train_dataloader),
                          )

    valid_loss = 0

    for epoch in range(EPOCHS):
        train_loss = train.do_train_epoch(dataloader=train_dataloader, 
                                          model=model, 
                                          loss_func=loss_func,
                                          optim=optimizer,
                                          lr_sched=lr_sched,
                                          #scaler=scaler,
                                          )
        
        valid_loss = train.do_valid_epoch(dataloader=valid_dataloader,
                                          model=model,
                                          loss_func=loss_func)

        print("Epoch: {}\tTrain loss: {:.2f}\tValid loss: {:.2f}\n".format(epoch, train_loss, valid_loss))

        trial.report(valid_loss, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return valid_loss


def objective_downstream(trial):
    kwargs = DEFAULT_HYPERPARAMS.copy()
    kwargs.update(simsiam_params(trial))

    pretrained_weight_path = './models/CIFAR10/pretrain_model_30.pth',
    
    shuffle = True
    num_workers = 4
    
    dataset_path = 'CIFAR10'
    size = 32
    
    if dataset_path == 'CIFAR10':
        dataset = CIFAR10(root='data/', download=True, transform=transforms.ToTensor())
    else:
        pass

    valid_size = dataset.__len__()//4
    data_split = [dataset.__len__()-valid_size, valid_size]
    
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, 
                                                                 lengths=data_split)

    train_dataloader = DataLoader(dataset=train_dataset, 
                                  batch_size=kwargs['batch_size'], 
                                  shuffle=shuffle, 
                                  num_workers=num_workers,
                                  )
    valid_dataloader = DataLoader(dataset=valid_dataset, 
                                  batch_size=kwargs['batch_size'], 
                                  shuffle=shuffle, 
                                  num_workers=num_workers,
                                  )
                                
    model = DownStreamNet(model_name='swsl_resnet50',
                          pretrained=False)

    weights = torch.load(pretrained_weight_path)
    try:
        model.backbone.load_state_dict(weights)
    except Exception as err:
        print("Attempted weight loading failed.")
        print(err)
        return 

    num_classes = len(dataset.class_to_idx)
    one_hot = nn.functional.one_hot(torch.arange(0,num_classes)).float().to(DEVICE)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), 
                                 lr=kwargs['max_lr'],
                                 )
    lr_sched = OneCycleLR(optimizer=optim,
                          max_lr=kwargs['max_lr'],
                          epochs=EPOCHS,
                          steps_per_epoch=len(train_dataloader),
                          )
                          
    valid_loss = 0

    for epoch in range(EPOCHS):
        train_loss = downstream.do_train_epoch(dataloader=train_dataloader, 
                                               model=model, 
                                               loss_func=loss_func,
                                               optim=optimizer,
                                               lr_sched=lr_sched,
                                               #scaler=scaler,
                                               )
        
        valid_loss = downstream.do_valid_epoch(dataloader=valid_dataloader,
                                               model=model,
                                               loss_func=loss_func)

        print("Epoch: {}\tTrain loss: {:.2f}\tValid loss: {:.2f}\n".format(epoch, train_loss, valid_loss))

        trial.report(valid_loss, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return valid_loss


def main():
    wandb_kwargs = {'entity': 'elfo', 
                    'project': 'simsiam-cifar10-downstream-optuna' # 'simsiam-cifar10-optuna' # 
                    }
    wandbc = WeightsAndBiasesCallback(metric_name="valid_loss", wandb_kwargs=wandb_kwargs)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective_downstream, n_trials=20, callbacks=[wandbc])

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study Statistics: ")
    print(f"   Number of finished trials: {len(study.trials)}")
    print(f"   Number of pruned trials:   {len(pruned_trials)}")
    print(f"   Number of complete trials: {len(complete_trials)}")

    print("BEST TRIAL: ")
    trial = study.best_trial
    print(f"   Valid loss: {trial.value}")
    print(f"   Params:     {trial.params.items()}")


if __name__ == '__main__':
    main()