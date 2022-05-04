import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import CIFAR10

from data import dataloader
from model import DownStreamNet, SimSiamNet
from loss import SimSiamLoss

import wandb

###

def do_train_epoch(dataloader: DataLoader,
                   model: SimSiamNet,
                   loss_func: nn.Module,
                   optim: torch.optim,
                   lr_sched: torch.optim,
                   class_map: torch.Tensor = None,
                   ):
    model = model.train()
    total_loss = 0
    step = 0

    for i, (input, label) in enumerate(dataloader):
        input, label = input.to('cuda'), label.to('cuda')

        predict = model(input)

        if torch.isnan(predict.sum()):
            print("NaNs found in output")

        loss = loss_func(predict, label)
        if torch.isnan(loss):
            print("NaNs in loss function, loss-sum: {}".format(torch.sum(loss)))
        total_loss += loss
        
        if i%100==0:
            wandb.log({'current_loss':loss,
                       'running_loss':total_loss, 
                       'batch_num':i,
                       })

        optim.zero_grad()
        loss.backward()
        optim.step()

        if lr_sched:
            lr_sched.step()
        
        if step%500==0:
            print("Step: {}\tLoss: {:.2f}".format(step, loss))
        step+=1

    return total_loss/len(dataloader)

def do_valid_epoch(dataloader: DataLoader,
                   model: SimSiamNet,
                   loss_func: nn.Module,
                   class_map: torch.Tensor = None,
                   demo: bool = False,
                   ):
    model = model.eval()

    with torch.no_grad():
        total_loss = 0

        for i, (input, label) in enumerate(dataloader):
            input, label = input.to('cuda'), label.to('cuda')
            
            predict = model(input)

            # if demo:
            #     plt.imshow(input.cpu().permute(1,2,0))
            #     plt.title()
            #     plt.show()

            loss = loss_func(predict, label)

            total_loss += loss

    return total_loss/len(dataloader)

def do_n_epochs(train_dataloader: DataLoader,
                valid_dataloader: DataLoader,
                model: SimSiamNet,
                loss_func: nn.Module,
                optim: torch.optim,
                lr_sched: torch.optim,
                num_epochs: int = 1,
                dataset_path: str = '',
                class_map: torch.Tensor = None,
                ):

    for epoch in range(num_epochs):
        train_loss = do_train_epoch(dataloader=train_dataloader, 
                                    model=model, 
                                    loss_func=loss_func,
                                    optim=optim,
                                    lr_sched=lr_sched,
                                    class_map=class_map,
                                    )
        
        valid_loss = do_valid_epoch(dataloader=valid_dataloader,
                                    model=model,
                                    loss_func=loss_func,
                                    class_map=class_map,
                                    )
        
        wandb.log({'train_loss': train_loss, 'valid_loss': valid_loss}, commit=True)

        print("Epoch: {}\tTrain loss: {:.2f}\tValid loss: {:.2f}\n".format(epoch, train_loss, valid_loss))

        dataset_name = dataset_path.split('/')[-1]

        torch.save( obj=model.state_dict(), f='models/'+dataset_name+'/'+f'model_{epoch+1}.pth' )

###

def train_task(dataset_path: str,
               pretrained_weight_path: str,
               init_lr: float,
               batch_size: int,
               shuffle: bool = True,
               num_workers: int = 4,
               num_epochs: int = 30,
               device: str = 'cuda',
               ):
    if dataset_path == 'CIFAR10':
        dataset = CIFAR10(root='data/', download=True, transform=transforms.ToTensor())
    else:
        pass

    valid_size = dataset.__len__()//4
    data_split = [dataset.__len__()-valid_size, valid_size]
    
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, 
                                                                 lengths=data_split)

    train_dataloader = DataLoader(dataset=train_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=shuffle, 
                                  num_workers=num_workers,
                                  )
    valid_dataloader = DataLoader(dataset=valid_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=shuffle, 
                                  num_workers=num_workers,
                                  )
                                
    model = DownStreamNet(model_name='swsl_resnet50',
                          pretrained=True)

    # LOAD MODEL ENCODER 
    weights = torch.load(pretrained_weight_path)
    try:
        model.backbone.load_state_dict(weights)
    except Exception as err:
        print("Attempted weight loading failed.")
        print(err)
        return 
    ####################

    # MAP CLASS LABELS TO VECTORS
    num_classes = len(dataset.class_to_idx)
    one_hot = nn.functional.one_hot(torch.arange(0,num_classes)).float().to(device)
    #############################

    loss_func = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(params=model.parameters(), 
                             lr=init_lr)
    lr_sched = OneCycleLR(optimizer=optim,
                          max_lr=init_lr,
                          epochs=num_epochs,
                          steps_per_epoch=len(train_dataloader),
                          )

    do_n_epochs(train_dataloader=train_dataloader,
                valid_dataloader=valid_dataloader,
                model=model,
                loss_func=loss_func,
                optim=optim,
                lr_sched=lr_sched,
                num_epochs=num_epochs,
                dataset_path=dataset_path,
                class_map=one_hot,
                )
    
###

def validate_task(dataset_path: str,
                  pretrained_weight_path: str,
                  batch_size: int = 1,
                  num_samples: int = 9,
                  device: str = 'cuda',
                  ):
    if dataset_path == 'CIFAR10':
        dataset = CIFAR10(root='data/', download=True, transform=transforms.ToTensor())
    else:
        pass

    idx_to_class = {v:k for k,v in dataset.class_to_idx.items()}

    valid_dataloader = DataLoader(dataset=dataset, 
                                  batch_size=batch_size, 
                                  )
                                
    model = DownStreamNet(model_name='swsl_resnet50',
                          pretrained=True)

    # LOAD MODEL ENCODER 
    weights = torch.load(pretrained_weight_path)
    try:
        model.load_state_dict(weights)
    except Exception as err:
        print("Attempted weight loading failed.")
        print(err)
        return 
    
    model = model.eval()
    total_num = 0
    correct_num = 0

    with torch.no_grad():
        samples = []

        for _, (input, label) in enumerate(valid_dataloader):
            input, label = input.to('cuda'), label.to('cuda')
            
            predict = model(input)

            label_idx = label.item()
            predict_idx = torch.argmax(predict).item()

            total_num += 1
            if label_idx == predict_idx:
                correct_num += 1

            if len(samples) < num_samples:
                samples.append([input,label_idx,predict_idx])
    
    fig_num = int(num_samples**(0.5))
    fig, axs = plt.subplots(fig_num,
                            fig_num,
                            constrained_layout=True,
                            )
    fig.suptitle(f'Accuracy: {correct_num/total_num}', fontsize=16)

    i,j = 0,0
    for (input, label, predict) in samples:
        axs[j][i].imshow(input[0].cpu().permute(1,2,0))
        axs[j][i].set_title(f'{idx_to_class[label]} - {idx_to_class[predict]}')
        if label != predict:
            rect = patches.Rectangle((0, 0), 31, 31, linewidth=3, edgecolor='r', facecolor='none')
            axs[j][i].add_patch(rect)
        i += 1
        if i >= fig_num:
            j += 1
            i = 0
    plt.show()
    

###

if __name__ == '__main__':
    wandb.init(entity='elfo',
               project='simsiam_birds', 
               name='CIFAR10_task_training'
               )

    train_task(dataset_path           = 'CIFAR10',
               pretrained_weight_path = './models/CIFAR10/pretrain_model_30.pth',
               init_lr                = 5e-4,
               batch_size             = 32,
               shuffle                = True,
               num_workers            = 4,
               num_epochs             = 30,
               device                 = 'cuda',
               )

    validate_task(dataset_path           = 'CIFAR10',
                  pretrained_weight_path = './models/CIFAR10/model_30.pth',
                  batch_size             = 1,
                  num_samples            = 16,
                  device                 = 'cuda',
                  )
    