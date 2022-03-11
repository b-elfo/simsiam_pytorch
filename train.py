import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import SimSiamNet

import wandb

###

def do_train_epoch(dataloader: DataLoader,
                   model: SimSiamNet,
                   loss_func: nn.Module,
                   optim: torch.optim,
                   lr_sched: torch.optim,
                   # scaler: torch.cuda.amp.GradScaler,
                   ):
    model = model.train()
    total_loss = 0
    step = 0
    p1_hist, p2_hist = [], []
    z1_hist, z2_hist = [], []

    for i, (target1, target2) in enumerate(dataloader):
        target1, target2 = target1.to('cuda'), target2.to('cuda')

        # with torch.cuda.amp.autocast():
        z1, p1 = model(target1)
        z2, p2 = model(target2)

        if torch.isnan(z1.sum()+z2.sum()+p1.sum()+p2.sum()):
            print("NaNs found in output\nz1-sum: {}\nz2-sum: {}\np1-sum: {}\np2-sum: {}".format(torch.sum(z1), torch.sum(z2), torch.sum(p1), torch.sum(p2)))

        loss = loss_func(p1,p2,z2,z1)
        if torch.isnan(loss):
            print("NaNs in loss function, loss-sum: {}".format(torch.sum(loss)))
        total_loss += loss
        
        if i%1==0:
            wandb.log({'running_loss':total_loss, 
                       'batch_num':i, 
                       'p1_std':p1.std(dim=1).mean(),
                       'p2_std':p2.std(dim=1).mean(),
                       'z1_std':z1.std(dim=1).mean(),
                       'z2_std':z2.std(dim=1).mean(),
                       })

        # scaler.scale(loss).backward()
        loss.backward()

        # scaler.step(optim)
        optim.step()
        optim.zero_grad()

        if lr_sched:
            lr_sched.step()
        
        if step%1000==0:
            print("Step: {}\tLoss: {:.2f}".format(step, loss))
        step+=1

        # scaler.update()

    return total_loss/len(dataloader)

def do_valid_epoch(dataloader: DataLoader,
                   model: SimSiamNet,
                   loss_func: nn.Module,
                   ):
    model = model.eval()

    with torch.no_grad():
        total_loss = 0

        for i, (target1, target2) in enumerate(dataloader):
            target1, target2 = target1.to('cuda'), target2.to('cuda')
            
            z1, p1 = model(target1)
            z2, p2 = model(target2)

            loss = loss_func(p1,p2,z2,z1)

            total_loss += loss

    return total_loss/len(dataloader)

def do_n_epochs(train_dataloader: DataLoader,
                valid_dataloader: DataLoader,
                model: SimSiamNet,
                loss_func: nn.Module,
                optim: torch.optim,
                lr_sched: torch.optim,
                num_epochs: int = 1,
                ):
    #scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        train_loss = do_train_epoch(dataloader=train_dataloader, 
                                    model=model, 
                                    loss_func=loss_func,
                                    optim=optim,
                                    lr_sched=lr_sched,
                                    #scaler=scaler,
                                    )
        
        valid_loss = do_valid_epoch(dataloader=valid_dataloader,
                                    model=model,
                                    loss_func=loss_func)
        
        wandb.log({'train_loss': train_loss, 'valid_loss': valid_loss}, commit=True)

        print("Epoch: {}\tTrain loss: {:.2f}\tValid loss: {:.2f}\n".format(epoch, train_loss, valid_loss))

        torch.save( obj=model.state_dict(), f=f'model_{epoch+1}.pth' )
