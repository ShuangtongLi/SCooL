import os
import numpy as np
import time
import argparse
from PIL import Image
from mpi4py import MPI
from math import ceil
from random import Random
import networkx as nx
import wandb
import torch
import torch.distributed as dist
import torch.utils.data.distributed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.multiprocessing import Process
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import torchvision.models as models
from torchvision.datasets.utils import check_integrity
import pickle
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import matplotlib
class Recorder(object):
    def __init__(self, args, rank,size, parent_rank_list,comm):
        self.record_accuracy = list()
        self.record_timing = list()
        self.record_comp_timing = list()
        self.record_comm_timing = list()
        self.record_losses = list()
        self.record_trainacc = list()
        self.total_record_timing = list()
        self.args = args
        self.rank = rank
        self.size=size
        self.saveFolderName = args.savePath + args.name + '_' + args.model
        self.parent_rank_list=parent_rank_list
        self.comm=comm
        if rank == 0 and os.path.isdir(self.saveFolderName)==False and self.args.save:
            os.mkdir(self.saveFolderName)

    def add_new(self,top1_group,train_loss_group,test_acc_group,test_loss_group,\
        mixing_weight_group,epoch):
        train_acc_group=self.my_gather([i for i in top1_group],root=0)
        test_acc_group=self.my_gather([i for i in test_acc_group],root=0)
        
        train_loss_group=self.my_gather(train_loss_group,root=0)
        test_loss_group=self.my_gather(test_loss_group,root=0)
        mixing_matrix=self.my_gather(mixing_weight_group,root=0)
        
        
        
       
        if self.rank==0:
            mixing_matrix=mixing_matrix_numpy(mixing_matrix)
            wandb.log({'mixing_matrix'+str(epoch):mixing_matrix})
            wandb_dict=dict()
            #print(len(train_acc_group))
            for i in range(len(train_acc_group)):
                wandb_dict['train_acc'+str(i)]=train_acc_group[i]
                wandb_dict['train_loss'+str(i)]=train_loss_group[i]
                wandb_dict['test_acc'+str(i)]=test_acc_group[i]
                wandb_dict['test_loss'+str(i)]=test_loss_group[i]
                
            wandb_dict['avg_train_accuracy']=sum(train_acc_group)/len(train_acc_group)
            wandb_dict['avg_train_loss']=sum(train_loss_group)/len(train_loss_group)
            wandb_dict['avg_test_acc']=sum(test_acc_group)/len(test_acc_group)
            print('epoch=',epoch,'avg test acc=',sum(test_acc_group)/len(test_acc_group))


            x_labels=[str(i) for i in range(len(mixing_matrix))]
            y_labels=[str(i) for i in range(len(mixing_matrix))]
            #wandb_dict['mixing_weight_heatmap']=[wandb.plots.HeatMap(x_labels, y_labels, mixing_matrix, show_text=False)]
            
            matplotlib.use('Agg')
            fig=plt.figure()
            plt.imshow(mixing_matrix, cmap='Purples')
            plt.colorbar()
            wandb_dict['mixing_weight_heatmap']=wandb.Image(plt)
            wandb_dict['epoch']=epoch

            wandb_dict=add_wandb_dict(wandb_dict,mixing_matrix,name='mixing_weight')
            '''
            for i in range(len(inner_product_matrix)):
                for j in range(len(inner_product_matrix[i])):
                    wandb_dict['inner_product'+str(i)+'_'+str(j)]=inner_product_matrix[i][j]
            '''
                    
            wandb.log(wandb_dict)
            plt.close()
    def preprocess_gather(self,send_buffer_group,parent_rank_list):
        effective_send_buffer_group=[]
        for i in parent_rank_list[self.rank]:
            effective_send_buffer_group.append(send_buffer_group[i])
        return effective_send_buffer_group
    def postprocess_gather(self,recv_buffer_group):
        effective_recv_buffer_group=[]
        for i in recv_buffer_group:
            effective_recv_buffer_group=effective_recv_buffer_group+i
        return effective_recv_buffer_group
    def my_gather(self,send_buffer_group,root):
        effective_send_buffer_group=self.preprocess_gather(send_buffer_group,self.parent_rank_list)
        tmp_recv_buffer_group=self.comm.gather(effective_send_buffer_group,root=root)
        if self.rank==root:
            recv_buffer_group=tmp_recv_buffer_group
            effective_recv_buffer_group=self.postprocess_gather(recv_buffer_group)
            return effective_recv_buffer_group
    




    def save_to_file(self):
        np.savetxt(self.saveFolderName+'/dsgd-lr'+str(self.args.lr)+'-budget'+str(self.args.budget)+'-r'+str(self.rank)+'-recordtime.log', self.total_record_timing, delimiter=',')
        np.savetxt(self.saveFolderName+'/dsgd-lr'+str(self.args.lr)+'-budget'+str(self.args.budget)+'-r'+str(self.rank)+'-time.log',  self.record_timing, delimiter=',')
        np.savetxt(self.saveFolderName+'/dsgd-lr'+str(self.args.lr)+'-budget'+str(self.args.budget)+'-r'+str(self.rank)+'-comptime.log',  self.record_comp_timing, delimiter=',')
        np.savetxt(self.saveFolderName+'/dsgd-lr'+str(self.args.lr)+'-budget'+str(self.args.budget)+'-r'+str(self.rank)+'-commtime.log',  self.record_comm_timing, delimiter=',')
        np.savetxt(self.saveFolderName+'/dsgd-lr'+str(self.args.lr)+'-budget'+str(self.args.budget)+'-r'+str(self.rank)+'-acc.log',  self.record_accuracy, delimiter=',')
        np.savetxt(self.saveFolderName+'/dsgd-lr'+str(self.args.lr)+'-budget'+str(self.args.budget)+'-r'+str(self.rank)+'-losses.log',  self.record_losses, delimiter=',')
        np.savetxt(self.saveFolderName+'/dsgd-lr'+str(self.args.lr)+'-budget'+str(self.args.budget)+'-r'+str(self.rank)+'-tacc.log',  self.record_trainacc, delimiter=',')
        with open(self.saveFolderName+'/ExpDescription', 'w') as f:
            f.write(str(self.args)+ '\n')
            f.write(self.args.description + '\n')
def get_mixing_mat(mixing_tensor_array,rank,size,comm):
    mixing_mat=comm.gather(mixing_tensor_array,root=0)
    mixing_matrix=torch.zeros(size,size)
    if rank==0:
        for i in range(len(mixing_mat)):
            for j in range(len(mixing_mat[i])):
                mixing_matrix[i][j]=mixing_mat[i][j]
    return mixing_matrix
def mixing_matrix_numpy(mixing_mat):
    size=len(mixing_mat)
    #mixing_matrix=torch.zeros(size,size)
    mixing_matrix=np.zeros((size,size))
    
    for i in range(len(mixing_mat)):
        for j in range(len(mixing_mat[i])):
            mixing_matrix[i][j]=mixing_mat[i][j]
    return mixing_matrix
def convert_targets(targets,mask):
    for i in range(len(mask)):
        value=mask[i]
        targets=torch.where(targets==value,i,targets)

    return targets

def test(model, test_loader, mask=None):
    #if mask!=None, mask should be a python list.
    model.eval()
    top1 = AverageMeter()
    loss1 = AverageMeter()
    # correct = 0
    # total = 0
    if mask==None:
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            outputs = model(inputs)
            acc1 = comp_accuracy(outputs, targets)
            cost = F.cross_entropy(outputs,targets)
            top1.update(acc1[0].cpu(), inputs.size(0))
            loss1.update(cost.item(), inputs.size(0))
        return top1.avg,loss1.avg
    else:
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            outputs = model(inputs)
            targets_=convert_targets(targets,mask)
            acc1 = comp_accuracy(outputs[:,mask], targets_)
            cost = F.cross_entropy(outputs,targets)
            top1.update(acc1[0].cpu(), inputs.size(0))
            loss1.update(cost.item(), inputs.size(0))
        return top1.avg,loss1.avg

def comp_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res 
def add_wandb_dict(wandb_dict,inner_product_matrix,name):
    for i in range(len(inner_product_matrix)):
        for j in range(len(inner_product_matrix[i])):
            wandb_dict[name+str(i)+'_'+str(j)]=inner_product_matrix[i][j]
    return wandb_dict
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0 
        self.avg = 0 
        self.sum = 0 
        self.count = 0 

    def update(self, val, n=1):
        self.val = val 
        self.sum += val * n 
        self.count += n
        self.avg = self.sum / self.count
class Recorder1(object):
    def __init__(self, args, rank,size):
        self.record_accuracy = list()
        self.record_timing = list()
        self.record_comp_timing = list()
        self.record_comm_timing = list()
        self.record_losses = list()
        self.record_trainacc = list()
        self.total_record_timing = list()
        self.args = args
        self.rank = rank
        self.size=size
        self.saveFolderName = args.savePath + args.name + '_' + args.model
        if rank == 0 and os.path.isdir(self.saveFolderName)==False and self.args.save:
            os.mkdir(self.saveFolderName)

    def add_new(self,comm,record_time,comp_time,comm_time,\
        epoch_time,top1,losses,test_acc,epoch,mixing_tensor_array,\
        test_loss,val_acc,val_loss,inner_product_tensor_array,\
        record_val_loss_group,record_val_acc_group):
        self.total_record_timing.append(record_time)
        self.record_timing.append(epoch_time)
        self.record_comp_timing.append(comp_time)
        self.record_comm_timing.append(comm_time)
        self.record_trainacc.append(top1.cpu())
        self.record_losses.append(losses)
        self.record_accuracy.append(test_acc.cpu())
        train_acc_group=comm.gather(top1.cpu(),root=0)
        test_acc_group=comm.gather(test_acc.cpu(),root=0)
        val_acc_group=comm.gather(val_acc.cpu(),root=0)
        train_loss_group=comm.gather(losses,root=0)
        test_loss_group=comm.gather(test_loss,root=0)
        val_loss_group=comm.gather(val_loss,root=0)



        val_loss_group_iter=[]
        for i in range(len(record_val_loss_group)):
            val_loss_iter=record_val_loss_group[i]
            val_loss_group_iter.append(comm.gather(val_loss_iter,root=0))



        val_acc_group_iter=[]
        for i in range(len(record_val_acc_group)):
            val_acc_iter=record_val_acc_group[i].cpu()
            val_acc_group_iter.append(comm.gather(val_acc_iter,root=0))




        if mixing_tensor_array==None:
            if self.rank==0:
                wandb_dict=dict()
                for i in range(self.size):
                    wandb_dict['train_acc'+str(i)]=train_acc_group[i]
                    wandb_dict['train_loss'+str(i)]=train_loss_group[i]
                    wandb_dict['test_acc'+str(i)]=test_acc_group[i]
                    wandb_dict['test_loss'+str(i)]=test_loss_group[i]
                    wandb_dict['val_acc'+str(i)]=val_acc_group[i]
                    wandb_dict['val_loss'+str(i)]=val_loss_group[i]
                wandb_dict['avg_train_accuracy']=sum(train_acc_group)/len(train_acc_group)
                wandb_dict['avg_train_loss']=sum(train_loss_group)/len(train_loss_group)
                wandb_dict['avg_test_acc']=sum(test_acc_group)/len(test_acc_group)
                wandb_dict['epoch']=epoch
                wandb.log(wandb_dict)
                print('epoch=',epoch,'avg test acc=',sum(test_acc_group)/len(test_acc_group))
        else:
            if self.rank==0:
                wandb_dict=dict()





                for m in range(len(val_loss_group_iter)):
                    for i in range(len(val_loss_group_iter[m])):
                        wandb_dict['val_acc'+str(i)+'iter'+str(m)]=val_acc_group_iter[m][i]
                        wandb_dict['val_loss'+str(i)+'iter'+str(m)]=val_loss_group_iter[m][i]







                for i in range(self.size):
                    wandb_dict['train_acc'+str(i)]=train_acc_group[i]
                    wandb_dict['train_loss'+str(i)]=train_loss_group[i]
                    wandb_dict['test_acc'+str(i)]=test_acc_group[i]
                    wandb_dict['test_loss'+str(i)]=test_loss_group[i]
                    wandb_dict['val_acc'+str(i)]=val_acc_group[i]
                    wandb_dict['val_loss'+str(i)]=val_loss_group[i]
                wandb_dict['avg_train_accuracy']=sum(train_acc_group)/len(train_acc_group)
                wandb_dict['avg_train_loss']=sum(train_loss_group)/len(train_loss_group)
                wandb_dict['avg_test_acc']=sum(test_acc_group)/len(test_acc_group)
                print('epoch=',epoch,'avg test acc=',sum(test_acc_group)/len(test_acc_group))
            mixing_matrix=get_mixing_mat(mixing_tensor_array,self.rank,self.size,comm)
            inner_product_matrix=get_mixing_mat(inner_product_tensor_array,self.rank,self.size,comm)
            if self.rank==0:
                x_labels=[str(i) for i in range(len(mixing_matrix))]
                y_labels=[str(i) for i in range(len(mixing_matrix))]
                #wandb_dict['mixing_weight_heatmap']=[wandb.plots.HeatMap(x_labels, y_labels, mixing_matrix, show_text=False)]
                
                matplotlib.use('Agg')
                fig=plt.figure()

                plt.imshow(mixing_matrix, cmap='Purples')
                #plt.imshow(mixing_matrix, cmap='viridis')
                #plt.clim(0, 1)
                plt.colorbar()
                wandb_dict['mixing_weight_heatmap']=wandb.Image(plt)
                wandb_dict['epoch']=epoch

                wandb_dict=add_wandb_dict(wandb_dict,inner_product_matrix,name='inner_product')
                wandb_dict=add_wandb_dict(wandb_dict,mixing_matrix,name='mixing_weight')
                '''
                for i in range(len(inner_product_matrix)):
                    for j in range(len(inner_product_matrix[i])):
                        wandb_dict['inner_product'+str(i)+'_'+str(j)]=inner_product_matrix[i][j]
                '''
                        
                wandb.log(wandb_dict)
                plt.close()


    def save_to_file(self):
        np.savetxt(self.saveFolderName+'/dsgd-lr'+str(self.args.lr)+'-budget'+str(self.args.budget)+'-r'+str(self.rank)+'-recordtime.log', self.total_record_timing, delimiter=',')
        np.savetxt(self.saveFolderName+'/dsgd-lr'+str(self.args.lr)+'-budget'+str(self.args.budget)+'-r'+str(self.rank)+'-time.log',  self.record_timing, delimiter=',')
        np.savetxt(self.saveFolderName+'/dsgd-lr'+str(self.args.lr)+'-budget'+str(self.args.budget)+'-r'+str(self.rank)+'-comptime.log',  self.record_comp_timing, delimiter=',')
        np.savetxt(self.saveFolderName+'/dsgd-lr'+str(self.args.lr)+'-budget'+str(self.args.budget)+'-r'+str(self.rank)+'-commtime.log',  self.record_comm_timing, delimiter=',')
        np.savetxt(self.saveFolderName+'/dsgd-lr'+str(self.args.lr)+'-budget'+str(self.args.budget)+'-r'+str(self.rank)+'-acc.log',  self.record_accuracy, delimiter=',')
        np.savetxt(self.saveFolderName+'/dsgd-lr'+str(self.args.lr)+'-budget'+str(self.args.budget)+'-r'+str(self.rank)+'-losses.log',  self.record_losses, delimiter=',')
        np.savetxt(self.saveFolderName+'/dsgd-lr'+str(self.args.lr)+'-budget'+str(self.args.budget)+'-r'+str(self.rank)+'-tacc.log',  self.record_trainacc, delimiter=',')
        with open(self.saveFolderName+'/ExpDescription', 'w') as f:
            f.write(str(self.args)+ '\n')
            f.write(self.args.description + '\n')
def preprocess_gather(send_buffer_group,parent_rank_list,rank):
    effective_send_buffer_group=[]
    for i in parent_rank_list[rank]:
        effective_send_buffer_group.append(send_buffer_group[i])
    return effective_send_buffer_group
def postprocess_gather(recv_buffer_group):
    effective_recv_buffer_group=[]
    for i in recv_buffer_group:
        effective_recv_buffer_group=effective_recv_buffer_group+i
    return effective_recv_buffer_group
def my_gather(send_buffer_group,root,rank,parent_rank_list,comm):
    effective_send_buffer_group=preprocess_gather(send_buffer_group,parent_rank_list,rank)
    tmp_recv_buffer_group=comm.gather(effective_send_buffer_group,root=root)
    if rank==root:
        recv_buffer_group=tmp_recv_buffer_group
        effective_recv_buffer_group=postprocess_gather(recv_buffer_group)
        return effective_recv_buffer_group

def get_true_correlation(send_buffer_group,rank,parent_rank_list,comm):
    mixing_mat=my_gather(send_buffer_group,0,rank,parent_rank_list,comm)
    
    if rank==0:
        size=len(mixing_mat)
        mixing_matrix=torch.zeros(size,size)
        mixing_mat=np.array(mixing_mat)
        

        for i in range(len(mixing_matrix)):
            for j in range(len(mixing_matrix[i])):
                set1=set(mixing_mat[i])
                set2=set(mixing_mat[j])
                intersection_set=set1.intersection(set2)
                intersection_num=len(intersection_set)
                mixing_matrix[i][j]=intersection_num
        print(mixing_matrix)
        
        matplotlib.use('Agg')
        fig=plt.figure()
        plt.imshow(mixing_matrix, cmap='Purples')
        plt.colorbar()
        wandb_dict=dict()
        wandb_dict['truth_heatmap']=wandb.Image(plt)
        wandb.log(wandb_dict)

def test_fast(model, test_loader, mask=None):
    #if mask!=None, mask should be a python list.
    model.eval()
    top1 = AverageMeter()
    loss1 = AverageMeter()
    # correct = 0
    # total = 0
    if mask==None:
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            outputs = model(inputs)
            acc1 = comp_accuracy(outputs, targets)
            cost = F.cross_entropy(outputs,targets)
            top1.update(acc1[0].cpu(), inputs.size(0))
            loss1.update(cost.item(), inputs.size(0))
            break
        return top1.avg,loss1.avg
    else:
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            outputs = model(inputs)
            targets_=convert_targets(targets,mask)
            acc1 = comp_accuracy(outputs[:,mask], targets_)
            cost = F.cross_entropy(outputs,targets)
            top1.update(acc1[0].cpu(), inputs.size(0))
            loss1.update(cost.item(), inputs.size(0))
            break
        return top1.avg,loss1.avg