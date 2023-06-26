import os
import numpy as np
import time
import argparse
import sys
import pdb
import copy
import datetime
from mpi4py import MPI
from math import ceil
from random import Random
import networkx as nx

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
cudnn.benchmark = True
from models import resnet,resnet1
import comm_helpers

def select_top_faction_mixing_weight_mask(args, child_rank_num, mixing_weight):
    index_group=np.argsort(mixing_weight)
    left_num=int(child_rank_num*args.mask_fraction)
    print('left_num=',left_num)
    index_group=index_group[-left_num:]
    index_group=index_group.tolist()
    return index_group

def mixing_weight_before_mask(child_rank_num, mixing_weight,mask=None):
    if mask==None:
        return mixing_weight
    else:
        mixing_weight1=np.zeros(child_rank_num)
        for i in range(len(mask)):
            mixing_weight1[mask[i]]=mixing_weight[i]
        return mixing_weight1

def check_model_params_var(model):
    param_var_sum=0
    param_mean_sum=0
    for i in model.params():
        m=i.clone().detach().data
        m_var=torch.var(m)
        param_var_sum=param_var_sum+m_var
        m_mean=torch.mean(m)
        param_mean_sum=param_mean_sum+m_mean
    return param_var_sum
def check_model_params_mean(model):
    param_var_sum=0
    param_mean_sum=0
    for i in model.params():
        m=i.clone().detach().data
        m_var=torch.var(m)
        param_var_sum=param_var_sum+m_var
        m_mean=torch.mean(m)
        param_mean_sum=param_mean_sum+m_mean
    return param_mean_sum
def smoothing_loss(current_mixing,momentum_mixing):
    return torch.norm(current_mixing-momentum_mixing)
class MixingMomentum(object):
    def __init__(self,size):
        self.size=size
        self.sum_mixing=torch.zeros(self.size).cuda()
        self.num=0
    def update(self,mixing_weight):
        mixing_weight=mixing_weight.detach()
        self.sum_mixing=mixing_weight+self.sum_mixing
        self.num=self.num+1
        new_mixing=self.sum_mixing/float(self.num)
        return new_mixing
class MixingExponential(object):
    def __init__(self,size,momentum=0.99):
        self.size=size
        self.mixing_buffer=torch.zeros(self.size).cuda()
        self.num=0
        self.momentum=momentum
    def update(self,mixing_weight):
        mixing_weight=mixing_weight.detach()
        if self.num==0:
            self.mixing_buffer=mixing_weight
            self.num=self.num+1
            return mixing_weight
        else:
            self.mixing_buffer=self.momentum*self.mixing_buffer+(1-self.momentum)*mixing_weight
            self.num=self.num+1
            return self.mixing_buffer
        return new_mixing

def get_gradient_norm2(grad_group,val_grad):
    new_grad_group=[comm_helpers.flatten_tensors(i).detach() for i in grad_group]
    key_tensor=comm_helpers.flatten_tensors(val_grad).detach()

    '''
    new_grad_group=[i/torch.norm(i) for i in new_grad_group]
    key_tensor=key_tensor/torch.norm(key_tensor)
    '''
    
    
    inner_product_group=[torch.sum(i*key_tensor) for i in new_grad_group]
    inner_product_group=torch.tensor(inner_product_group)
    #inner_product_group=inner_product_group/torch.norm(inner_product_group)
    #print('rank=',self.rank,inner_product_group)
    #note that inner_product_group does not have grad_fn after torch.tensor op.
    #print(inner_product_group)
    '''
    gradient_inner_product=util.get_mixing_mat(inner_product_group.view(-1),self.rank,self.size,self.comm)
    util.recorder_fast(gradient_inner_product,self.rank,name='val_prod')
    '''
    return inner_product_group.view(-1)
def zero_grad_params(model,set_to_none: bool = False):
    for p in model.params():
        if p.grad is not None:
            if set_to_none:
                p.grad = None
            else:
                if p.grad.grad_fn is not None:
                    p.grad.detach_()
                else:
                    p.grad.requires_grad_(False)
                p.grad.zero_()
def next_iter(loader):
    while(1):
        for i in loader:
            yield i

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        lr=param_group['lr']
    return lr
def get_two_model_params_subtract(model1,model2):
    param_list=[]
    with torch.no_grad():
        for param1,param2 in zip(model1.params(),model2.params()):
            param1_data=param1.clone().detach().data
            param2_data=param2.clone().detach().data
            param_list.append(param1_data-param2_data)
    return param_list


def build_model(args, device_name='cpu'):
    if args.dataset=='cifar10_tiny':
        num_classes=10
    if args.dataset=='cifar100_tiny':
        num_classes=100
    if args.dataset=='miniimagenet':
        num_classes=100
    # model = ResNet32(args.dataset == 'cifar10' and 10 or 100)
    #model = resnet.ResNet18(10//args.split_label_group_num)
    assert (device_name=='cpu' or device_name=='cuda')
    if device_name=='cpu':
        if args.model_arch=='CNN':
            model = resnet1.CNNCifar1(num_classes)
        elif args.model_arch=='resnet18':
            model=resnet1.ResNet18(num_classes)
        elif args.model_arch=='CNN_miniimagenet':
            model=resnet1.CNNminiimagenet(num_classes)
        else:
            raise NotImplementedError
    else:
        if args.model_arch=='CNN':
            model=resnet.CNNCifar1(num_classes)
        elif args.model_arch=='resnet18':
            model=resnet.ResNet18(num_classes)
        elif args.model_arch=='CNN_miniimagenet':
            model=resnet1.CNNminiimagenet(num_classes)
        else:
            raise NotImplementedError
    return model


def build_mixing_model(args, comm, child_rank_num, device_name,ref_model):
    
    assert (device_name=='cpu' or device_name=='cuda')
    if device_name=='cpu':
        if args.mixing_model_arch=='layerwise':
            model=resnet.Attention_layerwise_model_update_multilayer_mask(\
            ref_model,10, 0, comm,child_rank_num)
        if args.mixing_model_arch=='dot':
            model=resnet.Attention_dot_model_update_mask(\
            ref_model,10, 0, comm,child_rank_num)
        if args.mixing_model_arch=='layerwise_rand':
            model=resnet.Attention_layerwise_model_update_multilayer_mask_randchoose_channel(\
            ref_model,10, 0, comm,child_rank_num,device_name,args.choose_filter_num)
        if args.mixing_model_arch=='gcn':
            model=resnet.GCN(child_rank_num)
    else:
        if args.mixing_model_arch=='layerwise':
            model=resnet.Attention_layerwise_model_update_multilayer_mask(\
            ref_model,10, 0, comm,child_rank_num).cuda()
        if args.mixing_model_arch=='dot':
            model=resnet.Attention_dot_model_update_mask(\
            ref_model,10, 0, comm,child_rank_num).cuda()
        if args.mixing_model_arch=='layerwise_rand':
            model=resnet.Attention_layerwise_model_update_multilayer_mask_randchoose_channel(\
            ref_model,10, 0, comm,child_rank_num,device_name,args.choose_filter_num)
        if args.mixing_model_arch=='gcn':
            model=resnet.GCN(child_rank_num)
    return model
def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)
def load_gradient_to_model(model,grad):
    with torch.no_grad():
        for f, t in zip(grad, 
                        model.params()): 
            t.grad=f.clone().detach().data

def combinate_gradients(mixing_weight,grad_group,requires_grad=True,mask=None):
    
    if mask!=None:
        grad_group=[grad_group[i] for i in mask]
    assert len(mixing_weight)==len(grad_group)
    if requires_grad==True:
        new_grad=[0 for i in range(len(grad_group[0]))]
        for i in range(len(mixing_weight)):
            new_grad=[new_grad[j]+mixing_weight[i]*grad_group[i][j] for j in range(len(grad_group[i]))]
        return new_grad
    else:
        with torch.no_grad():
            new_grad=[0 for i in range(len(grad_group[0]))]
            for i in range(len(mixing_weight)):
                new_grad=[new_grad[j]+mixing_weight[i]*grad_group[i][j] for j in range(len(grad_group[i]))]
            return new_grad
def combinate_gradients_cpu(mixing_weight,grad_group,requires_grad=True,mask=None):
    anneal_lr = 1.
    if mask!=None:
        #anneal_lr = float(float(len(mask)) / float(len(grad_group)))
        grad_group=[grad_group[i] for i in mask]
    assert len(mixing_weight)==len(grad_group)
    if requires_grad==True:
        new_grad=[0 for i in range(len(grad_group[0]))]
        for i in range(len(mixing_weight)):
            new_grad=[new_grad[j]+mixing_weight[i]*grad_group[i][j].clone().detach().data.cpu() for j in range(len(grad_group[i]))]
        return new_grad
    else:
        with torch.no_grad():
            new_grad=[0 for i in range(len(grad_group[0]))]
            for i in range(len(mixing_weight)):
                new_grad=[new_grad[j]+mixing_weight[i]*anneal_lr*grad_group[i][j].clone().detach().data.cpu() for j in range(len(grad_group[i]))]
            return new_grad
def sync_init(comm, parent_rank_list, model_group, rank, size):
    layer_num=0
    for i in model_group[parent_rank_list[rank][0]].params():
        layer_num=layer_num+1
    senddata = [0.0 for i in range(layer_num)]
    recvdata = [[] for i in range(layer_num)]
    
    
    model=model_group[parent_rank_list[rank][0]]
    count=0
    for param in model.params():
        if rank==0:
            tmp = param.clone().detach().data.cpu().numpy()
            recvdata[count]=tmp
            
        else:
            recvdata[count]=None

        count=count+1

  
    torch.cuda.synchronize()
    comm.barrier()

    comm_start = time.time()
    for count in range(layer_num):
        print(count)
        recvdata[count]=comm.bcast(recvdata[count],root=0)
    torch.cuda.synchronize()    
    comm.barrier()
    
    comm_end = time.time()
    comm_t = (comm_end - comm_start)
    for child_rank_id in parent_rank_list[rank]:
        model=model_group[child_rank_id]   
        count=0
        for param in model.params():
            param.data = torch.Tensor(recvdata[count]).clone().detach().data
            count=count+1
    return comm_t
def sync_init1(comm, parent_rank_list, model_group, rank, size):
    layer_num=0
    for i in model_group[parent_rank_list[rank][0]].parameters():
        layer_num=layer_num+1
    senddata = [0.0 for i in range(layer_num)]
    recvdata = [[] for i in range(layer_num)]
    
    
    model=model_group[parent_rank_list[rank][0]]
    count=0
    for param in model.parameters():
        if rank==0:
            tmp = param.clone().detach().data.cpu().numpy()
            recvdata[count]=tmp
            
        else:
            recvdata[count]=None

        count=count+1

  
    torch.cuda.synchronize()
    comm.barrier()

    comm_start = time.time()
    for count in range(layer_num):
        print(count)
        recvdata[count]=comm.bcast(recvdata[count],root=0)
    torch.cuda.synchronize()    
    comm.barrier()
    
    comm_end = time.time()
    comm_t = (comm_end - comm_start)
    for child_rank_id in parent_rank_list[rank]:
        model=model_group[child_rank_id]   
        count=0
        for param in model.parameters():
            param.data = torch.Tensor(recvdata[count]).clone().detach().data
            count=count+1
    return comm_t


def sync_allreduce1(comm, parent_rank_list, child_rank_num, model_group, rank, size):
    layer_num=0
    for i in model_group[parent_rank_list[rank][0]].parameters():
        layer_num=layer_num+1
    senddata = [0.0 for i in range(layer_num)]
    recvdata = [[] for i in range(layer_num)]
    for child_rank_id in parent_rank_list[rank]:
        model=model_group[child_rank_id]
        count=0
        for param in model.parameters():
            tmp = param.clone().detach().data.cpu().numpy()
            senddata[count] = senddata[count]+tmp
            recvdata[count] = np.empty(senddata[count].shape, dtype = senddata[count].dtype)
            count=count+1
    torch.cuda.synchronize()
    comm.barrier()

    comm_start = time.time()
    for count in range(layer_num):
        comm.Allreduce(senddata[count], recvdata[count], op=MPI.SUM)
    torch.cuda.synchronize()    
    comm.barrier()
    
    comm_end = time.time()
    comm_t = (comm_end - comm_start)
    for child_rank_id in parent_rank_list[rank]:
        model=model_group[child_rank_id]   
        count=0
        for param in model.parameters():
            param_data = torch.Tensor(recvdata[count]).clone().detach().data
            param.data = param_data/float(child_rank_num)
            count=count+1
    return comm_t
def cpu_model_params_to_cuda_params(model,model1):
    with torch.no_grad():
        for (i,j) in zip(model.params(),model1.params()):
            j.data=i.clone().detach().data
            j.data=j.data.cuda()
def cpu_model_params_to_cuda_params1(model,model1):
    with torch.no_grad():
        for (i,j) in zip(model.parameters(),model1.parameters()):
            j.data=i.clone().detach().data
            j.data=j.data.cuda()
def cpu_model_grad_from_cuda_grad(model,model_grad):
    with torch.no_grad():
        for (i,j) in zip(model.params(),model_grad):
            grad_data=j.clone().detach().data
            grad_data=grad_data.cpu()
            i.grad=grad_data
def cpu_model_grad_from_cuda_grad1(model,model_grad):
    with torch.no_grad():
        for (i,j) in zip(model.parameters(),model_grad):
            grad_data=j.clone().detach().data
            grad_data=grad_data.cpu()
            i.grad=grad_data