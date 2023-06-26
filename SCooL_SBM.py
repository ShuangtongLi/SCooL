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
from models import vggnet
from models import wrn 
import util
import communicator_slow
from gen_graph import gen_graph
import util1
import comm_helpers
import util2
import SBM
from training_utils import *

def run(rank, size):

    # set random seed
    torch.manual_seed(args.randomSeed+rank)
    np.random.seed(args.randomSeed)

    # load data
    if args.train_ratio==None:
        train_loader_group=[[] for i in range(child_rank_num)]
        test_loader_group=[[] for i in range(child_rank_num)]
        mask_group=[[] for i in range(child_rank_num)]
        for child_rank_id in parent_rank_list[rank]:
            if args.experiment_setting == "non-iid":
                train_loader, test_loader, mask = \
                util.partition_non_iid_random_dataset(child_rank_id, child_rank_num, args)
            elif args.experiment_setting == "non-iid_SBM":
                train_loader, test_loader, mask = \
                util.partition_non_iid_determine_dataset(child_rank_id, child_rank_num, args)
            else:
                raise NotImplementedError
            
            train_loader_group[child_rank_id]=train_loader
            test_loader_group[child_rank_id]=test_loader
            mask_group[child_rank_id]=mask
    else:  
        raise NotImplementedError

    np.savetxt("truth_correlations.txt", np.array(mask_group))
    util1.get_true_correlation(mask_group,rank,parent_rank_list,comm)

    train_loader_iter_group=[[] for i in range(child_rank_num)]
    for child_rank_id in parent_rank_list[rank]:
        train_loader=train_loader_group[child_rank_id]
        train_loader_iter_group[child_rank_id]=next_iter(train_loader)

    # load base network topology
    #subGraphs = util.select_graph(args.graphid)
    
    # define graph activation scheme
    GP=gen_graph(size,args)
    print(GP)

    
    communicator=communicator_slow.GradientCommunicator(rank,size,GP,comm,parent_rank_list)

    # select neural network model
    model_group=[[] for i in range(child_rank_num)]
    for child_rank_id in parent_rank_list[rank]:
        model_group[child_rank_id]=build_model(args, device_name='cpu')
   

    criterion = nn.CrossEntropyLoss().cuda()
    #optimizer_a=torch.optim.Adam(model.params(),lr=args.lr, betas=(0.90, 0.999))
    

    optimizer_a_group=[[] for i in range(child_rank_num)]
    for child_rank_id in parent_rank_list[rank]:
        optimizer_a_group[child_rank_id]=optim.SGD(model_group[child_rank_id].params(), 
                              lr=args.lr,
                              momentum=args.momentum, 
                              weight_decay=args.weight_decay,
                              nesterov=args.nesterov)
    
    virtual_model_group=[[] for i in range(child_rank_num)]
    for child_rank_id in parent_rank_list[rank]:
        virtual_model_group[child_rank_id]=build_model(args, device_name='cpu')

    
    
    
    
    torch.manual_seed(args.randomSeed+rank)
    np.random.seed(args.randomSeed)
    








    var_before=[check_model_params_var(model_group[i]) for i in parent_rank_list[rank]]
    mean_before=[check_model_params_mean(model_group[i]) for i in parent_rank_list[rank]]
    print('before var=',var_before,'mean=',mean_before)
    sync_init(comm, parent_rank_list, model_group, rank, size)
    var_after=[check_model_params_var(model_group[i]) for i in parent_rank_list[rank]]
    mean_after=[check_model_params_mean(model_group[i]) for i in parent_rank_list[rank]]
    print('after var=',var_after,'mean=',mean_after)











    
    
    model_init=build_model(args, device_name='cpu')
    model_init.load_state_dict(copy.deepcopy(model_group[parent_rank_list[rank][0]].state_dict()))
    model_on_cuda=build_model(args, device_name='cuda')
    mixing_net_on_cuda=build_mixing_model(args, comm, child_rank_num, device_name='cuda',ref_model=model_group[child_rank_id])
   

    recorder = util1.Recorder(args,rank,size,parent_rank_list,comm)

    #print('parent_rank_list=',parent_rank_list)
    losses_group=[[] for i in range(child_rank_num)]
    top1_group=[[] for i in range(child_rank_num)]
    val_losses_group=[[] for i in range(child_rank_num)]
    val_top1_group=[[] for i in range(child_rank_num)]
    test_acc_group=[[] for i in range(child_rank_num)]
    test_loss_group=[[] for i in range(child_rank_num)]
    

    gradient_rank_group=[[] for i in range(child_rank_num)]
    model_update_rank_group=[[] for i in range(child_rank_num)]
    mixing_weight_group=[[] for i in range(child_rank_num)]
    mixing_weight_mask_group=[[] for i in range(child_rank_num)]
    for child_rank_id in parent_rank_list[rank]:
        mixing_weight_mask_group[child_rank_id]=None
    tic = time.time()
    

    
    for epoch in range(args.epoch):
        for child_rank_id in parent_rank_list[rank]:
            losses = util.AverageMeter()
            top1 = util.AverageMeter()
            model = model_group[child_rank_id]
            model_on_cuda.train()
            virtual_model=virtual_model_group[child_rank_id]
            virtual_model.load_state_dict(copy.deepcopy(model.state_dict()))
            # Start training each epoch
            train_loader_iter=train_loader_iter_group[child_rank_id]
            optimizer_a=optimizer_a_group[child_rank_id]
            for batch_idx in range(args.communicate_iteration):
                cpu_model_params_to_cuda_params(model,model_on_cuda)
                input,target=next(train_loader_iter)
                '''
                if child_rank_id==9:
                    print(target)
                '''

                
                input_var = to_var(input, requires_grad=False)
                target_var = to_var(target, requires_grad=False)
                #input_validation, target_validation = next(iter(train_loader))
                
                
                model_on_cuda.train()
                y_f_hat = model_on_cuda(input_var)
                cost = F.cross_entropy(y_f_hat, target_var)


                acc1 = util.comp_accuracy(y_f_hat, target_var)
                #print('rank=',rank,'child id=',child_rank_id)
                #print(losses_group[child_rank_id])
                losses.update(cost.item(), input.size(0))
                top1.update(acc1[0].cpu().numpy(), input.size(0))

                #print(model.state_dict())
                
                #record_optimizer_state_dict=optimizer_a.state_dict()
                model_gradient=torch.autograd.grad(cost,model_on_cuda.params())

                cpu_model_grad_from_cuda_grad(model,model_gradient)
                optimizer_a.step()
            
            gradient=get_two_model_params_subtract(virtual_model,model)
            gradient_rank_group[child_rank_id]=gradient
            model_update=get_two_model_params_subtract(model_init,model)
            model_update_rank_group[child_rank_id]=model_update
            losses_group[child_rank_id]=losses.avg
            top1_group[child_rank_id]=top1.avg


        train_acc_mat = []
        train_loss_mat = []
        for child_rank_id in parent_rank_list[rank]:
            tmp_acc_mat = []
            tmp_loss_mat = []
            model = model_group[child_rank_id]
            cpu_model_params_to_cuda_params(model,model_on_cuda)
            for neighbour_rank_id in parent_rank_list[rank]:
                test_loader = train_loader_group[neighbour_rank_id]
                mask = mask_group[neighbour_rank_id]
                test_acc,test_loss = util1.test_fast(model_on_cuda, test_loader, mask)
                #print(test_acc)
                tmp_acc_mat.append(test_acc)
                tmp_loss_mat.append(test_loss)
            train_acc_mat.append(tmp_acc_mat)
            train_loss_mat.append(tmp_loss_mat)
            #print("finish epoch = " + str(epoch) + " " + str(child_rank_id))
        #util2.saveMatrixToLocalLog(train_acc_mat, epoch, "TrainAcc", util2Path)
        #util2.saveMatrixToLocalLog(train_loss_mat, epoch, "TrainLoss", util2Path)
        print(train_acc_mat)
        print(train_loss_mat)

        train_loss_mat = np.array(train_loss_mat)
        def convertAccMat(mixingMat):
            factor = 1e-6
            Y = -mixingMat
            return Y
        
        def filter_mixingMat(mixingMat, mixing_weight_mask_group):
            if mixing_weight_mask_group[0] == None:
                return mixingMat
            else:
                result = np.zeros_like(mixingMat)
                for i in range(len(mixing_weight_mask_group)):
                    for ind in mixing_weight_mask_group[i]:
                        result[i][ind] = mixingMat[i][ind]
                return result


        Y = convertAccMat(train_loss_mat)
        Y = filter_mixingMat(Y, mixing_weight_mask_group)

        k = 2
        SBM_model = SBM.SBM(Y, k, scale = args.scale)
        SBM_model.variational_inference()
        mixing_weight_SBM = SBM_model.output_W()

        
        grad_group=communicator.communicate(gradient_rank_group)
        #print(len(grad_group))
        

     
        

        for child_rank_id in parent_rank_list[rank]:
            model=model_group[child_rank_id]
            virtual_model=virtual_model_group[child_rank_id]
            model.load_state_dict(copy.deepcopy(virtual_model.state_dict()))
            mixing_net_on_cuda.rank=child_rank_id
            mixing_weight_mask=mixing_weight_mask_group[child_rank_id]
            mixing_weight = mixing_weight_SBM[child_rank_id]

            if epoch == args.fix_graph_epoch:
                new_mask=select_top_faction_mixing_weight_mask(args, child_rank_num, mixing_weight)
                mixing_weight_mask_group[child_rank_id] = new_mask

            def mixing_weight_to_masked(mixing_weight, mixing_weight_mask = None):
                if mixing_weight_mask == None:
                    return mixing_weight
                new_mixing_weight = []
                for ind in mixing_weight_mask:
                    new_mixing_weight.append(mixing_weight[ind])
                new_mixing_weight = np.array(new_mixing_weight)
                #new_mixing_weight = new_mixing_weight / np.sum(new_mixing_weight)
                return new_mixing_weight

            mixing_weight = mixing_weight_to_masked(mixing_weight, mixing_weight_mask)
            mixing_weight = mixing_weight / np.sum(mixing_weight)






            '''
            def sampling_mixing_weights(mixing_weight):
                left_num=int(child_rank_num*args.mask_fraction)
                chosen_inds = np.random.choice(len(mixing_weight), left_num, replace=False, p=mixing_weight)
                for mixing_ind in range(len(mixing_weight)):
                    if mixing_ind not in chosen_inds:
                        mixing_weight[mixing_ind] = 0.
                return mixing_weight
            mixing_weight = sampling_mixing_weights(mixing_weight)
            '''









            new_grad=combinate_gradients_cpu(mixing_weight,grad_group,\
                requires_grad=False,mask=mixing_weight_mask)
             
            
            with torch.no_grad():
                for model_param,model_grad in zip(model.params(),new_grad):
                    model_param.data=model_param.clone().detach().data-model_grad.clone().detach().data
            #print('child_rank_id=',child_rank_id,'mixing=',mixing_weight)
            mixing_weight_group[child_rank_id]=\
            mixing_weight_before_mask(child_rank_num, mixing_weight,mixing_weight_mask)
        
        
        

        
       
        # evaluate test accuracy at the end of each epoch
        for child_rank_id in parent_rank_list[rank]:
            model=model_group[child_rank_id]
            test_loader=test_loader_group[child_rank_id]
            mask=mask_group[child_rank_id]
            cpu_model_params_to_cuda_params(model,model_on_cuda)
            test_acc,test_loss= util1.test(model_on_cuda, test_loader, mask)
            #print(test_acc)
            test_acc_group[child_rank_id]=test_acc
            test_loss_group[child_rank_id]=test_loss
            
        recorder.add_new(top1_group,losses_group,test_acc_group,test_loss_group,\
        mixing_weight_group,epoch)

def split_device(process_num,gpu_array,rank_id):
    gpu_num=len(gpu_array)
    rank_array=np.arange(process_num)
    rank_group=np.split(rank_array,gpu_num)
    for i in range(gpu_num):
        if rank_id in rank_group[i]:
            return i
        
def get_parent_rank_list():
    child_rank_array=np.arange(child_rank_num)
    parent_rank_list=np.array_split(child_rank_array,size)
    return parent_rank_list

def get_child_rank_list(parent_rank_list):
    child_rank_list=[]
    for i in range(child_rank_num):
        for j in range(len(parent_rank_list)):
            if i in parent_rank_list[j]:
                break
        child_rank_list.append(j)
    return child_rank_list

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--name','-n', default="default", type=str, help='experiment name')
    parser.add_argument('--description', type=str, help='experiment description')

    parser.add_argument('--model', default="res", type=str, help='model name: res/VGG/wrn')
    parser.add_argument('--lr', default=0.8, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.0, type=float, help='momentum')
    parser.add_argument('--epoch', '-e', default=10, type=int, help='total epoch')
    parser.add_argument('--bs', default=64, type=int, help='batch size on each worker')
    parser.add_argument('--warmup', action='store_true', help='use lr warmup or not')
    parser.add_argument('--nesterov', action='store_true', help='use nesterov momentum or not')

    parser.add_argument('--matcha', action='store_true', help='use MATCHA or not')
    parser.add_argument('--budget', type=float, help='comm budget')
    parser.add_argument('--graphid', default=0, type=int, help='the idx of base graph')
    
    parser.add_argument('--dataset', default='cifar10', type=str, help='the dataset')
    parser.add_argument('--datasetRoot', type=str, help='the path of dataset')
    parser.add_argument('--p', '-p', action='store_true', help='partition the dataset or not')
    parser.add_argument('--savePath' ,type=str, help='save path')
    
    parser.add_argument('--compress', action='store_true', help='use chocoSGD or not')    
    parser.add_argument('--consensus_lr', default=0.1, type=float, help='consensus_lr')
    parser.add_argument('--randomSeed', type=int, help='random seed')
    parser.add_argument('--save', action='store_true', help='save or not')
    parser.add_argument('--every_label_split_num', default=10, type=int)
    parser.add_argument('--split_label_group_num', default=10, type=int)
    parser.add_argument('--train_ratio', default=None, type=float)
    parser.add_argument('--val_ratio', default=None, type=float)
    parser.add_argument('--graph_name', default='single', type=str, help='graph type')
    parser.add_argument('--communicator', default='', type=str, help='communicator function')
    parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
    parser.add_argument('--mixing_weight_decay', default=5e-4, type=float)
    parser.add_argument('--mixing_momentum', default=0.0, type=float)
    parser.add_argument('--mixing_nesterov', action='store_true')
    parser.add_argument('--mixing_lr', default=0.1, type=float, help='mixing learning rate')
    parser.add_argument('--mixing_updates', default=1, type=int)
    parser.add_argument('--start_mixing_epoch', default=10, type=int)
    
    parser.add_argument('--communicate_iteration', default=1, type=int)
    parser.add_argument('--pretrain_epoch', default=-1, type=int)
    parser.add_argument('--smoothing_lambda', default=0.0, type=float)
    parser.add_argument('--child_rank_num', type=int, help='simulated node num')
    parser.add_argument('--fix_graph_epoch', type=int, help='when to mask mixing weight')
    parser.add_argument('--mask_fraction', type=float, help='proportion to mask mixing weight')
    parser.add_argument('--model_arch', type=str)
    parser.add_argument('--mixing_model_arch', type=str)
    parser.add_argument('--choose_filter_num', type=int)
    parser.add_argument('--scale', default=1.0, type=float)
    parser.add_argument('--experiment_setting', type=str, help='non-iid or non-iid_SBM setting, corresponding to exp setting in paper.')

    args = parser.parse_args()

    if not args.description:
        print('No experiment description, exit!')
        exit()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    child_rank_num=args.child_rank_num

    #devices = os.environ['CUDA_VISIBLE_DEVICES'].strip().split(',')
    devices=range(torch.cuda.device_count())
    print(devices)
    print(devices,flush=True)
    per_device_ranks = int(size/len(devices)) + 1
    print('wsize=',size)
    print('total device num=',len(devices))
    device_assignment=split_device(size,devices,rank)
    print('Device assignment: %s , %s'%(rank, device_assignment),flush=True)
    torch.cuda.set_device(int(device_assignment))


    parent_rank_list=get_parent_rank_list()
    child_rank_list=get_child_rank_list(parent_rank_list)
    if rank==0:
        print(parent_rank_list)
        print(child_rank_list)
    if rank==0:
        import wandb
        # Note: The following XXXXX should be replaced 
        # with your own wandb project name and user name. 
        wandb.init(project=XXXXX,entity=XXXXX,config=args)
    if rank==0:
        mixing_net_checkpoint_path=os.path.join('mixing_net_checkpoint',\
            str(datetime.datetime.now()))
        if not os.path.exists(mixing_net_checkpoint_path):
            os.makedirs(mixing_net_checkpoint_path)
    run(rank, size)

