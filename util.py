import os
import numpy as np
import time
import argparse
from PIL import Image
from mpi4py import MPI
from math import ceil
from random import Random

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

from models import *

#import GraphPreprocess 
class CIFAR10(object):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform= None,
            target_transform = None,
            download: bool = False,
    ) -> None:

        self.root=root
        torch._C._log_api_usage_once(f"torchvision.datasets.{self.__class__.__name__}")

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1)) 
        self.targets=np.array(self.targets)
        '''
        shuffle_index=np.arange(len(self.targets))
        np.random.shuffle(shuffle_index)
        self.data=self.data[shuffle_index]
        self.targets=self.targets[shuffle_index]
        '''
        
    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        torchvision.datasets.utils.download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)
    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

class my_dataset(Dataset):
    def __init__(
            self,
            data,
            targets,
            index_group,
            root=None,
            transform=  None,
            target_transform =None,
    ):

        
        self.data=data[index_group]
        self.targets=[targets[i] for i in index_group]
        self.transform=transform
        self.target_transform=target_transform
        self.target_set_list=list(set(self.targets))
        print('data_class_num=',len(self.data),'label_class_num=',len(self.targets),'allocated_labels=',self.target_set_list)
    def __getitem__(self, index: int) :
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        #target=self.target_set_list.index(target)

        return img, target

    def __len__(self) :
        return len(self.data)

    
class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """
    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234, isNonIID=False):
        self.data = data 
        self.partitions = [] 
        rng = Random() 
        rng.seed(seed) 
        data_len = len(data) 
        indexes = [x for x in range(0, data_len)] 
        
        rng.shuffle(indexes) 
        for i in range(10):
            train_data,label=self.data[i]
            print(label)
         
 
        for frac in sizes: 
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

        if isNonIID:
            self.partitions = __getNonIIDdata__(self, data, sizes, seed)

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

    def __getNonIIDdata__(self, data, sizes, seed):
        labelList = data.train_labels
        rng = Random()
        rng.seed(seed)
        a = [(label, idx) for idx, label in enumerate(labelList)]
        # Same Part
        labelIdxDict = dict()
        for label, idx in a:
                labelIdxDict.setdefault(label,[])
                labelIdxDict[label].append(idx)
        labelNum = len(labelIdxDict)
        labelNameList = [key for key in labelIdxDict]
        labelIdxPointer = [0] * labelNum
        partitions = [list() for i  in range(len(sizes))]
        eachPartitionLen= int(len(labelList)/len(sizes))
        majorLabelNumPerPartition = ceil(labelNum/len(partitions))
        basicLabelRatio = 0.4

        interval = 1
        labelPointer = 0

        #basic part
        for partPointer in range(len(partitions)):
            requiredLabelList = list()
            for _ in range(majorLabelNumPerPartition):
                requiredLabelList.append(labelPointer)
                labelPointer += interval
                if labelPointer > labelNum - 1:
                    labelPointer = interval
                    interval += 1
            for labelIdx in requiredLabelList:
                start = labelIdxPointer[labelIdx]
                idxIncrement = int(basicLabelRatio*len(labelIdxDict[labelNameList[labelIdx]]))
                partitions[partPointer].extend(labelIdxDict[labelNameList[labelIdx]][start:start+ idxIncrement])
                labelIdxPointer[labelIdx] += idxIncrement

        #random part
        remainLabels = list()
        for labelIdx in range(labelNum):
            remainLabels.extend(labelIdxDict[labelNameList[labelIdx]][labelIdxPointer[labelIdx]:])
        rng.shuffle(remainLabels)
        for partPointer in range(len(partitions)):
            idxIncrement = eachPartitionLen - len(partitions[partPointer])
            partitions[partPointer].extend(remainLabels[:idxIncrement])
            rng.shuffle(partitions[partPointer])
            remainLabels = remainLabels[idxIncrement:]
        return partitions

def partition_dataset(rank, size, args):
    print('==> load train data')
    if args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root=args.datasetRoot, 
                                                train=True, 
                                                download=True, 
                                                transform=transform_train)
 
        partition_sizes = [1.0 / size for _ in range(size)]
        partition = DataPartitioner(trainset, partition_sizes, isNonIID=False)
        partition = partition.use(rank)
        train_loader = torch.utils.data.DataLoader(partition, 
                                                batch_size=args.bs, 
                                                shuffle=True, 
                                                pin_memory=True)
 
        print('==> load test data')
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = torchvision.datasets.CIFAR10(root=args.datasetRoot, 
                                            train=False, 
                                            download=True, 
                                            transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, 
                                                batch_size=64, 
                                                shuffle=False, 
                                                num_workers=size)

    if args.dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        trainset = torchvision.datasets.CIFAR100(root=args.datasetRoot, 
                                                train=True, 
                                                download=True, 
                                                transform=transform_train)
 
        partition_sizes = [1.0 / size for _ in range(size)]
        partition = DataPartitioner(trainset, partition_sizes, isNonIID=False)
        partition = partition.use(rank)
        train_loader = torch.utils.data.DataLoader(partition, 
                                                batch_size=args.bs, 
                                                shuffle=True, 
                                                pin_memory=True)
 
        print('==> load test data')
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        testset = torchvision.datasets.CIFAR100(root=args.datasetRoot, 
                                            train=False, 
                                            download=True, 
                                            transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, 
                                                batch_size=64, 
                                                shuffle=False, 
                                                num_workers=size)
def expand_label_group(label_group,split):
    '''
    label_group: list[list],
    split(int): every list in label group evenly divided into split groups
    split(list): list represents the proportion every group devides, for example, [0.8,0.2] for train/val split,now
        we only support the case that split is a two-element tuple.
    '''
    group_num=len(label_group)
    
    if isinstance(split,int):
        new_label_group=[]
        for i in range(group_num):
            label_inner_group=label_group[i]
            every_split_num=len(label_inner_group)//split
            for j in range(split):
                if j==(split-1):
                    new_label_group.append(label_inner_group[j*every_split_num:])
                else:
                    new_label_group.append(label_inner_group[j*every_split_num:(j+1)*every_split_num])
        return new_label_group
    elif isinstance(split,list):
        new_label_group=[]
        new_label_group1=[]
        for i in range(group_num):

            label_inner_group=label_group[i]
            label_inner_group_num=len(label_inner_group)
            start_num=0
            for j in range(len(split)):
                num_ratio=split[j]
                num_samples=int(num_ratio*label_inner_group_num)
                if j==(len(split)-1):
                    new_label_group1.append(label_inner_group[start_num:])
                else:
                    end_num=start_num+num_samples
                    new_label_group.append(label_inner_group[start_num:end_num])
                    start_num=end_num
        return new_label_group,new_label_group1




def split_non_iid_dataset_tiny(data_object,every_label_split_num,train_val_ratio=None,num_labels=None):#train_val_ratio=[0.8,0.2]
    images=data_object.data
    labelList=data_object.targets
    a = [(label, idx) for idx, label in enumerate(labelList)]
    labelDict = [[] for i in range(num_labels)]
    for label, idx in a:
        labelDict[label].append(idx)
    for label in range(num_labels):
        tmp_label_group=labelDict[label]
        num_group=len(tmp_label_group)
        new_num_group=num_group
        labelDict[label]=tmp_label_group[:new_num_group]
    label_group=expand_label_group(labelDict,every_label_split_num)
    if train_val_ratio==None:
        return label_group
    else:
        train_label_group,val_label_group=expand_label_group(label_group,train_val_ratio)
        return train_label_group,val_label_group

def check_all_ascend(list1):
    for i in range(len(list1)):
        if i==0:
            previous=list1[i]
        else:
            if list1[i]<=previous:
                raise ValueError
            else:
                previous=list1[i]
def check_all_label_correct(all_lables,label_index):
    '''
    if return 0, then correct
    '''
    true_label=all_lables[label_index[0]]
    all_lables=np.array(all_lables)
    #print(all_lables[label_index])
    return sum(all_lables[label_index]!=true_label)

def select_partition(split_label_group_num,every_label_split_num,num_labels):
    a=num_labels//split_label_group_num
    group_index_partition=[]
    for i in range(split_label_group_num):
        for j in range(every_label_split_num):
            tmp=[]
            start=a*i*every_label_split_num+j
            for k in range(a):
                tmp.append(start+every_label_split_num*k)
            group_index_partition.append(tmp)
    return group_index_partition
def select_partition_rand_non_iid(split_label_group_num,every_label_split_num,args_randomseed,num_labels):
  
    total_shards_num=num_labels*every_label_split_num
    group_index_partition=np.arange(total_shards_num)
    group_index_partition=np.reshape(group_index_partition,(-1,every_label_split_num)).transpose()
    np.random.seed(args_randomseed)
    [np.random.shuffle(i) for i in group_index_partition]
    group_index_partition=group_index_partition.reshape((-1,num_labels//split_label_group_num))
    group_index_partition=group_index_partition.tolist()
    #print(group_index_partition)
    return group_index_partition
def select_partition_rand_non_iid1(split_label_group_num,every_label_split_num):
    num_labels=10
    total_shards_num=num_labels*every_label_split_num
    device_num=split_label_group_num*every_label_split_num
    np.random.seed(10)
    total_choice_group=[]
    for i in range(device_num):
        ind_choice=np.random.choice(total_shards_num,\
            num_labels//split_label_group_num, replace=False)
        total_choice_group.append(ind_choice)
    return total_choice_group
def select_partition_rand_non_iid2(split_label_group_num,every_label_split_num):
    num_labels=10
    group_index_partition=[[0,20],[1,21],[40,60],[41,61],[22,42],[23,43]]
    return group_index_partition
def fusion_index_by_partition(label_group_index,index_partition):
    new_label_group_index=[]
    for i in range(len(index_partition)):
        tmp=[]
        tmp_index_partition=index_partition[i]
        for j in range(len(tmp_index_partition)):
            new_index=label_group_index[tmp_index_partition[j]]
            tmp=tmp+new_index
        new_label_group_index.append(tmp)
    return new_label_group_index
def show_partition_labels(targets,targets_index):
    local_targets=[targets[i] for i in targets_index]
    return set(local_targets)
def any_two_intersect(label_group_index):
    for i in range(len(label_group_index)):
        for j in range(len(label_group_index)):
            if j==i:
                continue
            print(set(label_group_index[i])&set(label_group_index[j]))
def get_local_testset_index(train_label_set,test_label_group_index):
    #train_label_set should be a python set type
    train_label_list=list(train_label_set)
    tmp=[]
    for i in range(len(train_label_list)):
        tmp=tmp+test_label_group_index[train_label_list[i]]
    return tmp






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

class Recorder(object):
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

    def add_new(self,comm,record_time,comp_time,comm_time,epoch_time,top1,losses,test_acc,epoch,mixing_tensor_array=None):
        self.total_record_timing.append(record_time)
        self.record_timing.append(epoch_time)
        self.record_comp_timing.append(comp_time)
        self.record_comm_timing.append(comm_time)
        self.record_trainacc.append(top1.cpu())
        self.record_losses.append(losses)
        self.record_accuracy.append(test_acc.cpu())
        train_acc_group=comm.gather(top1.cpu(),root=0)
        test_acc_group=comm.gather(test_acc.cpu(),root=0)
        train_loss_group=comm.gather(losses,root=0)
        if mixing_tensor_array==None:
            if self.rank==0:
                import wandb
                wandb_dict=dict()
                for i in range(self.size):
                    wandb_dict['train_accuracy'+str(i)]=train_acc_group[i]
                    wandb_dict['train_loss'+str(i)]=train_loss_group[i]
                    wandb_dict['test_acc'+str(i)]=test_acc_group[i]
                wandb_dict['avg_train_accuracy']=sum(train_acc_group)/len(train_acc_group)
                wandb_dict['avg_train_loss']=sum(train_loss_group)/len(train_loss_group)
                wandb_dict['avg_test_acc']=sum(test_acc_group)/len(test_acc_group)
                wandb_dict['epoch']=epoch
                wandb.log(wandb_dict)
                print('epoch=',epoch,'avg test acc=',sum(test_acc_group)/len(test_acc_group))
        else:
            if self.rank==0:
                import wandb
                wandb_dict=dict()
                for i in range(self.size):
                    wandb_dict['train_accuracy'+str(i)]=train_acc_group[i]
                    wandb_dict['train_loss'+str(i)]=train_loss_group[i]
                    wandb_dict['test_acc'+str(i)]=test_acc_group[i]
                wandb_dict['avg_train_accuracy']=sum(train_acc_group)/len(train_acc_group)
                wandb_dict['avg_train_loss']=sum(train_loss_group)/len(train_loss_group)
                wandb_dict['avg_test_acc']=sum(test_acc_group)/len(test_acc_group)
                print('epoch=',epoch,'avg test acc=',sum(test_acc_group)/len(test_acc_group))
            mixing_matrix=get_mixing_mat(mixing_tensor_array,self.rank,self.size,comm)
            if self.rank==0:
                x_labels=[str(i) for i in range(len(mixing_matrix))]
                y_labels=[str(i) for i in range(len(mixing_matrix))]
                #wandb_dict['mixing_weight_heatmap']=[wandb.plots.HeatMap(x_labels, y_labels, mixing_matrix, show_text=False)]
                import matplotlib.pyplot as plt
                import matplotlib
                matplotlib.use('Agg')
                fig=plt.figure()
                plt.imshow(mixing_matrix, cmap='Purples')
                plt.colorbar()
                wandb_dict['mixing_weight_heatmap']=wandb.Image(plt)
                wandb_dict['epoch']=epoch
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
        


def test(model, test_loader):
    model.eval()
    top1 = AverageMeter()
    # correct = 0
    # total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
        outputs = model(inputs)
        acc1 = comp_accuracy(outputs, targets)
        top1.update(acc1[0], inputs.size(0))
    return top1.avg
def get_mixing_mat(mixing_tensor_array,rank,size,comm):
    mixing_mat=comm.gather(mixing_tensor_array,root=0)
    mixing_matrix=torch.zeros(size,size)
    if rank==0:
        import wandb
        for i in range(len(mixing_mat)):
            for j in range(len(mixing_mat[i])):
                mixing_matrix[i][j]=mixing_mat[i][j]
    return mixing_matrix
def get_true_correlation(label_set,rank,size,comm):
    mixing_mat=comm.gather(label_set,root=0)
    mixing_matrix=torch.zeros(size,size)
    if rank==0:
        mixing_mat=np.array(mixing_mat)
        print(mixing_mat)

        for i in range(len(mixing_matrix)):
            for j in range(len(mixing_matrix[i])):
                set1=set(mixing_mat[i])
                set2=set(mixing_mat[j])
                intersection_set=set1.intersection(set2)
                intersection_num=len(intersection_set)
                mixing_matrix[i][j]=intersection_num
        import matplotlib.pyplot as plt
        import matplotlib
        import wandb
        matplotlib.use('Agg')
        fig=plt.figure()
        plt.imshow(mixing_matrix, cmap='Purples')
        plt.colorbar()
        wandb_dict=dict()
        wandb_dict['truth_heatmap']=wandb.Image(plt)
        wandb.log(wandb_dict)
    return mixing_matrix
def recorder_fast(mat,rank,name=''):
    import wandb
    if rank==0:
        for i in range(len(mat)):
            for j in range(len(mat[i])):
                wandb_dict={
                name+str(i)+'_'+str(j)+'weight': mat[i][j]
                }

                wandb.log(wandb_dict)
        













def partition_non_iid_random_dataset(rank, size, args):
    import my_transforms
    '''
    size=args.split_label_group_num*args.every_label_split_num
    '''
    print('rank=',rank)
    print('==> load train data')
    
    if args.dataset == 'cifar10_tiny':
        '''
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4
            ),
        '''
        transform_train = transforms.Compose([
            my_transforms.RandomCrop(32, padding=4,fill=128),
            transforms.RandomHorizontalFlip(),
            #transforms.ColorJitter(brightness=0.9,contrast=0.9, saturation=0.9),
            transforms.ColorJitter(saturation=0.9),
            #Equalize(),
            transforms.ToTensor(),
            #Cutout(n_holes=1, length=16),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        '''
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        '''
        '''
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        '''
        '''
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        '''
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset1 = CIFAR10(root=args.datasetRoot, train=True, download=True)
        if args.train_ratio==None:
            train_label_group_index=split_non_iid_dataset_tiny(trainset1,args.every_label_split_num,num_labels=10)
            #print('train_label_group_index=',train_label_group_index)
            train_index_partition=select_partition_rand_non_iid(args.split_label_group_num,args.every_label_split_num,args.randomSeed,num_labels=10)
            #print('train_index_partition=',train_index_partition)
            train_label_group_index=fusion_index_by_partition(train_label_group_index,train_index_partition)
            for tmp_rank in range(size):
                print('rank=',tmp_rank,'allocated_labels=',
                    show_partition_labels(trainset1.targets,train_label_group_index[tmp_rank]),
                    'total_num=',len(train_label_group_index[tmp_rank]))
            #print(any_two_intersect(train_label_group_index))
            #print('resulted_train_label_group_index=',train_label_group_index)
            #print('rank=',rank,train_label_group_index[rank])

            '''
            train_label_partition=[]
            for i in range(len(train_index_partition)):
                tmp=[]
                for j in range(len(train_index_partition[i])):
                    tmp.append(trainset1.targets[train_label_group_index[train_index_partition[i][j]][0]])
                train_label_partition.append(tmp)
            print(train_label_partition)
            '''
            trainset=my_dataset(data=trainset1.data,targets=trainset1.targets,
                index_group=train_label_group_index[rank],transform=transform_train)
            train_loader = torch.utils.data.DataLoader(trainset, 
                                                batch_size=args.bs, 
                                                shuffle=True, 
                                                pin_memory=True)
        else:
            train_val_ratio=[args.train_ratio,args.val_ratio]
            train_label_group_index,val_label_group_index=split_non_iid_dataset_tiny(trainset1,
                args.every_label_split_num,train_val_ratio=train_val_ratio,num_labels=10)
            #any_two_intersect(train_label_group_index+val_label_group_index)
            #print('train_label_group_index=',train_label_group_index)


            train_index_partition=select_partition_rand_non_iid(args.split_label_group_num,args.every_label_split_num,args.randomSeed,num_labels=10)
            #train_index_partition=select_partition(args.split_label_group_num,args.every_label_split_num,num_labels=10)
            #print('train_index_partition=',train_index_partition)
            train_label_group_index=fusion_index_by_partition(train_label_group_index,train_index_partition)
            '''
            for tmp_rank in range(size):
                print('rank=',tmp_rank,'allocated_labels=',
                    show_partition_labels(trainset1.targets,train_label_group_index[tmp_rank]),
                    'total_num=',len(train_label_group_index[tmp_rank]))
            '''
            trainset=my_dataset(data=trainset1.data,targets=trainset1.targets,
                index_group=train_label_group_index[rank],transform=transform_train)


            #val_index_partition=select_partition(args.split_label_group_num,args.every_label_split_num)
            val_label_group_index=fusion_index_by_partition(val_label_group_index,train_index_partition)
           
            #val_label_group_index=fusion_index_by_partition(val_label_group_index,val_index_partition)
            '''
            for tmp_rank in range(size):
                print('rank=',tmp_rank,'trainset_allocated_labels=',
                    show_partition_labels(trainset1.targets,train_label_group_index[tmp_rank]),
                    'total_num=',len(train_label_group_index[tmp_rank]))
            for tmp_rank in range(size):
                print('rank=',tmp_rank,'valset_allocated_labels=',
                    show_partition_labels(trainset1.targets,val_label_group_index[tmp_rank]),
                    'total_num=',len(val_label_group_index[tmp_rank]))
            '''
            valset=my_dataset(data=trainset1.data,targets=trainset1.targets,
                index_group=val_label_group_index[rank],transform=transform_test)
            train_loader = torch.utils.data.DataLoader(trainset, 
                                                batch_size=args.bs, 
                                                shuffle=True, 
                                                pin_memory=True)
            
            val_loader = torch.utils.data.DataLoader(valset, 
                                                batch_size=100, 
                                                shuffle=True, 
                                                pin_memory=True)
            
        testset1 = CIFAR10(root=args.datasetRoot, train=False, download=True)
        test_label_group_index=split_non_iid_dataset_tiny(testset1,1,num_labels=10)
        #print('train_label_group_index=',train_label_group_index)
        test_label_index=get_local_testset_index(set(trainset.targets),test_label_group_index)
        testset=my_dataset(data=testset1.data,targets=testset1.targets,
                index_group=test_label_index,transform=transform_test)
        
        #print(any_two_intersect(train_label_group_index))
        #print('resulted_train_label_group_index=',train_label_group_index)
        #print('rank=',rank,train_label_group_index[rank])

        '''
        train_label_partition=[]
        for i in range(len(train_index_partition)):
            tmp=[]
            for j in range(len(train_index_partition[i])):
                tmp.append(trainset1.targets[train_label_group_index[train_index_partition[i][j]][0]])
            train_label_partition.append(tmp)
        print(train_label_partition)
        '''
        test_loader = torch.utils.data.DataLoader(testset, 
                                                batch_size=64, 
                                                shuffle=False, 
                                                num_workers=4)




        if args.train_ratio==None:
            return train_loader,test_loader,testset.target_set_list
        else:
            return train_loader,val_loader,test_loader,testset.target_set_list



    
 
def partition_non_iid_determine_dataset(rank, size, args):
    import my_transforms
    '''
    size=args.split_label_group_num*args.every_label_split_num
    '''
    print('rank=',rank)
    print('==> load train data')
    
    if args.dataset == 'cifar10_tiny':
        '''
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4
            ),
        '''
        transform_train = transforms.Compose([
            my_transforms.RandomCrop(32, padding=4,fill=128),
            transforms.RandomHorizontalFlip(),
            #transforms.ColorJitter(brightness=0.9,contrast=0.9, saturation=0.9),
            transforms.ColorJitter(saturation=0.9),
            #Equalize(),
            transforms.ToTensor(),
            #Cutout(n_holes=1, length=16),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        '''
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        '''
        '''
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        '''
        '''
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        '''
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset1 = CIFAR10(root=args.datasetRoot, train=True, download=True)
        if args.train_ratio==None:
            train_label_group_index=split_non_iid_dataset_tiny(trainset1,args.every_label_split_num,num_labels=10)
            #print('train_label_group_index=',train_label_group_index)
            train_index_partition=select_partition(args.split_label_group_num,args.every_label_split_num,num_labels=10)
            #print('train_index_partition=',train_index_partition)
            train_label_group_index=fusion_index_by_partition(train_label_group_index,train_index_partition)
            for tmp_rank in range(size):
                print('rank=',tmp_rank,'allocated_labels=',
                    show_partition_labels(trainset1.targets,train_label_group_index[tmp_rank]),
                    'total_num=',len(train_label_group_index[tmp_rank]))
            #print(any_two_intersect(train_label_group_index))
            #print('resulted_train_label_group_index=',train_label_group_index)
            #print('rank=',rank,train_label_group_index[rank])

            '''
            train_label_partition=[]
            for i in range(len(train_index_partition)):
                tmp=[]
                for j in range(len(train_index_partition[i])):
                    tmp.append(trainset1.targets[train_label_group_index[train_index_partition[i][j]][0]])
                train_label_partition.append(tmp)
            print(train_label_partition)
            '''
            trainset=my_dataset(data=trainset1.data,targets=trainset1.targets,
                index_group=train_label_group_index[rank],transform=transform_train)
            train_loader = torch.utils.data.DataLoader(trainset, 
                                                batch_size=args.bs, 
                                                shuffle=True, 
                                                pin_memory=True)
        else:
            train_val_ratio=[args.train_ratio,args.val_ratio]
            train_label_group_index,val_label_group_index=split_non_iid_dataset_tiny(trainset1,
                args.every_label_split_num,train_val_ratio=train_val_ratio,num_labels=10)
            #any_two_intersect(train_label_group_index+val_label_group_index)
            #print('train_label_group_index=',train_label_group_index)


            train_index_partition=select_partition(args.split_label_group_num,args.every_label_split_num,num_labels=10)
            #train_index_partition=select_partition(args.split_label_group_num,args.every_label_split_num,num_labels=10)
            #print('train_index_partition=',train_index_partition)
            train_label_group_index=fusion_index_by_partition(train_label_group_index,train_index_partition)
            '''
            for tmp_rank in range(size):
                print('rank=',tmp_rank,'allocated_labels=',
                    show_partition_labels(trainset1.targets,train_label_group_index[tmp_rank]),
                    'total_num=',len(train_label_group_index[tmp_rank]))
            '''
            trainset=my_dataset(data=trainset1.data,targets=trainset1.targets,
                index_group=train_label_group_index[rank],transform=transform_train)


            #val_index_partition=select_partition(args.split_label_group_num,args.every_label_split_num)
            val_label_group_index=fusion_index_by_partition(val_label_group_index,train_index_partition)
           
            #val_label_group_index=fusion_index_by_partition(val_label_group_index,val_index_partition)
            '''
            for tmp_rank in range(size):
                print('rank=',tmp_rank,'trainset_allocated_labels=',
                    show_partition_labels(trainset1.targets,train_label_group_index[tmp_rank]),
                    'total_num=',len(train_label_group_index[tmp_rank]))
            for tmp_rank in range(size):
                print('rank=',tmp_rank,'valset_allocated_labels=',
                    show_partition_labels(trainset1.targets,val_label_group_index[tmp_rank]),
                    'total_num=',len(val_label_group_index[tmp_rank]))
            '''
            valset=my_dataset(data=trainset1.data,targets=trainset1.targets,
                index_group=val_label_group_index[rank],transform=transform_test)
            train_loader = torch.utils.data.DataLoader(trainset, 
                                                batch_size=args.bs, 
                                                shuffle=True, 
                                                pin_memory=True)
            
            val_loader = torch.utils.data.DataLoader(valset, 
                                                batch_size=100, 
                                                shuffle=True, 
                                                pin_memory=True)
            
        testset1 = CIFAR10(root=args.datasetRoot, train=False, download=True)
        test_label_group_index=split_non_iid_dataset_tiny(testset1,1,num_labels=10)
        #print('train_label_group_index=',train_label_group_index)
        test_label_index=get_local_testset_index(set(trainset.targets),test_label_group_index)
        testset=my_dataset(data=testset1.data,targets=testset1.targets,
                index_group=test_label_index,transform=transform_test)
        
        #print(any_two_intersect(train_label_group_index))
        #print('resulted_train_label_group_index=',train_label_group_index)
        #print('rank=',rank,train_label_group_index[rank])

        '''
        train_label_partition=[]
        for i in range(len(train_index_partition)):
            tmp=[]
            for j in range(len(train_index_partition[i])):
                tmp.append(trainset1.targets[train_label_group_index[train_index_partition[i][j]][0]])
            train_label_partition.append(tmp)
        print(train_label_partition)
        '''
        test_loader = torch.utils.data.DataLoader(testset, 
                                                batch_size=64, 
                                                shuffle=False, 
                                                num_workers=4)




        if args.train_ratio==None:
            return train_loader,test_loader,testset.target_set_list
        else:
            return train_loader,val_loader,test_loader,testset.target_set_list



    
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class Equalize(object):
    def __init__(self):
        pass
    def __call__(self,img):
        from PIL import Image, ImageOps, ImageEnhance
        return ImageOps.equalize(img)