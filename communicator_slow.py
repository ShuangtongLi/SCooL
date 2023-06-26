import numpy as np
import time
import torch
from mpi4py import MPI
from compressors import get_top_k

from comm_helpers import flatten_tensors, unflatten_tensors

import torch.nn as nn
        
class Communicator(object):
    """ Classs designed for communicating local models at workers """
    def __init__(self, rank, size, comm):
        self.comm = comm
        self.rank = rank
        self.size = size

   



      
class GradientCommunicator(Communicator):
    """ decentralized averaging according to a topology sequence """
    def __init__(self, rank, size, topology, comm, parent_rank_list):
        super(GradientCommunicator, self).__init__(rank, size, comm)
        self.topology = topology
        self.parent_rank_list=parent_rank_list
        


   
    def preprocess_gather(self,send_buffer_group,parent_rank_list):
        effective_send_buffer_group=[]
        self.tensor_list=send_buffer_group[parent_rank_list[self.rank][0]]
        for i in parent_rank_list[self.rank]:
            flattened_send_buffer=send_buffer_group[i]
            flattened_send_buffer=flatten_tensors(flattened_send_buffer).cpu()
            effective_send_buffer_group.append(flattened_send_buffer)
        return effective_send_buffer_group
    def postprocess_gather(self,recv_buffer_group):
        effective_recv_buffer_group=[]
        for i in recv_buffer_group:
            effective_recv_buffer_group=effective_recv_buffer_group+i
        #print(len(effective_recv_buffer_group))
        for i in range(len(effective_recv_buffer_group)):
            grad=effective_recv_buffer_group[i]
            new_grad=unflatten_tensors(grad.cuda(), self.tensor_list)
            effective_recv_buffer_group[i]=new_grad

        return effective_recv_buffer_group
    def communicate(self,send_buffer_group):
        self.comm.barrier()
        effective_send_buffer_group=self.preprocess_gather(send_buffer_group,self.parent_rank_list)
        for i in range(self.size):
            tmp_recv_buffer_group=self.comm.gather(effective_send_buffer_group,root=i)
            if i==self.rank:
                recv_buffer_group=tmp_recv_buffer_group
        effective_recv_buffer_group=self.postprocess_gather(recv_buffer_group)
        self.comm.barrier()
        return effective_recv_buffer_group
        


   