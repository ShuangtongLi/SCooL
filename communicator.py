import numpy as np
import time
import torch
from mpi4py import MPI
from compressors import get_top_k

from comm_helpers import flatten_tensors, unflatten_tensors

import torch.nn as nn
        
class Communicator(object):
    """ Classs designed for communicating local models at workers """
    def __init__(self, rank, size):
        self.comm = MPI.COMM_WORLD
        self.rank = rank
        self.size = size

    def communicate(self, model):
        # stack all model parameters into one tensor list
        self.tensor_list = list()
        for param in model.parameters():
            self.tensor_list.append(param.data)

        # necessary preprocessing
        self.prepare_comm_buffer()

        # communication happens here
        # record the communication time
        comm_time = self.averaging()

        # Update local models
        self.reset_model()

        return comm_time

    def prepare_comm_buffer(self):
        raise NotImplemented

    def averaging(self):
        raise NotImplemented

    def reset_model(self,model):
        raise NotImplemented

    
        

class centralizedCommunicator(Communicator):
    """ Perform AllReduce at each iteration """
    def __init__(self, rank, size):
        super(centralizedCommunicator, self).__init__(rank, size)

    
    def prepare_comm_buffer(self):
        # faltten tensors
        self.send_buffer = flatten_tensors(self.tensor_list).cpu()

    def averaging(self):
        self.comm.barrier()
        tic = time.time()

        # AllReduce
        self.recv_buffer = self.comm.allreduce(self.send_buffer, op=MPI.SUM)
        self.recv_buffer.div_(self.size)
        
        self.comm.barrier()
        toc = time.time()

        return toc - tic

    def reset_model(self):
        # Reset local models to be the averaged model
        for f, t in zip(unflatten_tensors(
                        self.recv_buffer.cuda(), self.tensor_list), 
                        self.tensor_list):
            t.set_(f)

class decenCommunicator(Communicator):
    """ decentralized averaging according to a topology sequence """
    def __init__(self, rank, size, topology):
        super(decenCommunicator, self).__init__(rank, size)
        self.topology = topology
        


    def prepare_comm_buffer(self):
        # faltten tensors
        self.send_buffer = flatten_tensors(self.tensor_list).cpu()
        self.recv_buffer = torch.zeros_like(self.send_buffer)


    def averaging(self):
        
        self.comm.barrier()
        tic = time.time()
        

        # decentralized averaging
        neighbor_weight=self.topology[self.rank]
        for neighbor_rank in range(len(neighbor_weight)):
            if neighbor_weight[neighbor_rank]>0:
                if neighbor_rank==self.rank:
                    self.recv_buffer.add_(neighbor_weight[neighbor_rank], self.send_buffer)
                    
                else:
                    self.recv_tmp = self.comm.sendrecv(self.send_buffer, source=neighbor_rank, dest = neighbor_rank)
                    # Aggregate neighbors' models: alpha * sum_j x_j
                    self.recv_buffer.add_(neighbor_weight[neighbor_rank], self.recv_tmp)

        
        self.comm.barrier()
        toc = time.time()

        return toc - tic


    def reset_model(self,model):
        # Reset local models to be the averaged model
        with torch.no_grad():
            for f, t in zip(unflatten_tensors(
                            self.recv_buffer.cuda(), self.tensor_list), 
                            model.parameters()): 
                t.data=f
    
    def communicate(self, model):
        # get activated topology at current iteration
        

        # if no subgraphs are activated,
        # then directly start next iteration
        

        # stack all model parameters into one tensor list
        self.tensor_list = list()
        for param in model.parameters():
            self.tensor_list.append(param.data)

        # necessary preprocess
        self.prepare_comm_buffer()

        # decentralized averaging according to activated topology
        # record the communication time
        comm_time = self.averaging()

        # update local models
        self.reset_model(model)

        return comm_time
    def communicate_grad(self, model):
        # get activated topology at current iteration
        

        # if no subgraphs are activated,
        # then directly start next iteration
        

        # stack all model parameters into one tensor list
        self.tensor_list = list()
        for param in model.parameters():
            self.tensor_list.append(param.grad.data)

        # necessary preprocess
        self.prepare_comm_buffer()

        # decentralized averaging according to activated topology
        # record the communication time
        self.gradient_averaging()

        

        return 0
    def gradient_averaging(self):
        
        self.comm.barrier()
        tic = time.time()
        self.gradient_group=[]

        # decentralized averaging
        neighbor_weight=self.topology[self.rank]
        for neighbor_rank in range(len(neighbor_weight)):
            if neighbor_rank==self.rank:
                self.gradient_group.append(self.send_buffer)
                print('my_shape=',self.send_buffer.size())
                
            else:
                self.recv_tmp = self.comm.sendrecv(self.send_buffer, source=neighbor_rank, dest = neighbor_rank)
                # Aggregate neighbors' models: alpha * sum_j x_j
                self.gradient_group.append(self.recv_tmp)
                print(self.recv_tmp.size())
        inner_product_group=[sum(i*self.send_buffer).numpy() for i in self.gradient_group]
        
        print('rank=',self.rank,np.around(np.array(inner_product_group),5))
        print('rank=',self.rank,'my_inner_product=',inner_product_group[self.rank])

        
        self.comm.barrier()
        return 0
        

class decenGradientCommunicator(Communicator):
    """ decentralized averaging according to a topology sequence """
    def __init__(self, rank, size, topology):
        super(decenGradientCommunicator, self).__init__(rank, size)
        self.topology = topology
        


    def prepare_comm_buffer(self):
        # faltten tensors
        self.send_buffer = flatten_tensors(self.tensor_list).cpu()
        self.recv_buffer = torch.zeros_like(self.send_buffer)


    def averaging(self):
        
        self.comm.barrier()
        tic = time.time()
        

        # decentralized averaging
        neighbor_weight=self.topology[self.rank]
        for neighbor_rank in range(len(neighbor_weight)):
            if neighbor_weight[neighbor_rank]>0:
                if neighbor_rank==self.rank:
                    self.recv_buffer.add_(neighbor_weight[neighbor_rank], self.send_buffer)
                    
                else:
                    self.recv_tmp = self.comm.sendrecv(self.send_buffer, source=neighbor_rank, dest = neighbor_rank)
                    # Aggregate neighbors' models: alpha * sum_j x_j
                    self.recv_buffer.add_(neighbor_weight[neighbor_rank], self.recv_tmp)

        
        self.comm.barrier()
        toc = time.time()

        return toc - tic


    def reset_model(self,model):
        # Reset local models to be the averaged model
        with torch.no_grad():
            for f, t in zip(unflatten_tensors(
                            self.recv_buffer.cuda(), self.tensor_list), 
                            model.parameters()): 
                t.grad.data=f
    
    def communicate(self, model):
        # get activated topology at current iteration
        

        # if no subgraphs are activated,
        # then directly start next iteration
        

        # stack all model parameters into one tensor list
        self.tensor_list = list()
        for param in model.parameters():
            self.tensor_list.append(param.grad.data)

        # necessary preprocess
        self.prepare_comm_buffer()

        # decentralized averaging according to activated topology
        # record the communication time
        comm_time = self.averaging()

        # update local models
        self.reset_model(model)

        return comm_time
    def communicate_grad(self, model):
        # get activated topology at current iteration
        

        # if no subgraphs are activated,
        # then directly start next iteration
        

        # stack all model parameters into one tensor list
        self.tensor_list = list()
        for param in model.parameters():
            self.tensor_list.append(param.grad.data)

        # necessary preprocess
        self.prepare_comm_buffer()

        # decentralized averaging according to activated topology
        # record the communication time
        self.gradient_averaging()

        

        return 0
    def gradient_averaging(self):
        
        self.comm.barrier()
        tic = time.time()
        self.gradient_group=[]

        # decentralized averaging
        neighbor_weight=self.topology[self.rank]
        for neighbor_rank in range(len(neighbor_weight)):
            if neighbor_rank==self.rank:
                self.gradient_group.append(self.send_buffer)
                print('my_shape=',self.send_buffer.size())
                
            else:
                self.recv_tmp = self.comm.sendrecv(self.send_buffer, source=neighbor_rank, dest = neighbor_rank)
                # Aggregate neighbors' models: alpha * sum_j x_j
                self.gradient_group.append(self.recv_tmp)
                print(self.recv_tmp.size())
        inner_product_group=[sum(i*self.send_buffer).numpy() for i in self.gradient_group]
        
        print('rank=',self.rank,np.around(np.array(inner_product_group),5))
        print('rank=',self.rank,'my_inner_product=',inner_product_group[self.rank])

        
        self.comm.barrier()
        return 0
class GradientCommunicator(Communicator):
    """ decentralized averaging according to a topology sequence """
    def __init__(self, rank, size, topology):
        super(GradientCommunicator, self).__init__(rank, size)
        self.topology = topology
        


    def prepare_comm_buffer(self):
        # faltten tensors
        self.send_buffer = flatten_tensors(self.tensor_list).cpu()
        self.recv_buffer = torch.zeros_like(self.send_buffer)
    def normalize_grad_group(self):
        grad_norm_record=torch.norm(self.recv_buffer_group[self.rank])+1e-10
        for i in range(len(self.recv_buffer_group)):
            grad=self.recv_buffer_group[i]
            grad_norm=torch.norm(grad)+1e-10
            new_grad=grad/grad_norm*grad_norm_record
            self.recv_buffer_group[i]=new_grad
    
            
    def gradient_threshold(self): 
        threshold=0.01
        indicator_num_group=[]
        self.pruned_gradient_group=[]
        for i in range(len(self.recv_buffer_group)):
            cur_gradient=self.recv_buffer_group[i]
            cur_gradient_abs=torch.abs(cur_gradient)
            max_entry=torch.max(cur_gradient_abs)
            level=max_entry*threshold
            indicator=cur_gradient_abs<level
            new_grad=cur_gradient*indicator
            indicator_num_group.append(torch.sum(indicator))#/len(self.recv_buffer_group[0]))
            #self.pruned_gradient_group.append(new_grad)
            self.pruned_gradient_group.append(torch.clamp(cur_gradient,min=-level,max=level))
        #print('rank=',self.rank,'num=',indicator_num_group)
        for i in range(len(self.pruned_gradient_group)):
            grad=self.pruned_gradient_group[i]
            new_grad=unflatten_tensors(grad.cuda(), self.tensor_list)
            self.pruned_gradient_group[i]=new_grad



    def averaging(self):
        
        self.comm.barrier()
        tic = time.time()
        
        self.recv_buffer_group=[torch.zeros_like(self.send_buffer) for i in range(self.size)]
        # decentralized averaging
        neighbor_weight=self.topology[self.rank]
        for neighbor_rank in range(len(neighbor_weight)):
            if neighbor_weight[neighbor_rank]>0:
                if neighbor_rank==self.rank:
                    self.recv_buffer_group[neighbor_rank]=self.send_buffer
                    
                else:
                    self.recv_tmp = self.comm.sendrecv(self.send_buffer, source=neighbor_rank, dest = neighbor_rank)
                    # Aggregate neighbors' models: alpha * sum_j x_j
                    self.recv_buffer_group[neighbor_rank]=self.recv_tmp
        #self.normalize_grad_group()
        #print('rank=',self.rank,'norm=',[torch.norm(i) for i in self.recv_buffer_group])
        self.gradient_threshold()
        for i in range(len(self.recv_buffer_group)):
            grad=self.recv_buffer_group[i]
            new_grad=unflatten_tensors(grad.cuda(), self.tensor_list)
            self.recv_buffer_group[i]=new_grad

        
        self.comm.barrier()
        toc = time.time()

        return toc - tic


    def reset_model(self,model):
        # Reset local models to be the averaged model
        with torch.no_grad():
            for f, t in zip(unflatten_tensors(
                            self.recv_buffer.cuda(), self.tensor_list), 
                            model.parameters()): 
                t.grad.data=f
    
    def communicate(self, grad):
        # get activated topology at current iteration
        

        # if no subgraphs are activated,
        # then directly start next iteration
        

        # stack all model parameters into one tensor list
        self.tensor_list = list()
        for param_grad in grad:
            self.tensor_list.append(param_grad.data)

        # necessary preprocess
        self.prepare_comm_buffer()

        # decentralized averaging according to activated topology
        # record the communication time
        comm_time = self.averaging()

       

        return self.recv_buffer_group

    
        
class adaptive_decenGradientCommunicator(Communicator):
    """ decentralized averaging according to a topology sequence """
    def __init__(self, rank, size, topology):
        super(adaptive_decenGradientCommunicator, self).__init__(rank, size)
        self.topology = topology
        


    def prepare_comm_buffer(self):
        # faltten tensors
        self.send_buffer = flatten_tensors(self.tensor_list).cpu()
        self.recv_buffer = torch.zeros_like(self.send_buffer)


    def averaging(self):
        
        self.comm.barrier()
        tic = time.time()
        
        self.recv_buffer_group=[torch.zeros_like(self.send_buffer) for i in range(self.size)]
        # decentralized averaging
        neighbor_weight=self.topology[self.rank]
        for neighbor_rank in range(len(neighbor_weight)):
            if neighbor_weight[neighbor_rank]>0:
                if neighbor_rank==self.rank:
                    self.recv_buffer_group[neighbor_rank]=self.send_buffer
                    
                else:
                    self.recv_tmp = self.comm.sendrecv(self.send_buffer, source=neighbor_rank, dest = neighbor_rank)
                    # Aggregate neighbors' models: alpha * sum_j x_j
                    self.recv_buffer_group[neighbor_rank]=self.recv_tmp
        self.mixing_weight=torch.zeros(self.size)
        for i in range(self.size):
            neighbour_gradient=self.recv_buffer_group[i]
            self.mixing_weight[i]=torch.sum(self.send_buffer*neighbour_gradient)
        self.mixing_weight=self.mixing_weight/torch.sum(torch.abs(self.mixing_weight))
        for i in range(self.size):
            neighbour_gradient=self.recv_buffer_group[i]
            self.recv_buffer.add_(self.mixing_weight[i], neighbour_gradient)

        
        self.comm.barrier()
        toc = time.time()

        return toc - tic


    def reset_model(self,model):
        # Reset local models to be the averaged model
        with torch.no_grad():
            for f, t in zip(unflatten_tensors(
                            self.recv_buffer.cuda(), self.tensor_list), 
                            model.parameters()): 
                t.grad.data=f
    
    def communicate(self, model):
        # get activated topology at current iteration
        

        # if no subgraphs are activated,
        # then directly start next iteration
        

        # stack all model parameters into one tensor list
        self.tensor_list = list()
        for param in model.parameters():
            self.tensor_list.append(param.grad.data)

        # necessary preprocess
        self.prepare_comm_buffer()

        # decentralized averaging according to activated topology
        # record the communication time
        comm_time = self.averaging()

        # update local models
        self.reset_model(model)

        return comm_time

    def communicate_grad(self, model):
        # get activated topology at current iteration
        

        # if no subgraphs are activated,
        # then directly start next iteration
        

        # stack all model parameters into one tensor list
        self.tensor_list = list()
        for param in model.parameters():
            self.tensor_list.append(param.grad.data)

        # necessary preprocess
        self.prepare_comm_buffer()

        # decentralized averaging according to activated topology
        # record the communication time
        self.gradient_averaging()

        

        return 0
    def gradient_averaging(self):
        
        self.comm.barrier()
        tic = time.time()
        self.gradient_group=[]

        # decentralized averaging
        neighbor_weight=self.topology[self.rank]
        for neighbor_rank in range(len(neighbor_weight)):
            if neighbor_rank==self.rank:
                self.gradient_group.append(self.send_buffer)
                print('my_shape=',self.send_buffer.size())
                
            else:
                self.recv_tmp = self.comm.sendrecv(self.send_buffer, source=neighbor_rank, dest = neighbor_rank)
                # Aggregate neighbors' models: alpha * sum_j x_j
                self.gradient_group.append(self.recv_tmp)
                print(self.recv_tmp.size())
        inner_product_group=[sum(i*self.send_buffer).numpy() for i in self.gradient_group]
        
        print('rank=',self.rank,np.around(np.array(inner_product_group),5))
        print('rank=',self.rank,'my_inner_product=',inner_product_group[self.rank])

        
        self.comm.barrier()
        return 0
class adaptive_val_decenGradientCommunicator(Communicator):
    """ decentralized averaging according to a topology sequence """
    def __init__(self, rank, size, topology):
        super(adaptive_val_decenGradientCommunicator, self).__init__(rank, size)
        self.topology = topology
        self.criterion= nn.CrossEntropyLoss().cuda()
        self.mixing_weight_momentum_buffer=None
        


    def prepare_comm_buffer(self):
        # faltten tensors
        self.send_buffer = flatten_tensors(self.tensor_list).cpu()
        self.recv_buffer = torch.zeros_like(self.send_buffer)
    def get_val_gradient(self,model,val_loader):

        self.comm.barrier()
        model.zero_grad()
        model.train()
        for batch_idx, (data, target) in enumerate(val_loader):
            # data loading 
            data, target = data.cuda(non_blocking = True), target.cuda(non_blocking = True)                
            
            # forward pass
            output = model(data)
            loss = self.criterion(output, target)
            loss.backward()
        model.train()
        self.val_tensor_list = list()
        for param in model.parameters():
            self.val_tensor_list.append(param.grad.data)
        val_gradient=flatten_tensors(self.val_tensor_list).cpu()
        return val_gradient



    def averaging(self,model,val_loader):
        
        self.comm.barrier()
        tic = time.time()
        
        self.recv_buffer_group=[torch.zeros_like(self.send_buffer) for i in range(self.size)]
        # decentralized averaging
        neighbor_weight=self.topology[self.rank]
        for neighbor_rank in range(len(neighbor_weight)):
            if neighbor_weight[neighbor_rank]>0:
                if neighbor_rank==self.rank:
                    self.recv_buffer_group[neighbor_rank]=self.send_buffer
                    
                else:
                    self.recv_tmp = self.comm.sendrecv(self.send_buffer, source=neighbor_rank, dest = neighbor_rank)
                    # Aggregate neighbors' models: alpha * sum_j x_j
                    self.recv_buffer_group[neighbor_rank]=self.recv_tmp
        self.val_gradient=self.get_val_gradient(model,val_loader)
        self.mixing_weight=torch.zeros(self.size)
        for i in range(self.size):
            neighbour_gradient=self.recv_buffer_group[i]/torch.sqrt(torch.norm(self.recv_buffer_group[i]))
            self.mixing_weight[i]=torch.sum(self.val_gradient*neighbour_gradient)
        self.mixing_weight=self.mixing_weight/torch.sum(torch.abs(self.mixing_weight))
        if self.mixing_weight_momentum_buffer==None:
            self.mixing_weight_momentum_buffer=self.mixing_weight
        else:
            self.mixing_weight=0.9*self.mixing_weight_momentum_buffer+0.1*self.mixing_weight
            self.mixing_weight_momentum_buffer=self.mixing_weight
        for i in range(self.size):
            neighbour_gradient=self.recv_buffer_group[i]
            self.recv_buffer.add_(self.mixing_weight[i], neighbour_gradient)

        
        self.comm.barrier()
        toc = time.time()

        return toc - tic


    def reset_model(self,model):
        # Reset local models to be the averaged model
        with torch.no_grad():
            for f, t in zip(unflatten_tensors(
                            self.recv_buffer.cuda(), self.tensor_list), 
                            model.parameters()): 
                t.grad.data=f
    
    def communicate(self, model, val_loader):
        # get activated topology at current iteration
        

        # if no subgraphs are activated,
        # then directly start next iteration
        

        # stack all model parameters into one tensor list
        self.tensor_list = list()
        for param in model.parameters():
            self.tensor_list.append(param.grad.data)

        # necessary preprocess
        self.prepare_comm_buffer()

        # decentralized averaging according to activated topology
        # record the communication time
        comm_time = self.averaging(model,val_loader)

        # update local models
        self.reset_model(model)

        return comm_time
    
    def communicate_grad(self, model):
        # get activated topology at current iteration
        

        # if no subgraphs are activated,
        # then directly start next iteration
        

        # stack all model parameters into one tensor list
        self.tensor_list = list()
        for param in model.parameters():
            self.tensor_list.append(param.grad.data)

        # necessary preprocess
        self.prepare_comm_buffer()

        # decentralized averaging according to activated topology
        # record the communication time
        self.gradient_averaging()

        

        return 0
    def gradient_averaging(self):
        
        self.comm.barrier()
        tic = time.time()
        self.gradient_group=[]

        # decentralized averaging
        neighbor_weight=self.topology[self.rank]
        for neighbor_rank in range(len(neighbor_weight)):
            if neighbor_rank==self.rank:
                self.gradient_group.append(self.send_buffer)
                print('my_shape=',self.send_buffer.size())
                
            else:
                self.recv_tmp = self.comm.sendrecv(self.send_buffer, source=neighbor_rank, dest = neighbor_rank)
                # Aggregate neighbors' models: alpha * sum_j x_j
                self.gradient_group.append(self.recv_tmp)
                print(self.recv_tmp.size())
        inner_product_group=[sum(i*self.send_buffer).numpy() for i in self.gradient_group]
        
        print('rank=',self.rank,np.around(np.array(inner_product_group),5))
        print('rank=',self.rank,'my_inner_product=',inner_product_group[self.rank])

        
        self.comm.barrier()
        return 0
        

class ChocoCommunicator(Communicator):
    """ decentralized averaging using compressed gradients (top-k) """
    
    def __init__(self, rank, size, topology, ratio, consensus_lr):
        super(ChocoCommunicator, self).__init__(rank, size)
        self.topology = topology
        self.neighbor_weight = topology.neighbor_weight
        self.iter = 0

        self.initialized = False
        self.consensus_lr = consensus_lr
        self.ratio = ratio


    def prepare_comm_buffer(self):
        # flatten tensors
        # If not initialized, then initialize x_hat and s
        self.x = flatten_tensors(self.tensor_list).cpu()
        if not self.initialized:
            self.x_hat = torch.zeros_like(self.x)
            self.s = torch.zeros_like(self.x)
            self.initialized = True

        tic = time.time()
        # get compressed message
        # here, we use top_k compressor on GPU
        # one can define more in compressors.py
        self.send_buffer = self.x - self.x_hat
        values, indices = get_top_k(self.send_buffer.cuda(), self.ratio)
        toc = time.time()

        values, indices = values.cpu(), indices.cpu()
        self.compressed = {"values":values, "indices":indices}

        return toc - tic



    def averaging(self, active_flags):
        self.comm.barrier()
        tic = time.time()

        # decentralized averaging according to activated topology
        degree = 0
        for graph_id, flag in enumerate(active_flags):
            if flag == 0:
                continue
            else:
                if self.topology.neighbors_info[graph_id][self.rank] != -1:
                    degree += 1
                    neighbor_rank = self.topology.neighbors_info[graph_id][self.rank]
                    # Receive neighbor's message q_j
                    self.recv_tmp = self.comm.sendrecv(self.compressed, source=neighbor_rank, dest = neighbor_rank)
                    # Update aggregated model s += sum w_ij q_j
                    self.s[self.recv_tmp["indices"]] += self.neighbor_weight * self.recv_tmp["values"]

        # Compute self weight
        selfweight = 1 - degree * self.neighbor_weight
        # Update aggregated model s += w_ii q_i
        self.s[self.compressed["indices"]] += selfweight * self.compressed["values"]
        # Update x_hat = x_hat + q_i
        self.x_hat[self.compressed["indices"]] += self.compressed["values"]
        # Update local model parameters: x = x + consensus_lr*(s-x_hat)
        self.x.add_(self.consensus_lr, self.s).sub_(self.consensus_lr, self.x_hat)
        
        self.comm.barrier()
        toc = time.time()

        return toc - tic


    def reset_model(self):
        # Reset local models to be the averaged model
        for f, t in zip(unflatten_tensors(
                        self.x.cuda(), self.tensor_list), 
                        self.tensor_list):
            t.set_(f)

    def communicate(self, model):
        # get activated topology at current iteration
        active_flags = self.topology.active_flags[self.iter]
        self.iter += 1

        # if no subgraphs are activated,
        # then directly start next iteration
        if np.sum(active_flags) == 0:
            return 0

        # stack all model parameters into one tensor list
        self.tensor_list = list()
        for param in model.parameters():
            self.tensor_list.append(param.data)

        # necessary preprocess
        # there is an additional encoding time
        encode_time = self.prepare_comm_buffer()

        # decentralized averaging
        # record the communication time
        comm_time = self.averaging(active_flags)

        # update local models
        self.reset_model()

        return encode_time + comm_time