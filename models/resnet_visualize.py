import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import torch.nn.init as init
import comm_helpers
import util
import copy
class Mixing_Net_Attention2(nn.Module):
    def __init__(self, input, hidden1, output, rank, comm,size):
        super(Mixing_Net_Attention2, self).__init__()
        self.register_parameter(name='trans_mat', param=torch.nn.Parameter(torch.ones(1)))
        self.rank=rank
        self.comm=comm
        self.size=size
        self.grad_group_buffer=[None for i in range(size)]
        self.key_tensor_buffer=None
        self.grad_group_buffer_layerwise=[[None for i in range(size)] for j in range(10)]
        self.key_tensor_buffer_layerwise=[None for i in range(10)]
        self.inner_product_buffer=[None for i in range(size)]
        self.momentum=0.99
        self.iteration=0
        self.count=0.0
        self.selected_indices=None

    def forward1(self, grad_group,val_grad):
        inner_product_group=self.get_gradient_norm(grad_group,val_grad).view(-1,1).cuda()
        x = inner_product_group*self.trans_mat*self.trans_mat
        #print('x=',x)
        #x=F.relu(x)
        mixing_weight=F.softmax(x.view(-1))
        #print('trans_mat=',self.trans_mat)
        #print(mixing_weight)
        return mixing_weight.view(-1)
    def forward(self, grad_group,val_grad,model):
        mixing_weight=torch.zeros(len(grad_group))
        mixing_weight[self.rank]=1.0
        
        
        #mixing_weight=torch.ones(10)/10.0
        return mixing_weight.view(-1)
    
    def forward2(self, grad_group,val_grad,model):
        #self.get_index(model)
        #self.get_gradient_norm_group(grad_group,val_grad)
        #self.get_gradient_norm(grad_group,val_grad)
        #self.get_gradient_norm_inner_product_momentum(grad_group,val_grad)
        #self.get_gradient_norm_group_layerwise_momentum(grad_group,val_grad)
        #self.get_gradient_norm_pruned_group_layerwise(grad_group,val_grad,model)
        #self.get_gradient_norm2(grad_group,val_grad)
        
        '''
        mixing_weight=torch.zeros(len(grad_group))
        mixing_weight[self.rank]=1.0
        '''
        
        
        
        mixing_weight=torch.zeros(len(grad_group))
        a=self.rank//2
        a1=a*2
        a2=a*2+1
        mixing_weight[a1]=0.5
        mixing_weight[a2]=0.5
        
        
        
        #mixing_weight=torch.ones(10)/10.0
        return mixing_weight.view(-1)
    
    def update_momentum_buffer(self,grad_group,key_tensor):
        for ind,grad,grad_buffer in zip(range(self.size),grad_group,self.grad_group_buffer): 
            if grad_buffer==None:
                grad_buffer=torch.zeros_like(grad)
            
            grad_buffer=self.momentum*grad_buffer+(1-self.momentum)*grad
            self.grad_group_buffer[ind]=grad_buffer

        if self.key_tensor_buffer==None:
            self.key_tensor_buffer=torch.zeros_like(key_tensor)
        
        self.key_tensor_buffer=self.momentum*self.key_tensor_buffer+(1-self.momentum)*key_tensor

        return self.grad_group_buffer,self.key_tensor_buffer
    def update_avg_buffer(self,grad_group,key_tensor):
        for ind,grad,grad_buffer in zip(range(self.size),grad_group,self.grad_group_buffer): 
            if grad_buffer==None:
                grad_buffer=torch.zeros_like(grad)
            
            grad_buffer=grad_buffer+grad
            self.grad_group_buffer[ind]=grad_buffer

        if self.key_tensor_buffer==None:
            self.key_tensor_buffer=torch.zeros_like(key_tensor)
        
        self.key_tensor_buffer=self.key_tensor_buffer+key_tensor
        #self.count=self.count+1
        #we do not need add count, for we will do noramlization later

        return self.grad_group_buffer,self.key_tensor_buffer
    def get_gradient_norm2_1(self,grad_group,val_grad,model):
        if self.selected_indices==None:
            model_param_group=[]
            start_ind=0
            selected_indices=[]
            layer_ind=0
            for i in model.params():

                param_data=i.detach().data
                len_param=torch.numel(param_data)
                model_param_group.append(param_data)

                if layer_ind%2==0 and layer_ind>=4:
                    param_data=param_data.view(-1)
                    top_values,top_indices=torch.topk(torch.abs(param_data),k=max(int(len_param*0.1),1))
                    selected_indices.append(start_ind+top_indices)
                    if self.rank==0:
                        print('start_ind=',start_ind,'top_ind=',top_indices)
                start_ind=start_ind+len_param
                layer_ind=layer_ind+1

            selected_indices=torch.cat(selected_indices)
            self.selected_indices=selected_indices

        #val_grad=grad_group[self.rank]
        new_grad_group=[comm_helpers.flatten_tensors(i).detach() for i in grad_group]
        key_tensor=comm_helpers.flatten_tensors(val_grad).detach()

        

        new_grad_group,key_tensor=self.update_avg_buffer(new_grad_group,key_tensor)


        new_grad_group=[i.index_select(0,self.selected_indices) for i in new_grad_group]
        key_tensor=key_tensor.index_select(0,self.selected_indices)

        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        
        inner_product_group=[torch.sum(i*key_tensor) for i in new_grad_group]
        inner_product_group=torch.tensor(inner_product_group)
        
        return inner_product_group
    def get_gradient_norm2(self,grad_group,val_grad,model,model_update_group):
        val_grad=model_update_group[self.rank]
        new_grad_group=[comm_helpers.flatten_tensors(i).detach() for i in model_update_group]
        key_tensor=comm_helpers.flatten_tensors(val_grad).detach()

        

        #new_grad_group,key_tensor=self.update_avg_buffer(new_grad_group,key_tensor)

        '''
        new_grad_group=[i.index_select(0,self.selected_indices) for i in new_grad_group]
        key_tensor=key_tensor.index_select(0,self.selected_indices)
        '''

        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        
        inner_product_group=[torch.sum(i*key_tensor) for i in new_grad_group]
        inner_product_group=torch.tensor(inner_product_group)
        
        return inner_product_group
    

    
    
    
    
    