import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import torch.nn.init as init
import comm_helpers
import util
import numpy as np
import copy

def to_var(x, requires_grad=True):
    
    if torch.cuda.is_available():
        x = x.cuda()
    
    return Variable(x, requires_grad=requires_grad)


class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def params(self):
        for name, param in self.named_params(self):
            yield param
    def custom_get_name_param(self):
        for name, param in self.named_params(self):
            yield name,param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()  # https://blog.csdn.net/qq_39709535/article/details/81866686
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)


class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)

        self.register_buffer('weight', to_var(ignore.weight.clone().detach().data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.clone().detach().data, requires_grad=True))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaConv2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Conv2d(*args, **kwargs)

        self.in_channels = ignore.in_channels
        self.out_channels = ignore.out_channels
        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups
        self.kernel_size = ignore.kernel_size

        self.register_buffer('weight', to_var(ignore.weight.clone().detach().data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.clone().detach().data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaConvTranspose2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.ConvTranspose2d(*args, **kwargs)

        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x, output_size=None):
        output_padding = self._output_padding(x, output_size)
        return F.conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding,
                                  output_padding, self.groups, self.dilation)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaBatchNorm2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm2d(*args, **kwargs)

        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats

        if self.affine:
            self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                            self.training or not self.track_running_stats, self.momentum, self.eps)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]
class MetaGroupNorm(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.GroupNorm(*args, **kwargs)

        self.num_groups=ignore.num_groups
        self.num_channels=ignore.num_channels
        self.eps=ignore.eps

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

        

    def forward(self, x):
        return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]



def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, MetaLinear) or isinstance(m, MetaConv2d):
        init.kaiming_normal(m.weight)

class LambdaLayer(MetaModule):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(MetaModule):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = MetaConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = MetaGroupNorm(8,planes)
        self.conv2 = MetaConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = MetaGroupNorm(8,planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    MetaConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    MetaGroupNorm(8,self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18(MetaModule):
    def __init__(self, num_classes, block=BasicBlock, num_blocks=[3,3,3]):
        super(ResNet18, self).__init__()
        self.in_planes = 16

        self.conv1 = MetaConv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = MetaGroupNorm(8,16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = MetaLinear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



class VNet(MetaModule):
    def __init__(self, input, hidden1, output):
        super(VNet, self).__init__()
        self.linear1 = MetaLinear(input, hidden1)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = MetaLinear(hidden1, output)
        # self.linear3 = MetaLinear(hidden2, output)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        # x = self.linear2(x)
        # x = self.relu1(x)
        out = self.linear2(x)
        return F.sigmoid(out)
class Mixing_Net(nn.Module):
    def __init__(self,n_node,rank):
        super(Mixing_Net, self).__init__()

        init_tensor=torch.zeros(n_node)
        '''
        init_tensor=torch.ones(n_node)*(-3)
        init_tensor[rank]=3.0
        '''

        
        self.register_parameter(name='mixing_weight', param=torch.nn.Parameter(init_tensor))



    def forward(self):
        mixing_weight=F.sigmoid(self.mixing_weight)
        norm_c = torch.sum(mixing_weight).detach()

        if norm_c != 0:
            mixing_weight = mixing_weight / norm_c

        return mixing_weight
    def forward1(self):
        mixing_weight=torch.ones(10)/10.0
        return mixing_weight
class Mixing_Net_Sigmoid(nn.Module):
    def __init__(self,n_node):
        super(Mixing_Net_Sigmoid, self).__init__()

        

        
        self.register_parameter(name='mixing_weight', param=torch.nn.Parameter(torch.zeros(n_node)))



    def forward(self):
        mixing_weight=F.sigmoid(self.mixing_weight)/10.0
        return mixing_weight
    def forward1(self):
        mixing_weight=torch.ones(10)/10.0
        return mixing_weight
class SoftMaxMixing_Net(nn.Module):
    def __init__(self,n_node,rank):
        super(SoftMaxMixing_Net, self).__init__()

        
        self.n_node=n_node
        self.rank=rank

        
        self.register_parameter(name='mixing_weight', param=torch.nn.Parameter(torch.zeros(self.n_node)))



    def forward(self,a,b,c):
        mixing_weight=F.softmax(self.mixing_weight)
        

        return mixing_weight.view(-1)
    def forward2(self,a,b,c):
        '''
        mixing_weight=torch.zeros(self.n_node)
        mixing_weight[self.rank]=1.0
        '''
        
        #mixing_weight=torch.ones(10)/10.0
        
        
        mixing_weight=torch.zeros(self.n_node)
        a=self.rank//2
        a1=a*2
        a2=a*2+1
        mixing_weight[a1]=0.5
        mixing_weight[a2]=0.5
        
        return mixing_weight.view(-1)
    def forward1(self,a,b,c):
        '''
        mixing_weight=torch.zeros(self.n_node)
        mixing_weight[self.rank]=1.0
        '''
        
        #mixing_weight=torch.ones(10)/10.0
        
        
        mixing_weight=torch.zeros(self.n_node)
        mixing_weight[self.rank]=1.0
        
        return mixing_weight.view(-1)
    def reset_params(self):
        self.mixing_weight.data=torch.zeros(self.n_node)

class Mixing_Net_Attention(MetaModule):
    def __init__(self, input, hidden1, output, rank):
        super(Mixing_Net_Attention, self).__init__()
        self.linear1 = MetaLinear(input, hidden1)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = MetaLinear(hidden1, output)
        self.rank=rank
        # self.linear3 = MetaLinear(hidden2, output)

    def forward(self, grad_group):
        inner_product_group=self.get_gradient_norm(grad_group).view(-1,1).cuda()
        x = self.linear1(inner_product_group)
        x = self.relu1(x)
        # x = self.linear2(x)
        # x = self.relu1(x)
        out = self.linear2(x)#+inner_product_group.view(-1,1)

        mixing_weight=F.sigmoid(out)
        norm_c = torch.sum(mixing_weight)

        if norm_c != 0:
            mixing_weight = mixing_weight / norm_c
        return mixing_weight.view(-1)
    def forward1(self, grad_group):
        mixing_weight=torch.zeros(len(grad_group))
        mixing_weight[self.rank]=1.0
        mixing_weight=torch.ones(10)/10.0
        return mixing_weight.view(-1)
    def get_gradient_norm(self,grad_group):
        new_grad_group=[comm_helpers.flatten_tensors(i).detach() for i in grad_group]
        key_tensor=new_grad_group[self.rank]
        inner_product_group=[torch.sum(i*key_tensor) for i in new_grad_group]
        inner_product_group=torch.tensor(inner_product_group)
        #print('rank=',self.rank,inner_product_group)
        #note that inner_product_group does not have grad_fn after torch.tensor op.
        return inner_product_group
class Mixing_Net_Attention1(nn.Module):
    def __init__(self, input, hidden1, output, rank):
        super(Mixing_Net_Attention1, self).__init__()
        self.register_parameter(name='trans_mat', param=torch.nn.Parameter(torch.ones(1)))
        self.rank=rank

    def forward(self, grad_group):
        inner_product_group=self.get_gradient_norm(grad_group).view(-1,1).cuda()
        x = inner_product_group*self.trans_mat*self.trans_mat
        #print('x=',x)
        x=F.relu(x)
        mixing_weight=F.softmax(x.view(-1))
        #print('trans_mat=',self.trans_mat)
        #print(mixing_weight)
        return mixing_weight.view(-1)
    def forward1(self, grad_group):
        '''
        mixing_weight=torch.zeros(len(grad_group))
        mixing_weight[self.rank]=1.0
        '''
        '''
        mixing_weight=torch.zeros(len(grad_group))
        a=self.rank//2
        a1=a*2
        a2=a*2+1
        mixing_weight[a1]=0.5
        mixing_weight[a2]=0.5
        '''
        mixing_weight=torch.ones(10)/10.0
        return mixing_weight.view(-1)
    def normalize_gradient_group(self,grad_group):
        dim=grad_group[0].size(0)
        #print(dim,(float(dim))**(1.0/2))
        
        return [i/(float(dim))**(1.0/2) for i in grad_group]
    def get_gradient_norm(self,grad_group):
        new_grad_group=[comm_helpers.flatten_tensors(i).detach() for i in grad_group]
        #new_grad_group=self.normalize_gradient_group(new_grad_group)
        
        key_tensor=new_grad_group[self.rank]
        
        inner_product_group=[torch.sum(i*key_tensor) for i in new_grad_group]
        inner_product_group=torch.tensor(inner_product_group)*100
        #print('rank=',self.rank,inner_product_group)
        #note that inner_product_group does not have grad_fn after torch.tensor op.
        #print(inner_product_group)
        return inner_product_group
class CNNCifar(MetaModule):
    def __init__(self, num_classes):
        super(CNNCifar, self).__init__()
        self.conv1 = MetaConv2d(3, 6, kernel_size=5)
        self.conv2 = MetaConv2d(6, 16, kernel_size=5)
        self.fc1 = MetaLinear(16 * 5 * 5, 120)
        self.fc2 = MetaLinear(120, 100)
        self.fc3 = MetaLinear(100, num_classes)
        #self.dropout=nn.Dropout(0.25)
        # self.weight_keys = [['fc3.weight', 'fc3.bias'],
        #                     ['fc2.weight', 'fc2.bias'],
        #                     ['fc1.weight', 'fc1.bias'],
        #                     ['conv2.weight', 'conv2.bias'],
        #                     ['conv1.weight', 'conv1.bias'],
        #                     ]

        # self.weight_keys = [['conv1.weight', 'conv1.bias'],
        #                     ['conv2.weight', 'conv2.bias'],
        #                     ['fc2.weight', 'fc2.bias'],
        #                     ['fc3.weight', 'fc3.bias'],
        #                     ['fc1.weight', 'fc1.bias'],
        #                     ]

        

    def forward(self, x):
        x = nn.MaxPool2d(2, 2)(F.relu(self.conv1(x)))
        x = nn.MaxPool2d(2, 2)(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = self.dropout(x)
        x = self.fc3(x)
        #return F.log_softmax(x, dim=1)
        return x
class CNNminiimagenet(MetaModule):
    def __init__(self, num_classes):
        super(CNNminiimagenet, self).__init__()
        self.conv1 = MetaConv2d(3, 32, kernel_size=3)
        self.conv2 = MetaConv2d(32, 32, kernel_size=3)
        self.conv3 = MetaConv2d(32, 32, kernel_size=3)
        self.conv4 = MetaConv2d(32, 32, kernel_size=3)
        self.group_norm1=MetaGroupNorm(4,32)
        self.group_norm2=MetaGroupNorm(4,32)
        self.group_norm3=MetaGroupNorm(4,32)
        self.group_norm4=MetaGroupNorm(4,32)
        self.fc1 = MetaLinear(32 * 5 * 5, num_classes)
        #self.dropout=nn.Dropout(0.25)
        # self.weight_keys = [['fc3.weight', 'fc3.bias'],
        #                     ['fc2.weight', 'fc2.bias'],
        #                     ['fc1.weight', 'fc1.bias'],
        #                     ['conv2.weight', 'conv2.bias'],
        #                     ['conv1.weight', 'conv1.bias'],
        #                     ]

        # self.weight_keys = [['conv1.weight', 'conv1.bias'],
        #                     ['conv2.weight', 'conv2.bias'],
        #                     ['fc2.weight', 'fc2.bias'],
        #                     ['fc3.weight', 'fc3.bias'],
        #                     ['fc1.weight', 'fc1.bias'],
        #                     ]

        

    def forward(self, x):
        x = nn.MaxPool2d(2, 2)(self.group_norm1(F.relu(self.conv1(x))))
        x = nn.MaxPool2d(2, 2)(self.group_norm2(F.relu(self.conv2(x))))
        x = nn.MaxPool2d(2, 2)(self.group_norm3(F.relu(self.conv3(x))))
        x = nn.MaxPool2d(2, 1)(self.group_norm4(F.relu(self.conv4(x))))
        x = x.view(-1, 32 * 5 * 5)
        #x = self.dropout(x)
        x = self.fc1(x)
        #return F.log_softmax(x, dim=1)
        return x
class CNNCifar1(MetaModule):
    def __init__(self, num_classes):
        super(CNNCifar1, self).__init__()
        self.conv1 = MetaConv2d(3, 32, kernel_size=5)
        self.conv2 = MetaConv2d(32, 64, kernel_size=5)
        self.fc1 = MetaLinear(64 * 5 * 5, 512)
        self.fc2 = MetaLinear(512, num_classes)
        
        #self.dropout=nn.Dropout(0.25)
        # self.weight_keys = [['fc3.weight', 'fc3.bias'],
        #                     ['fc2.weight', 'fc2.bias'],
        #                     ['fc1.weight', 'fc1.bias'],
        #                     ['conv2.weight', 'conv2.bias'],
        #                     ['conv1.weight', 'conv1.bias'],
        #                     ]

        # self.weight_keys = [['conv1.weight', 'conv1.bias'],
        #                     ['conv2.weight', 'conv2.bias'],
        #                     ['fc2.weight', 'fc2.bias'],
        #                     ['fc3.weight', 'fc3.bias'],
        #                     ['fc1.weight', 'fc1.bias'],
        #                     ]

        

    def forward(self, x):
        x = nn.MaxPool2d(2, 2)(F.relu(self.conv1(x)))
        x = nn.MaxPool2d(2, 2)(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
       
        #return F.log_softmax(x, dim=1)
        return x
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

    def forward1(self, grad_group,val_grad):
        inner_product_group=self.get_gradient_norm(grad_group,val_grad).view(-1,1).cuda()
        x = inner_product_group*self.trans_mat*self.trans_mat
        #print('x=',x)
        #x=F.relu(x)
        mixing_weight=F.softmax(x.view(-1))
        #print('trans_mat=',self.trans_mat)
        #print(mixing_weight)
        return mixing_weight.view(-1)
    def forward2(self, grad_group):
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
    def normalize_gradient_group(self,grad_group):
        dim=grad_group[0].size(0)
        #print(dim,(float(dim))**(1.0/2))
        
        return [i/(float(dim))**(1.0/2) for i in grad_group]
    def get_gradient_norm1(self,grad_group,val_grad):
        new_grad_group=[comm_helpers.flatten_tensors(i).detach() for i in grad_group]
        #new_grad_group=self.normalize_gradient_group(new_grad_group)
        
        key_tensor=comm_helpers.flatten_tensors(val_grad).detach()
        
        inner_product_group=[torch.sum(i*key_tensor) for i in new_grad_group]
        inner_product_group=torch.tensor(inner_product_group)*100
        #print('rank=',self.rank,inner_product_group)
        #note that inner_product_group does not have grad_fn after torch.tensor op.
        #print(inner_product_group)
        return inner_product_group
    def get_index(self,model):
        if self.rank==0:
            for name, param in model.custom_get_name_param():
                print(name,param.size())
                #print(name,param.data.size())
    def forward(self, grad_group,val_grad,model):
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
    def update_momentum_buffer1(self,grad_group,key_tensor):
        for ind,grad,grad_buffer in zip(range(self.size),grad_group,self.grad_group_buffer): 
            if grad_buffer==None:
                grad_buffer=grad
            else:
                grad_buffer=self.momentum*grad_buffer+(1-self.momentum)*grad
            self.grad_group_buffer[ind]=grad_buffer

        if self.key_tensor_buffer==None:
            self.key_tensor_buffer=key_tensor
        else:
            self.key_tensor_buffer=self.momentum*self.key_tensor_buffer+(1-self.momentum)*key_tensor

        return self.grad_group_buffer,self.key_tensor_buffer
    def update_momentum_buffer(self,grad_group,key_tensor):
        for ind,grad,grad_buffer in zip(range(self.size),grad_group,self.grad_group_buffer): 
            if grad_buffer==None:
                grad_buffer=torch.zeros_like(grad)
            else:
                grad_buffer=self.momentum*grad_buffer+(1-self.momentum)*grad
            self.grad_group_buffer[ind]=grad_buffer

        if self.key_tensor_buffer==None:
            self.key_tensor_buffer=torch.zeros_like(key_tensor)
        else:
            self.key_tensor_buffer=self.momentum*self.key_tensor_buffer+(1-self.momentum)*key_tensor

        return self.grad_group_buffer,self.key_tensor_buffer
    def update_innner_product_momentum_buffer(self,inner_product_group):
        for ind,grad,grad_buffer in zip(range(self.size),inner_product_group,self.inner_product_buffer): 
            if grad_buffer==None:
                grad_buffer=grad
            else:
                grad_buffer=self.momentum*grad_buffer+(1-self.momentum)*grad
            self.inner_product_buffer[ind]=grad_buffer

        return self.inner_product_buffer

    def get_gradient_norm(self,grad_group,val_grad):
        new_grad_group=[comm_helpers.flatten_tensors(i).detach() for i in grad_group]
        key_tensor=comm_helpers.flatten_tensors(val_grad).detach()

        new_grad_group,key_tensor=self.update_momentum_buffer(new_grad_group,key_tensor)

        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        
        inner_product_group=[torch.sum(i*key_tensor) for i in new_grad_group]
        inner_product_group=torch.tensor(inner_product_group)
        #print('rank=',self.rank,inner_product_group)
        #note that inner_product_group does not have grad_fn after torch.tensor op.
        #print(inner_product_group)
        gradient_inner_product=util.get_mixing_mat(inner_product_group.view(-1),self.rank,self.size,self.comm)
        util.recorder_fast(gradient_inner_product,self.rank,name='val_prod')
        return inner_product_group
    def get_gradient_norm2_2(self,grad_group,val_grad):
        new_grad_group=[comm_helpers.flatten_tensors(i).detach() for i in grad_group]
        key_tensor=comm_helpers.flatten_tensors(val_grad).detach()


        new_grad_group=[torch.abs(i) for i in new_grad_group]
        key_tensor=torch.abs(key_tensor)

        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        '''
        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        '''
        
        #print('OK')
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
    def get_gradient_norm2_6(self,grad_group,val_grad):
        new_grad_group=[comm_helpers.flatten_tensors(i).detach() for i in grad_group]
        key_tensor=comm_helpers.flatten_tensors(val_grad).detach()


        
        
        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        
        inner_product_group=[torch.norm(i-key_tensor) for i in new_grad_group]
        inner_product_group=torch.tensor(inner_product_group)
        
        
        return inner_product_group.view(-1)
    def get_gradient_norm2_7(self,grad_group,val_grad):
        new_grad_group=[comm_helpers.flatten_tensors(i).detach() for i in grad_group]
        key_tensor=comm_helpers.flatten_tensors(val_grad).detach()

        new_grad_group,key_tensor=self.update_momentum_buffer(new_grad_group,key_tensor)
        
        
        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        
        inner_product_group=[torch.sum(i*key_tensor) for i in new_grad_group]
        inner_product_group=torch.tensor(inner_product_group)
        
        
        return inner_product_group.view(-1)
    def get_gradient_norm2_8(self,grad_group,val_grad):
        new_grad_group=[comm_helpers.flatten_tensors(i).detach() for i in grad_group]
        key_tensor=comm_helpers.flatten_tensors(val_grad).detach()


        new_grad_group=[torch.abs(i) for i in new_grad_group]
        key_tensor=torch.abs(key_tensor)

        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        '''
        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        '''
        
        #print('OK')
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
    def get_gradient_norm2_14(self,grad_group,val_grad,model):
        layer_index=-2
        model_param_group=[]
        for i in model.params():
            model_param_group.append(i.detach().data)

        layer_param=model_param_group[layer_index]
        layer_param_abs_sum=torch.sum(torch.abs(layer_param),dim=1)
        if self.rank==0:
            print(layer_param_abs_sum)
        top_values,top_indices=torch.topk(layer_param_abs_sum,k=2)
        '''
        a=self.rank//2
        a1=a*2
        a2=a*2+1
        selected_ind=torch.LongTensor([a1,a2]).cuda()
        '''
        new_grad_group=[i[layer_index].index_select(0,top_indices) for i in grad_group]
        key_tensor=val_grad[layer_index].index_select(0,top_indices)

        new_grad_group=[torch.sum(i,dim=0) for i in new_grad_group]
        key_tensor=torch.sum(key_tensor,dim=0)
        '''
        new_grad_group=[i[a1]+i[a2] for i in new_grad_group]
        key_tensor=key_tensor[a1]+key_tensor[a2]
        '''
        
        

        new_grad_group=[comm_helpers.flatten_tensors([i]).detach() for i in new_grad_group]
        key_tensor=comm_helpers.flatten_tensors([key_tensor]).detach()


        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        '''
        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        '''
        
        #print('OK')
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
    def get_gradient_norm2_16(self,grad_group,val_grad,model):
        layer_index=-2
        model_param_group=[]
        for i in model.params():
            model_param_group.append(i.detach().data)

        layer_param=model_param_group[layer_index]
        layer_param_abs_sum=torch.sum(torch.abs(layer_param),dim=0)
        
        top_values,top_indices=torch.topk(layer_param_abs_sum,k=10)
        if self.rank==0:
            print(top_values)
            print(top_indices)
        '''
        a=self.rank//2
        a1=a*2
        a2=a*2+1
        selected_ind=torch.LongTensor([a1,a2]).cuda()
        '''
        new_grad_group=[i[layer_index].index_select(1,top_indices) for i in grad_group]
        key_tensor=val_grad[layer_index].index_select(1,top_indices)

        
        new_grad_group=[torch.sum(i,dim=0) for i in new_grad_group]
        key_tensor=torch.sum(key_tensor,dim=0)
        
        '''
        new_grad_group=[i[a1]+i[a2] for i in new_grad_group]
        key_tensor=key_tensor[a1]+key_tensor[a2]
        '''
        
        

        new_grad_group=[comm_helpers.flatten_tensors([i]).detach() for i in new_grad_group]
        key_tensor=comm_helpers.flatten_tensors([key_tensor]).detach()


        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        '''
        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        '''
        
        #print('OK')
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
    def get_gradient_norm2_22(self,grad_group,val_grad,model):
        layer_index=-4
        model_param_group=[]
        for i in model.params():
            model_param_group.append(i.detach().data)

        layer_param=model_param_group[layer_index].view(-1)
        layer_param_abs_sum=torch.abs(layer_param)
        
        
        '''
        if self.rank==0:
            print(top_values)
            print(top_indices)
        '''
        '''
        new_grad_group=[i[layer_index].view(-1).index_select(0,top_indices) for i in grad_group]
        key_tensor=val_grad[layer_index].view(-1).index_select(0,top_indices)
        '''
        key_tensor=grad_group[self.rank]
        '''
        new_grad_group=[F.relu(-i[layer_index]) for i in grad_group]
        key_tensor=F.relu(-val_grad[layer_index])
        '''
        
        
        '''
        new_grad_group=[i[a1]+i[a2] for i in new_grad_group]
        key_tensor=key_tensor[a1]+key_tensor[a2]
        '''
        
        

        new_grad_group=[comm_helpers.flatten_tensors(i).detach() for i in grad_group]
        key_tensor=comm_helpers.flatten_tensors(key_tensor).detach()
        '''
        new_grad_group=[F.relu(-i) for i in new_grad_group]
        key_tensor=F.relu(-key_tensor)
        '''


        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)


        '''
        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        '''
        
        #print('OK')
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
    def get_gradient_norm2(self,grad_group,val_grad,model,model_sub):
        layer_index=-4
        
        layer_param=model_sub[layer_index].view(-1)
        layer_param_abs_sum=torch.abs(layer_param)
        
        top_values,top_indices=torch.topk(layer_param_abs_sum,k=100)
        if self.rank==0 or self.rank==2:
            #print(layer_param[top_indices])
            print('rank=',self.rank,top_indices)
        '''
        a=self.rank//2
        a1=a*2
        a2=a*2+1
        selected_ind=torch.LongTensor([a1,a2]).cuda()
        '''
        new_grad_group=[i[layer_index].view(-1).index_select(0,top_indices) for i in grad_group]
        key_tensor=val_grad[layer_index].view(-1).index_select(0,top_indices)

        
        

        new_grad_group=[comm_helpers.flatten_tensors([i]).detach() for i in new_grad_group]
        key_tensor=comm_helpers.flatten_tensors([key_tensor]).detach()


        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        '''
        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        '''
        
        #print('OK')
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

    def get_gradient_norm2_21(self,grad_group,val_grad,model):
        layer_index=-6
        '''
        model_param_group=[]
        for i in model.params():
            model_param_group.append(i.detach().data)

        layer_param=model_param_group[layer_index]
        layer_param_abs_sum=torch.sum(torch.abs(layer_param),dim=0)
        
        top_values,top_indices=torch.topk(layer_param_abs_sum,k=10)
        if self.rank==0:
            print(top_values)
            print(top_indices)
        '''
       
        new_grad_group=[i[layer_index] for i in grad_group]
        key_tensor=val_grad[layer_index]

        
        new_grad_group=[torch.abs(i) for i in new_grad_group]
        key_tensor=torch.abs(key_tensor)
        
        '''
        new_grad_group=[i[a1]+i[a2] for i in new_grad_group]
        key_tensor=key_tensor[a1]+key_tensor[a2]
        '''
        
        

        new_grad_group=[comm_helpers.flatten_tensors([i]).detach() for i in new_grad_group]
        key_tensor=comm_helpers.flatten_tensors([key_tensor]).detach()


        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        '''
        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        '''
        
        #print('OK')
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
    def get_gradient_norm2_11(self,grad_group,val_grad,model):
        model_param_group=[]
        for i in model.params():
            model_param_group.append(i.detach().data)

        layer_param=model_param_group[-2]
        max_value,max_ind=torch.max(torch.abs(layer_param),dim=1)

        if self.rank==0:
            print('max_ind=',max_ind)
            print('max_value=',max_value)

        new_grad_group=[i[-2] for i in grad_group]
        key_tensor=val_grad[-2]

    

        new_grad_group=[comm_helpers.flatten_tensors([i]).detach() for i in new_grad_group]
        key_tensor=comm_helpers.flatten_tensors([key_tensor]).detach()


        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        '''
        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        '''
        
        #print('OK')
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
    def get_gradient_norm2_12(self,grad_group,val_grad,model):
        model_param_group=[]
        for i in model.params():
            model_param_group.append(i.detach().data)

        layer_param=model_param_group[-2].view(-1)
        max_value,max_ind=torch.max(torch.abs(layer_param),dim=0)
        
        if self.rank==0:
            print('max_ind=',max_ind)
            print('max_value=',max_value)
        

        new_grad_group=[i[-2] for i in grad_group]
        key_tensor=val_grad[-2]

    

        new_grad_group=[comm_helpers.flatten_tensors([i]).detach() for i in new_grad_group]
        key_tensor=comm_helpers.flatten_tensors([key_tensor]).detach()

        new_grad_group=[i[max_ind] for i in new_grad_group]
        key_tensor=key_tensor[max_ind]

        if self.rank==0:
            print(new_grad_group)
            print(key_tensor)


        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        '''
        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        '''
        
        #print('OK')
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
    def get_gradient_norm2_9(self,grad_group,val_grad):
        new_grad_group=[i[-1] for i in grad_group]
        key_tensor=val_grad[-1]

        new_grad_group=[i[self.rank] for i in new_grad_group]
        key_tensor=key_tensor[self.rank]

        #print([i.size() for i in new_grad_group])

        new_grad_group=[comm_helpers.flatten_tensors([i]).detach() for i in new_grad_group]
        key_tensor=comm_helpers.flatten_tensors([key_tensor]).detach()


        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        '''
        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        '''
        
        #print('OK')
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
    def get_gradient_norm2_3(self,grad_group,val_grad):
        new_grad_group=[comm_helpers.flatten_tensors(i).detach() for i in grad_group]
        key_tensor=comm_helpers.flatten_tensors(val_grad).detach()


        

        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        '''
        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        '''
        
        #print('OK')
        inner_product_group=[torch.abs(torch.sum(i*key_tensor)) for i in new_grad_group]
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
    def get_gradient_norm2_5(self,grad_group,val_grad):
        new_grad_group=[comm_helpers.flatten_tensors(i).detach() for i in grad_group]
        key_tensor=comm_helpers.flatten_tensors(val_grad).detach()


        new_grad_group=[torch.nn.functional.relu(i) for i in new_grad_group]
        key_tensor=torch.nn.functional.relu(key_tensor)

        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        '''
        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        '''
        
        #print('OK')
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
    def get_gradient_norm2_4(self,grad_group,val_grad):
        new_grad_group=[comm_helpers.flatten_tensors(i).detach() for i in grad_group]
        key_tensor=comm_helpers.flatten_tensors(val_grad).detach()

        new_grad_group_max=[torch.max(torch.abs(i)) for i in new_grad_group]
        key_tensor_max=torch.max(torch.abs(key_tensor))

        threshold=0.01

        new_grad_group_indicator=[torch.abs(i)>(threshold*j) for i,j in zip(new_grad_group,new_grad_group_max)]
        key_tensor_indicator=key_tensor>(threshold*key_tensor_max)

        print([torch.sum(i) for i in new_grad_group_indicator])
        print(torch.sum(key_tensor_indicator))

        new_grad_group=[i*j for i,j in zip(new_grad_group,new_grad_group_indicator)]
        key_tensor=key_tensor*key_tensor_indicator

        '''
        new_grad_group=[torch.abs(i) for i in new_grad_group]
        key_tensor=torch.abs(key_tensor)

        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        '''
        '''
        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        '''
        
        #print('OK')
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
    def get_gradient_norm_inner_product_momentum(self,grad_group,val_grad):
        new_grad_group=[comm_helpers.flatten_tensors(i).detach() for i in grad_group]
        key_tensor=comm_helpers.flatten_tensors(val_grad).detach()

        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        
        inner_product_group=[torch.sum(i*key_tensor) for i in new_grad_group]
        inner_product_group=torch.tensor(inner_product_group)
        #print('rank=',self.rank,inner_product_group)
        #note that inner_product_group does not have grad_fn after torch.tensor op.
        #print(inner_product_group)
        inner_product_group=self.update_innner_product_momentum_buffer(inner_product_group.view(-1))
        gradient_inner_product=util.get_mixing_mat(torch.tensor(inner_product_group).view(-1),self.rank,self.size,self.comm)
        util.recorder_fast(gradient_inner_product,self.rank,name='val_prod')
        return inner_product_group
    def get_gradient_norm_group_layerwise(self,grad_group,val_grad):
        self.iteration=self.iteration+1
        
        for layer_index in range(len(val_grad)):
            new_grad_group=[comm_helpers.flatten_tensors(i[layer_index]).detach() for i in grad_group]
            key_tensor=comm_helpers.flatten_tensors(val_grad[layer_index]).detach()
            print(layer_index)
            #new_grad_group,key_tensor=self.update_momentum_buffer(new_grad_group,key_tensor)

            new_grad_group=[i/torch.norm(i) for i in new_grad_group]
            key_tensor=key_tensor/torch.norm(key_tensor)
            
            inner_product_group=[torch.sum(i*key_tensor) for i in new_grad_group]
            inner_product_group=torch.tensor(inner_product_group)
            #print('rank=',self.rank,inner_product_group)
            #note that inner_product_group does not have grad_fn after torch.tensor op.
            #print(inner_product_group)
            if self.iteration%20==0:

                gradient_inner_product=util.get_mixing_mat(inner_product_group.view(-1),self.rank,self.size,self.comm)
                
                util.recorder_fast(gradient_inner_product,self.rank,name=str(layer_index)+'_val_prod')
        return inner_product_group
    def update_momentum_buffer_layerwise(self,grad_group,key_tensor,index):
        for ind,grad,grad_buffer in zip(range(self.size),grad_group,self.grad_group_buffer_layerwise[index]):
            if grad_buffer==None:
                grad_buffer=grad
            else:
                grad_buffer=self.momentum*grad_buffer+(1-self.momentum)*grad
            self.grad_group_buffer_layerwise[index][ind]=grad_buffer

        if self.key_tensor_buffer_layerwise[index]==None:
            self.key_tensor_buffer_layerwise[index]=key_tensor
        else:
            self.key_tensor_buffer_layerwise[index]=self.momentum*self.key_tensor_buffer_layerwise[index]+(1-self.momentum)*key_tensor

        return self.grad_group_buffer_layerwise[index],self.key_tensor_buffer_layerwise[index]
    def get_gradient_norm_group_layerwise_momentum(self,grad_group,val_grad):
        self.iteration=self.iteration+1
        
        for layer_index in range(len(val_grad)):
            new_grad_group=[comm_helpers.flatten_tensors(i[layer_index]).detach() for i in grad_group]
            key_tensor=comm_helpers.flatten_tensors(val_grad[layer_index]).detach()

            #print(layer_index)
            new_grad_group,key_tensor=self.update_momentum_buffer_layerwise(new_grad_group,key_tensor,layer_index)

            new_grad_group=[i/torch.norm(i) for i in new_grad_group]
            key_tensor=key_tensor/torch.norm(key_tensor)
            
            inner_product_group=[torch.sum(i*key_tensor) for i in new_grad_group]
            inner_product_group=torch.tensor(inner_product_group)
            #print('rank=',self.rank,inner_product_group)
            #note that inner_product_group does not have grad_fn after torch.tensor op.
            #print(inner_product_group)
            if self.iteration%20==0:

                gradient_inner_product=util.get_mixing_mat(inner_product_group.view(-1),self.rank,self.size,self.comm)
                
                util.recorder_fast(gradient_inner_product,self.rank,name=str(layer_index)+'_val_prod')
        return inner_product_group
    def prune_conv_kernel(self,kernel):
        norm_a=torch.norm(kernel,p=1,dim=(1,2,3))
        max_ind=torch.argmax(norm_a)
        return max_ind
    def prune_fc_weight(self,kernel):
        norm_a=torch.norm(kernel,p=1,dim=1)
        max_ind=torch.argmax(norm_a)
        return max_ind
    def prune_single_weight(self,kernel):
        max_ind=torch.argmax(torch.abs(kernel))
        return max_ind
    def replicate_grads(self,grad_group,val_grad):
        new_grad_group=[]
        for i in range(len(grad_group)):
            tmp=[]
            for j in range(len(grad_group[i])):
                grad_=grad_group[i][j].data
                tmp.append(grad_)
            new_grad_group.append(tmp)
        new_val_grad=[]
        for j in range(len(val_grad)):
            grad_=val_grad[j].data
            new_val_grad.append(grad_)
        return new_grad_group,new_val_grad
    def get_gradient_norm_pruned_group(self,grad_group,val_grad,model):
        self.iteration=self.iteration+1
        grad_group,val_grad=self.replicate_grads(grad_group,val_grad)
        params=list(model.params())
        for i in range(len(params)):
            grad_group[i]=list(grad_group[i])
        val_grad=list(val_grad)
        for i in range(len(params)):
            kernel=params[i].data
            if i in [0,2]:
                ind=self.prune_conv_kernel(kernel)
                for m in range(len(grad_group)):
                    grad_=grad_group[m][i]
                    grad_=grad_[ind]
                    grad_group[m][i]=grad_
                val_grad[i]=val_grad[i][ind]
            '''
            if i in [1,3,5,7,9]:
                ind=self.prune_single_weight(kernel)
                for m in range(len(grad_group)):
                    grad_=grad_group[m][i]
                    grad_=grad_[ind]
                    grad_group[m][i]=grad_.view(1)
                val_grad[i]=val_grad[i][ind]
            '''
            if i in [4,6,8]:
                ind=self.prune_fc_weight(kernel)
                for m in range(len(grad_group)):
                    grad_=grad_group[m][i]
                    grad_=grad_[ind]
                    grad_group[m][i]=grad_
                val_grad[i]=val_grad[i][ind]
        
        for layer_index in range(len(val_grad)):
            new_grad_group=[comm_helpers.flatten_tensors(i[layer_index]).detach() for i in grad_group]
            key_tensor=comm_helpers.flatten_tensors(val_grad[layer_index]).detach()

            #new_grad_group,key_tensor=self.update_momentum_buffer(new_grad_group,key_tensor)

            new_grad_group=[i/torch.norm(i) for i in new_grad_group]
            key_tensor=key_tensor/torch.norm(key_tensor)
            
            inner_product_group=[torch.sum(i*key_tensor) for i in new_grad_group]
            inner_product_group=torch.tensor(inner_product_group)
            #print('rank=',self.rank,inner_product_group)
            #note that inner_product_group does not have grad_fn after torch.tensor op.
            #print(inner_product_group)
            if self.iteration%20==0:

                gradient_inner_product=util.get_mixing_mat(inner_product_group.view(-1),self.rank,self.size,self.comm)
                
                util.recorder_fast(gradient_inner_product,self.rank,name=str(layer_index)+'_val_prod')
        return inner_product_group
    def get_gradient_norm_pruned_group_layerwise(self,grad_group,val_grad,model):
        self.iteration=self.iteration+1
        grad_group,val_grad=self.replicate_grads(grad_group,val_grad)
        params=list(model.params())
        for i in range(len(params)):
            grad_group[i]=list(grad_group[i])
        val_grad=list(val_grad)
        for i in range(len(params)):
            kernel=params[i].data
            if i in [0,2]:
                ind=self.prune_conv_kernel(kernel)
                for m in range(len(grad_group)):
                    grad_=grad_group[m][i]
                    grad_=grad_[ind]
                    grad_group[m][i]=grad_
                val_grad[i]=val_grad[i][ind]
            '''
            if i in [1,3,5,7,9]:
                ind=self.prune_single_weight(kernel)
                for m in range(len(grad_group)):
                    grad_=grad_group[m][i]
                    grad_=grad_[ind]
                    grad_group[m][i]=grad_.view(1)
                val_grad[i]=val_grad[i][ind]
            '''
            if i in [4,6,8]:
                ind=self.prune_fc_weight(kernel)
                for m in range(len(grad_group)):
                    grad_=grad_group[m][i]
                    grad_=grad_[ind]
                    grad_group[m][i]=grad_
                val_grad[i]=val_grad[i][ind]
        for layer_index in range(len(val_grad)):
            new_grad_group=[comm_helpers.flatten_tensors(i[layer_index]).detach() for i in grad_group]
            key_tensor=comm_helpers.flatten_tensors(val_grad[layer_index]).detach()

            #print(layer_index)
            new_grad_group,key_tensor=self.update_momentum_buffer_layerwise(new_grad_group,key_tensor,layer_index)

            new_grad_group=[i/torch.norm(i) for i in new_grad_group]
            key_tensor=key_tensor/torch.norm(key_tensor)
            
            inner_product_group=[torch.sum(i*key_tensor) for i in new_grad_group]
            inner_product_group=torch.tensor(inner_product_group)
            #print('rank=',self.rank,inner_product_group)
            #note that inner_product_group does not have grad_fn after torch.tensor op.
            #print(inner_product_group)
            if self.iteration%20==0:

                gradient_inner_product=util.get_mixing_mat(inner_product_group.view(-1),self.rank,self.size,self.comm)
                
                util.recorder_fast(gradient_inner_product,self.rank,name=str(layer_index)+'_val_prod')

        
        return inner_product_group

class Mixing_Net_Attention3(nn.Module):
    def __init__(self, input, hidden1, output, rank, comm,size):
        super(Mixing_Net_Attention3, self).__init__()
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

    def forward(self, grad_group,val_grad,model):
        #print(len(list(model.params())))
        inner_product_group=self.get_gradient_norm_inner_product_momentum(grad_group,val_grad).cuda()
        x = inner_product_group*self.trans_mat*self.trans_mat
        x = F.relu(x)
        mixing_weight =x/torch.sum(x)
        #mixing_weight=F.softmax(x.view(-1),dim=0)
        return mixing_weight.view(-1)
    def forward2(self, grad_group):
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
    
    def get_gradient_norm1(self,grad_group,val_grad):
        new_grad_group=[comm_helpers.flatten_tensors(i).detach() for i in grad_group]
        #new_grad_group=self.normalize_gradient_group(new_grad_group)
        
        key_tensor=comm_helpers.flatten_tensors(val_grad).detach()
        
        inner_product_group=[torch.sum(i*key_tensor) for i in new_grad_group]
        inner_product_group=torch.tensor(inner_product_group)*100
        #print('rank=',self.rank,inner_product_group)
        #note that inner_product_group does not have grad_fn after torch.tensor op.
        #print(inner_product_group)
        return inner_product_group
    def get_index(self,model):
        if self.rank==0:
            for name, param in model.custom_get_name_param():
                print(name,param.size())
                #print(name,param.data.size())
    def forward1(self, grad_group,val_grad,model):
        #self.get_index(model)
        #self.get_gradient_norm_group(grad_group,val_grad)
        #self.get_gradient_norm(grad_group,val_grad)
        #self.get_gradient_norm_inner_product_momentum(grad_group,val_grad)
        #self.get_gradient_norm_group_layerwise_momentum(grad_group,val_grad)
        inner_product_group=self.get_gradient_norm_inner_product_momentum(grad_group,val_grad)
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
    
    def update_innner_product_momentum_buffer(self,inner_product_group):
        for ind,grad,grad_buffer in zip(range(self.size),inner_product_group,self.inner_product_buffer): 
            if grad_buffer==None:
                grad_buffer=grad
            else:
                grad_buffer=self.momentum*grad_buffer+(1-self.momentum)*grad
            self.inner_product_buffer[ind]=grad_buffer

        return self.inner_product_buffer

    
    def get_gradient_norm_inner_product_momentum(self,grad_group,val_grad):
        new_grad_group=[comm_helpers.flatten_tensors(i).detach() for i in grad_group]
        key_tensor=comm_helpers.flatten_tensors(val_grad).detach()
        print(len(key_tensor))

        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        
        inner_product_group=[torch.sum(i*key_tensor) for i in new_grad_group]
        inner_product_group=torch.tensor(inner_product_group)
        #print('rank=',self.rank,inner_product_group)
        #note that inner_product_group does not have grad_fn after torch.tensor op.
        #print(inner_product_group)
        inner_product_group=self.update_innner_product_momentum_buffer(inner_product_group.view(-1))
        gradient_inner_product=util.get_mixing_mat(torch.tensor(inner_product_group).view(-1),self.rank,self.size,self.comm)
        util.recorder_fast(gradient_inner_product,self.rank,name='val_prod')
        return torch.tensor(inner_product_group).view(-1)
    
   
    def prune_conv_kernel(self,kernel):
        norm_a=torch.norm(kernel,p=1,dim=(1,2,3))
        max_ind=torch.argmax(norm_a)
        return max_ind
    def prune_fc_weight(self,kernel):
        norm_a=torch.norm(kernel,p=1,dim=1)
        max_ind=torch.argmax(norm_a)
        return max_ind
    def prune_single_weight(self,kernel):
        max_ind=torch.argmax(torch.abs(kernel))
        return max_ind
    def replicate_grads(self,grad_group,val_grad):
        new_grad_group=[]
        for i in range(len(grad_group)):
            tmp=[]
            for j in range(len(grad_group[i])):
                grad_=grad_group[i][j].data
                tmp.append(grad_)
            new_grad_group.append(tmp)
        new_val_grad=[]
        for j in range(len(val_grad)):
            grad_=val_grad[j].data
            new_val_grad.append(grad_)
        return new_grad_group,new_val_grad
    

class Attention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, in_dim, out_dim, rank, comm,size):
        super().__init__()


        self.in_dim=in_dim
        self.out_dim=out_dim

        self.w_qs = nn.Linear(in_dim, out_dim, bias=False)
        self.w_ks = nn.Linear(in_dim, out_dim, bias=False)


        self.rank=rank
        self.comm=comm
        self.size=size
        self.grad_group_buffer=[None for i in range(size)]
        self.key_tensor_buffer=None
        self.grad_group_buffer_layerwise=[[None for i in range(size)] for j in range(10)]
        self.key_tensor_buffer_layerwise=[None for i in range(10)]
        self.inner_product_buffer=[None for i in range(size)]
        self.inner_product_buffer_forward=[None for i in range(size)]
        self.mixing_weight_buffer=None
        self.momentum=0.99
        self.mixing_momentum=0.99
        #self.register_parameter(name='mixing_momentum', param=torch.nn.Parameter(0.5*torch.ones(1)))
        self.iteration=0
        


    def forward(self, grad_group,val_grad,model):
        q,k=self.preprocess3(grad_group,val_grad)

        '''
        q=q*50
        k=k*50
        '''
        
        
        
        q = self.w_qs(q)
        k = self.w_ks(k.view(1,-1)).view(-1,1)



        mixing_weight=torch.matmul(q,k)
        #mixing_weight=mixing_weight/(self.out_dim**0.5)
        mixing_weight=mixing_weight.view(-1)
        print(mixing_weight)
        #print(mixing_weight)
        mixing_weight=F.softmax(mixing_weight,dim=0)

        mixing_weight=mixing_weight.view(-1)



        return mixing_weight
    def update_mixing_momentum_buffer(self,mixing_weight):
        if self.mixing_weight_buffer==None:
            self.mixing_weight_buffer=mixing_weight.detach()
            new_mixing_weight=mixing_weight
        else:
            self.mixing_weight_buffer=self.mixing_momentum*self.mixing_weight_buffer+\
            (1-self.mixing_momentum)*mixing_weight.detach()
            new_mixing_weight=self.mixing_momentum*self.mixing_weight_buffer+\
            (1-self.mixing_momentum)*mixing_weight
        #print('.......',new_mixing_weight)
        return new_mixing_weight
    def update_innner_product_momentum_buffer(self,inner_product_group):
        for ind,grad,grad_buffer in zip(range(self.size),inner_product_group,self.inner_product_buffer): 
            if grad_buffer==None:
                grad_buffer=grad
            else:
                grad_buffer=self.mixing_momentum*grad_buffer+(1-self.mixing_momentum)*grad
            self.inner_product_buffer[ind]=grad_buffer.detach()
            self.inner_product_buffer_forward[ind]=grad_buffer
        print('.......',self.inner_product_buffer_forward)

        return inner_product_group
    def update_momentum_buffer(self,grad_group,key_tensor):
        for ind,grad,grad_buffer in zip(range(self.size),grad_group,self.grad_group_buffer): 
            if grad_buffer==None:
                grad_buffer=grad
            else:
                grad_buffer=self.momentum*grad_buffer+(1-self.momentum)*grad
            self.grad_group_buffer[ind]=grad_buffer

        if self.key_tensor_buffer==None:
            self.key_tensor_buffer=key_tensor
        else:
            self.key_tensor_buffer=self.momentum*self.key_tensor_buffer+(1-self.momentum)*key_tensor

        return self.grad_group_buffer,self.key_tensor_buffer
    def preprocess(self,grad_group,val_grad):
        new_grad_group=[comm_helpers.flatten_tensors(i).detach() for i in grad_group]
        key_tensor=comm_helpers.flatten_tensors(val_grad).detach()
        




        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        #new_grad_group,key_tensor=self.update_momentum_buffer(new_grad_group,key_tensor)
        #print(new_grad_group)
        new_grad_group=torch.stack(new_grad_group)#conver list of tensors to a pytorch tensor
        '''
        print('mean=',[torch.mean(i) for i in new_grad_group])
        print('abs=',[torch.mean(torch.abs(i)) for i in new_grad_group])
        '''
        return new_grad_group,key_tensor
    def preprocess1(self,grad_group,val_grad):
        new_grad_group=[comm_helpers.flatten_tensors(i).detach() for i in grad_group]
        key_tensor=comm_helpers.flatten_tensors(val_grad).detach()
        




        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        #print(new_grad_group)
        new_grad_group=torch.stack(new_grad_group)#conver list of tensors to a pytorch tensor
        
        return new_grad_group,key_tensor
    def preprocess2(self,grad_group,val_grad):
        new_grad_group=[comm_helpers.flatten_tensors(i).detach() for i in grad_group]
        key_tensor=comm_helpers.flatten_tensors(val_grad).detach()
        




        new_grad_group=[i-torch.mean(i) for i in new_grad_group]
        key_tensor=key_tensor-torch.mean(key_tensor)

        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        #new_grad_group,key_tensor=self.update_momentum_buffer(new_grad_group,key_tensor)
        #print(new_grad_group)
        new_grad_group=torch.stack(new_grad_group)#conver list of tensors to a pytorch tensor
        '''
        print('mean=',[torch.mean(i) for i in new_grad_group])
        print('abs=',[torch.mean(torch.abs(i)) for i in new_grad_group])
        '''
        return new_grad_group,key_tensor
    def preprocess3(self,grad_group,val_grad):
        new_grad_group=[comm_helpers.flatten_tensors(i).detach() for i in grad_group]
        key_tensor=comm_helpers.flatten_tensors(val_grad).detach()



        new_grad_group=[torch.clamp(i,-0.1,0.1) for i in new_grad_group]
        key_tensor=torch.clamp(key_tensor,-0.1,0.1)
        




        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        #new_grad_group,key_tensor=self.update_momentum_buffer(new_grad_group,key_tensor)
        #print(new_grad_group)
        new_grad_group=torch.stack(new_grad_group)#conver list of tensors to a pytorch tensor
        '''
        print('mean=',[torch.mean(i) for i in new_grad_group])
        print('abs=',[torch.mean(torch.abs(i)) for i in new_grad_group])
        '''
        return new_grad_group,key_tensor


class Attention_layerwise(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, model, out_dim,rank, comm,size):
        super().__init__()


        #self.in_dim=in_dim
        self.out_dim=out_dim
        self.in_dim_group=[]
        self.w_qs_group=[]
        self.w_ks_group=[]
        self.num_weights=0
        for i in model.params():
            param_data=i.detach().data
            param_data=param_data.view(param_data.size(0),-1)
            param_size=param_data.size()
            in_dim=param_size[1]
            self.in_dim_group.append(in_dim)
            self.w_qs_group.append(nn.Linear(in_dim, out_dim, bias=False))
            self.w_ks_group.append(nn.Linear(in_dim, out_dim, bias=False))
            self.num_weights=self.num_weights+1
            '''
            self.w_qs_group[-1].weight.data=self.w_qs_group[-1].weight.data
            self.w_ks_group[-1].weight.data=self.w_ks_group[-1].weight.data*10
            '''
        self.w_qs_layers = nn.ModuleList(self.w_qs_group)
        self.w_ks_layers = nn.ModuleList(self.w_ks_group)
            


        self.rank=rank
        self.comm=comm
        self.size=size
        self.grad_group_buffer=[None for i in range(size)]
        self.key_tensor_buffer=None
        self.grad_group_buffer_layerwise=[[None for i in range(size)] for j in range(10)]
        self.key_tensor_buffer_layerwise=[None for i in range(10)]
        self.inner_product_buffer=[None for i in range(size)]
        self.inner_product_buffer_forward=[None for i in range(size)]
        self.mixing_weight_buffer=None
        self.momentum=0.99
        self.mixing_momentum=0.99
        #self.register_parameter(name='mixing_momentum', param=torch.nn.Parameter(0.5*torch.ones(1)))
        self.iteration=0
        


    def forward(self, grad_group,val_grad,model):
        q_group,k_group=self.preprocess(grad_group,val_grad)
        #q_group size=[num_weights,neighbour_num,num_filter,in_dim] list of tensors
        #k_group size=[num_weights,num_filter,in_dim] list of tensors

        '''
        q=q*50
        k=k*50
        '''
        
        prod_sum=torch.zeros(self.size).cuda()
        for i in range(self.num_weights):
            q=q_group[i]
            k=k_group[i]
            neighbour_num,num_filter,in_dim=q.size()
            q = self.w_qs_layers[i](q)
            k = self.w_qs_layers[i](k)

            q=q.view(neighbour_num,-1)
            k=k.view(-1,1)



            prod=torch.matmul(q,k).view(-1)/num_filter
            prod_sum=prod_sum+prod
            #mixing_weight=mixing_weight/(self.out_dim**0.5)
        prod_sum=prod_sum/self.num_weights
        prod_sum=prod_sum.view(-1)
        #print('rank=',self.rank,prod_sum)
        #print(mixing_weight)
        mixing_weight=F.softmax(prod_sum,dim=0)

        mixing_weight=mixing_weight.view(-1)



        return mixing_weight
    def update_mixing_momentum_buffer(self,mixing_weight):
        if self.mixing_weight_buffer==None:
            self.mixing_weight_buffer=mixing_weight.detach()
            new_mixing_weight=mixing_weight
        else:
            self.mixing_weight_buffer=self.mixing_momentum*self.mixing_weight_buffer+\
            (1-self.mixing_momentum)*mixing_weight.detach()
            new_mixing_weight=self.mixing_momentum*self.mixing_weight_buffer+\
            (1-self.mixing_momentum)*mixing_weight
        #print('.......',new_mixing_weight)
        return new_mixing_weight
    def update_innner_product_momentum_buffer(self,inner_product_group):
        for ind,grad,grad_buffer in zip(range(self.size),inner_product_group,self.inner_product_buffer): 
            if grad_buffer==None:
                grad_buffer=grad
            else:
                grad_buffer=self.mixing_momentum*grad_buffer+(1-self.mixing_momentum)*grad
            self.inner_product_buffer[ind]=grad_buffer.detach()
            self.inner_product_buffer_forward[ind]=grad_buffer
        print('.......',self.inner_product_buffer_forward)

        return inner_product_group
    def update_momentum_buffer(self,grad_group,key_tensor):
        for ind,grad,grad_buffer in zip(range(self.size),grad_group,self.grad_group_buffer): 
            if grad_buffer==None:
                grad_buffer=grad
            else:
                grad_buffer=self.momentum*grad_buffer+(1-self.momentum)*grad
            self.grad_group_buffer[ind]=grad_buffer

        if self.key_tensor_buffer==None:
            self.key_tensor_buffer=key_tensor
        else:
            self.key_tensor_buffer=self.momentum*self.key_tensor_buffer+(1-self.momentum)*key_tensor

        return self.grad_group_buffer,self.key_tensor_buffer
    def preprocess(self,grad_group,val_grad):
        grad_group,val_grad=self.preprocess_moemntum(grad_group,val_grad)
        new_grad_group=[]
        for i in range(self.num_weights):
            tmp_grad_group=[]
            for j in range(self.size):
                model_param=grad_group[j][i].detach()
                model_param=model_param.view(model_param.size(0),-1)
                tmp_grad_group.append(model_param)
            tmp_grad_group=torch.stack(tmp_grad_group)
            new_grad_group.append(tmp_grad_group)
        key_tensor_group=[]
        for i in range(self.num_weights):
            model_param=val_grad[i].detach()
            model_param=model_param.view(model_param.size(0),-1)
            key_tensor_group.append(model_param)
        



        '''
        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        '''
        #new_grad_group,key_tensor=self.update_momentum_buffer(new_grad_group,key_tensor)
        #print(new_grad_group)
        
        '''
        print('mean=',[torch.mean(i) for i in new_grad_group])
        print('abs=',[torch.mean(torch.abs(i)) for i in new_grad_group])
        '''
        return new_grad_group,key_tensor_group
    def preprocess1(self,grad_group,val_grad):
        new_grad_group=[]
        for i in range(self.num_weights):
            tmp_grad_group=[]
            for j in range(self.size):
                model_param=grad_group[j][i].detach()
                model_param=model_param.view(model_param.size(0),-1)
                tmp_grad_group.append(model_param/torch.norm(model_param))
            tmp_grad_group=torch.stack(tmp_grad_group)
            new_grad_group.append(tmp_grad_group)
        key_tensor_group=[]
        for i in range(self.num_weights):
            model_param=val_grad[i].detach()
            model_param=model_param.view(model_param.size(0),-1)
            key_tensor_group.append(model_param)
        



        '''
        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        '''
        #new_grad_group,key_tensor=self.update_momentum_buffer(new_grad_group,key_tensor)
        #print(new_grad_group)
        
        '''
        print('mean=',[torch.mean(i) for i in new_grad_group])
        print('abs=',[torch.mean(torch.abs(i)) for i in new_grad_group])
        '''
        return new_grad_group,key_tensor_group
    def preprocess2(self,grad_group,val_grad):
        new_grad_group=[]
        for i in range(self.num_weights):
            tmp_grad_group=[]
            for j in range(self.size):
                model_param=grad_group[j][i].detach()
                model_param=model_param.view(model_param.size(0),-1)
                tmp_grad_group.append(model_param/torch.norm(model_param))
            tmp_grad_group=torch.stack(tmp_grad_group)
            new_grad_group.append(tmp_grad_group)
        key_tensor_group=[]
        for i in range(self.num_weights):
            model_param=val_grad[i].detach()
            model_param=model_param.view(model_param.size(0),-1)
            key_tensor_group.append(model_param)
        



        '''
        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        '''
        #new_grad_group,key_tensor=self.update_momentum_buffer(new_grad_group,key_tensor)
        #print(new_grad_group)
        
        '''
        print('mean=',[torch.mean(i) for i in new_grad_group])
        print('abs=',[torch.mean(torch.abs(i)) for i in new_grad_group])
        '''
        return new_grad_group,key_tensor_group
    def preprocess_moemntum(self,grad_group,val_grad):
        new_grad_group=[comm_helpers.flatten_tensors(i).detach() for i in grad_group]
        key_tensor=comm_helpers.flatten_tensors(val_grad).detach()
        new_grad_group,key_tensor=self.update_momentum_buffer(new_grad_group,key_tensor)
        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)

        new_grad_group=[comm_helpers.unflatten_tensors(i,val_grad) for i in new_grad_group]
        key_tensor=comm_helpers.unflatten_tensors(key_tensor,val_grad)
        return new_grad_group,key_tensor


class Attention_layerwise_gru(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, model, out_dim,rank, comm,size):
        super().__init__()


        #self.in_dim=in_dim
        self.out_dim=out_dim
        self.in_dim_group=[]
        self.w_qs_group=[]
        self.w_ks_group=[]
        self.num_weights=0
        for i in model.params():
            param_data=i.detach().data
            param_data=param_data.view(param_data.size(0),-1)
            param_size=param_data.size()
            in_dim=param_size[1]
            self.in_dim_group.append(in_dim)
            self.w_qs_group.append(nn.Linear(in_dim, out_dim, bias=False))
            self.w_ks_group.append(nn.Linear(in_dim, out_dim, bias=False))
            self.num_weights=self.num_weights+1
            self.w_qs_group[-1].weight.data=self.w_qs_group[-1].weight.data
            self.w_ks_group[-1].weight.data=self.w_ks_group[-1].weight.data*10
        self.w_qs_layers = nn.ModuleList(self.w_qs_group)
        self.w_ks_layers = nn.ModuleList(self.w_ks_group)
            


        self.rank=rank
        self.comm=comm
        self.size=size
        self.grad_group_buffer=[None for i in range(size)]
        self.key_tensor_buffer=None
        self.grad_group_buffer_layerwise=[[None for i in range(size)] for j in range(10)]
        self.key_tensor_buffer_layerwise=[None for i in range(10)]
        self.inner_product_buffer=[None for i in range(size)]
        self.inner_product_buffer_forward=[None for i in range(size)]
        self.mixing_weight_buffer=None
        self.momentum=0.99
        self.mixing_momentum=0.99
        #self.register_parameter(name='mixing_momentum', param=torch.nn.Parameter(0.5*torch.ones(1)))
        self.iteration=0


        self.history_trajectory=[]
        
        #self.rnn=nn.GRU(input_size=1,hidden_size=1,num_layers=1,batch_first=False)
        self.rnn=nn.GRU(input_size=1,hidden_size=1,num_layers=1,batch_first=False,bias=True)
    def forward(self, grad_group,val_grad,model):
        q_group,k_group=self.preprocess1(grad_group,val_grad)
        #q_group size=[num_weights,neighbour_num,num_filter,in_dim] list of tensors
        #k_group size=[num_weights,num_filter,in_dim] list of tensors

        '''
        q=q*50
        k=k*50
        '''
        
        prod_sum=torch.zeros(self.size).cuda()
        for i in range(self.num_weights):
            q=q_group[i]
            k=k_group[i]
            neighbour_num,num_filter,in_dim=q.size()
            q = self.w_qs_layers[i](q)
            k = self.w_qs_layers[i](k)

            q=q.view(neighbour_num,-1)
            k=k.view(-1,1)



            prod=torch.matmul(q,k).view(-1)/num_filter
            prod_sum=prod_sum+prod
            #mixing_weight=mixing_weight/(self.out_dim**0.5)
        prod_sum=prod_sum/self.num_weights
        prod_sum=prod_sum.view(-1)
        #print('rank=',self.rank,prod_sum)
        #print(mixing_weight)
        print('prod_sum size=',prod_sum.size())
        prod_sum=self.gru_forward(prod_sum)
        mixing_weight=F.softmax(prod_sum,dim=0)

        mixing_weight=mixing_weight.view(-1)


        #mixing_weight=self.gru_forward(mixing_weight)



        return mixing_weight
    def update_mixing_momentum_buffer(self,mixing_weight):

        if self.mixing_weight_buffer==None:
            self.mixing_weight_buffer=mixing_weight.detach()
            new_mixing_weight=mixing_weight
        else:
            self.mixing_weight_buffer=self.mixing_momentum*self.mixing_weight_buffer+\
            (1-self.mixing_momentum)*mixing_weight.detach()
            new_mixing_weight=self.mixing_momentum*self.mixing_weight_buffer+\
            (1-self.mixing_momentum)*mixing_weight
        #print('.......',new_mixing_weight)
        return new_mixing_weight
    def gru_forward(self,mixing_weight):
        
        if len(self.history_trajectory)==0:
            self.history_trajectory.append(mixing_weight.detach().data)
            
            return mixing_weight
        else:
            
            history_trajectory=copy.deepcopy(self.history_trajectory)
            self.history_trajectory.append(mixing_weight.detach().data)
            history_trajectory.append(mixing_weight)
            history_trajectory=torch.stack(history_trajectory).cuda()
            history_trajectory=history_trajectory.view(-1,self.size,1)
            '''
            if self.rank==0:
                print(self.history_trajectory)
            '''
            h0=history_trajectory[0].view(1,10,1)
            output,hn=self.rnn(history_trajectory,h0)
            return hn.view(-1)

    def update_innner_product_momentum_buffer(self,inner_product_group):
        for ind,grad,grad_buffer in zip(range(self.size),inner_product_group,self.inner_product_buffer): 
            if grad_buffer==None:
                grad_buffer=grad
            else:
                grad_buffer=self.mixing_momentum*grad_buffer+(1-self.mixing_momentum)*grad
            self.inner_product_buffer[ind]=grad_buffer.detach()
            self.inner_product_buffer_forward[ind]=grad_buffer
        print('.......',self.inner_product_buffer_forward)

        return inner_product_group
    def update_momentum_buffer(self,grad_group,key_tensor):
        for ind,grad,grad_buffer in zip(range(self.size),grad_group,self.grad_group_buffer): 
            if grad_buffer==None:
                grad_buffer=grad
            else:
                grad_buffer=self.momentum*grad_buffer+(1-self.momentum)*grad
            self.grad_group_buffer[ind]=grad_buffer

        if self.key_tensor_buffer==None:
            self.key_tensor_buffer=key_tensor
        else:
            self.key_tensor_buffer=self.momentum*self.key_tensor_buffer+(1-self.momentum)*key_tensor

        return self.grad_group_buffer,self.key_tensor_buffer
    def preprocess(self,grad_group,val_grad):
        new_grad_group=[]
        for i in range(self.num_weights):
            tmp_grad_group=[]
            for j in range(self.size):
                model_param=grad_group[j][i].detach()
                model_param=model_param.view(model_param.size(0),-1)
                tmp_grad_group.append(model_param)
            tmp_grad_group=torch.stack(tmp_grad_group)
            new_grad_group.append(tmp_grad_group)
        key_tensor_group=[]
        for i in range(self.num_weights):
            model_param=val_grad[i].detach()
            model_param=model_param.view(model_param.size(0),-1)
            key_tensor_group.append(model_param)
        



        '''
        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        '''
        #new_grad_group,key_tensor=self.update_momentum_buffer(new_grad_group,key_tensor)
        #print(new_grad_group)
        
        '''
        print('mean=',[torch.mean(i) for i in new_grad_group])
        print('abs=',[torch.mean(torch.abs(i)) for i in new_grad_group])
        '''
        return new_grad_group,key_tensor_group
    def preprocess1(self,grad_group,val_grad):
        new_grad_group=[]
        for i in range(self.num_weights):
            tmp_grad_group=[]
            for j in range(self.size):
                model_param=grad_group[j][i].detach()
                model_param=model_param.view(model_param.size(0),-1)
                tmp_grad_group.append(model_param/torch.norm(model_param))
            tmp_grad_group=torch.stack(tmp_grad_group)
            new_grad_group.append(tmp_grad_group)
        key_tensor_group=[]
        for i in range(self.num_weights):
            model_param=val_grad[i].detach()
            model_param=model_param.view(model_param.size(0),-1)
            key_tensor_group.append(model_param)
        



        '''
        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        '''
        #new_grad_group,key_tensor=self.update_momentum_buffer(new_grad_group,key_tensor)
        #print(new_grad_group)
        
        '''
        print('mean=',[torch.mean(i) for i in new_grad_group])
        print('abs=',[torch.mean(torch.abs(i)) for i in new_grad_group])
        '''
        return new_grad_group,key_tensor_group
    def preprocess2(self,grad_group,val_grad):
        new_grad_group=[]
        for i in range(self.num_weights):
            tmp_grad_group=[]
            for j in range(self.size):
                model_param=grad_group[j][i].detach()
                model_param=model_param.view(model_param.size(0),-1)
                tmp_grad_group.append(model_param/torch.norm(model_param))
            tmp_grad_group=torch.stack(tmp_grad_group)
            new_grad_group.append(tmp_grad_group)
        key_tensor_group=[]
        for i in range(self.num_weights):
            model_param=val_grad[i].detach()
            model_param=model_param.view(model_param.size(0),-1)
            key_tensor_group.append(model_param)
        



        '''
        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        '''
        #new_grad_group,key_tensor=self.update_momentum_buffer(new_grad_group,key_tensor)
        #print(new_grad_group)
        
        '''
        print('mean=',[torch.mean(i) for i in new_grad_group])
        print('abs=',[torch.mean(torch.abs(i)) for i in new_grad_group])
        '''
        return new_grad_group,key_tensor_group









class Attention_layerwise_multilayer(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, model, out_dim,rank, comm,size):
        super().__init__()


        #self.in_dim=in_dim
        self.out_dim=out_dim
        self.in_dim_group=[]
        self.w_qs_group=[]
        self.w_ks_group=[]
        self.num_weights=0
        for i in model.params():
            param_data=i.detach().data
            param_data=param_data.view(param_data.size(0),-1)
            param_size=param_data.size()
            in_dim=param_size[1]
            self.in_dim_group.append(in_dim)
            self.w_qs_group.append(nn.Linear(in_dim, out_dim, bias=False))
            self.w_ks_group.append(nn.Linear(in_dim, out_dim, bias=False))
            self.num_weights=self.num_weights+1
            '''
            self.w_qs_group[-1].weight.data=self.w_qs_group[-1].weight.data
            self.w_ks_group[-1].weight.data=self.w_ks_group[-1].weight.data*10
            '''
        self.w_qs_layers = nn.ModuleList(self.w_qs_group)
        self.w_ks_layers = nn.ModuleList(self.w_ks_group)
            


        self.rank=rank
        self.comm=comm
        self.size=size
        self.grad_group_buffer=[None for i in range(size)]
        self.key_tensor_buffer=None
        self.grad_group_buffer_layerwise=[[None for i in range(size)] for j in range(10)]
        self.key_tensor_buffer_layerwise=[None for i in range(10)]
        self.inner_product_buffer=[None for i in range(size)]
        self.inner_product_buffer_forward=[None for i in range(size)]
        self.mixing_weight_buffer=None
        self.momentum=0.99
        self.mixing_momentum=0.99
        #self.register_parameter(name='mixing_momentum', param=torch.nn.Parameter(0.5*torch.ones(1)))
        self.iteration=0
        '''
        self.register_parameter(name='scale_', param=torch.nn.Parameter(torch.ones(1)))
        self.register_parameter(name='bias_', param=torch.nn.Parameter(torch.zeros(1)))
        '''
        self.linear1=nn.Linear(1,100)
        self.linear2=nn.Linear(100,1)


    def forward(self, grad_group,val_grad,model):
        q_group,k_group=self.preprocess(grad_group,val_grad)
        #q_group size=[num_weights,neighbour_num,num_filter,in_dim] list of tensors
        #k_group size=[num_weights,num_filter,in_dim] list of tensors

        '''
        q=q*50
        k=k*50
        '''
        
        prod_sum=torch.zeros(self.size).cuda()
        for i in range(self.num_weights):
            q=q_group[i]
            k=k_group[i]
            neighbour_num,num_filter,in_dim=q.size()
            q = self.w_qs_layers[i](q)
            k = self.w_qs_layers[i](k)

            q=q.view(neighbour_num,-1)
            k=k.view(-1,1)



            prod=torch.matmul(q,k).view(-1)/num_filter
            prod_sum=prod_sum+prod
            #mixing_weight=mixing_weight/(self.out_dim**0.5)
        prod_sum=prod_sum/self.num_weights
        prod_sum=prod_sum.view(-1)
        #print('rank=',self.rank,prod_sum)
        #print(mixing_weight)
        mixing_weight=F.softmax(prod_sum,dim=0)

        mixing_weight=mixing_weight.view(-1)



        return mixing_weight
    def update_mixing_momentum_buffer(self,mixing_weight):
        if self.mixing_weight_buffer==None:
            self.mixing_weight_buffer=mixing_weight.detach()
            new_mixing_weight=mixing_weight
        else:
            self.mixing_weight_buffer=self.mixing_momentum*self.mixing_weight_buffer+\
            (1-self.mixing_momentum)*mixing_weight.detach()
            new_mixing_weight=self.mixing_momentum*self.mixing_weight_buffer+\
            (1-self.mixing_momentum)*mixing_weight
        #print('.......',new_mixing_weight)
        return new_mixing_weight
    def update_innner_product_momentum_buffer(self,inner_product_group):
        for ind,grad,grad_buffer in zip(range(self.size),inner_product_group,self.inner_product_buffer): 
            if grad_buffer==None:
                grad_buffer=grad
            else:
                grad_buffer=self.mixing_momentum*grad_buffer+(1-self.mixing_momentum)*grad
            self.inner_product_buffer[ind]=grad_buffer.detach()
            self.inner_product_buffer_forward[ind]=grad_buffer
        print('.......',self.inner_product_buffer_forward)

        return inner_product_group
    def update_momentum_buffer(self,grad_group,key_tensor):
        for ind,grad,grad_buffer in zip(range(self.size),grad_group,self.grad_group_buffer): 
            if grad_buffer==None:
                grad_buffer=grad
            else:
                grad_buffer=self.momentum*grad_buffer+(1-self.momentum)*grad
            self.grad_group_buffer[ind]=grad_buffer

        if self.key_tensor_buffer==None:
            self.key_tensor_buffer=key_tensor
        else:
            self.key_tensor_buffer=self.momentum*self.key_tensor_buffer+(1-self.momentum)*key_tensor

        return self.grad_group_buffer,self.key_tensor_buffer
    def preprocess(self,grad_group,val_grad):
        grad_group,val_grad=self.preprocess_learn_layer(grad_group,val_grad)
        new_grad_group=[]
        for i in range(self.num_weights):
            tmp_grad_group=[]
            for j in range(self.size):
                model_param=grad_group[j][i]
                model_param=model_param.view(model_param.size(0),-1)
                tmp_grad_group.append(model_param)
            tmp_grad_group=torch.stack(tmp_grad_group)
            new_grad_group.append(tmp_grad_group)
        key_tensor_group=[]
        for i in range(self.num_weights):
            model_param=val_grad[i]
            model_param=model_param.view(model_param.size(0),-1)
            key_tensor_group.append(model_param)
        



        '''
        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        '''
        #new_grad_group,key_tensor=self.update_momentum_buffer(new_grad_group,key_tensor)
        #print(new_grad_group)
        
        '''
        print('mean=',[torch.mean(i) for i in new_grad_group])
        print('abs=',[torch.mean(torch.abs(i)) for i in new_grad_group])
        '''
        return new_grad_group,key_tensor_group
    def preprocess1(self,grad_group,val_grad):
        new_grad_group=[]
        for i in range(self.num_weights):
            tmp_grad_group=[]
            for j in range(self.size):
                model_param=grad_group[j][i].detach()
                model_param=model_param.view(model_param.size(0),-1)
                tmp_grad_group.append(model_param/torch.norm(model_param))
            tmp_grad_group=torch.stack(tmp_grad_group)
            new_grad_group.append(tmp_grad_group)
        key_tensor_group=[]
        for i in range(self.num_weights):
            model_param=val_grad[i].detach()
            model_param=model_param.view(model_param.size(0),-1)
            key_tensor_group.append(model_param)
        



        '''
        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        '''
        #new_grad_group,key_tensor=self.update_momentum_buffer(new_grad_group,key_tensor)
        #print(new_grad_group)
        
        '''
        print('mean=',[torch.mean(i) for i in new_grad_group])
        print('abs=',[torch.mean(torch.abs(i)) for i in new_grad_group])
        '''
        return new_grad_group,key_tensor_group
    def preprocess2(self,grad_group,val_grad):
        new_grad_group=[]
        for i in range(self.num_weights):
            tmp_grad_group=[]
            for j in range(self.size):
                model_param=grad_group[j][i].detach()
                model_param=model_param.view(model_param.size(0),-1)
                tmp_grad_group.append(model_param/torch.norm(model_param))
            tmp_grad_group=torch.stack(tmp_grad_group)
            new_grad_group.append(tmp_grad_group)
        key_tensor_group=[]
        for i in range(self.num_weights):
            model_param=val_grad[i].detach()
            model_param=model_param.view(model_param.size(0),-1)
            key_tensor_group.append(model_param)
        



        '''
        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        '''
        #new_grad_group,key_tensor=self.update_momentum_buffer(new_grad_group,key_tensor)
        #print(new_grad_group)
        
        '''
        print('mean=',[torch.mean(i) for i in new_grad_group])
        print('abs=',[torch.mean(torch.abs(i)) for i in new_grad_group])
        '''
        return new_grad_group,key_tensor_group
    def preprocess_moemntum(self,grad_group,val_grad):
        new_grad_group=[comm_helpers.flatten_tensors(i).detach() for i in grad_group]
        key_tensor=comm_helpers.flatten_tensors(val_grad).detach()
        new_grad_group,key_tensor=self.update_momentum_buffer(new_grad_group,key_tensor)
        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)

        new_grad_group=[comm_helpers.unflatten_tensors(i,val_grad) for i in new_grad_group]
        key_tensor=comm_helpers.unflatten_tensors(key_tensor,val_grad)
        return new_grad_group,key_tensor
    def preprocess_abs(self,grad_group,val_grad):
        new_grad_group=[comm_helpers.flatten_tensors(i).detach() for i in grad_group]
        key_tensor=comm_helpers.flatten_tensors(val_grad).detach()

        new_grad_group=[torch.abs(i) for i in new_grad_group]
        key_tensor=torch.abs(key_tensor)

        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)

        new_grad_group=[comm_helpers.unflatten_tensors(i,val_grad) for i in new_grad_group]
        key_tensor=comm_helpers.unflatten_tensors(key_tensor,val_grad)
        return new_grad_group,key_tensor
    def preprocess_relu(self,grad_group,val_grad):
        new_grad_group=[comm_helpers.flatten_tensors(i).detach() for i in grad_group]
        key_tensor=comm_helpers.flatten_tensors(val_grad).detach()

        new_grad_group=[torch.nn.functional.relu(i) for i in new_grad_group]
        key_tensor=torch.nn.functional.relu(key_tensor)

        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)

        new_grad_group=[comm_helpers.unflatten_tensors(i,val_grad) for i in new_grad_group]
        key_tensor=comm_helpers.unflatten_tensors(key_tensor,val_grad)
        return new_grad_group,key_tensor
    def preprocess_learn(self,grad_group,val_grad):
        new_grad_group=[comm_helpers.flatten_tensors(i).detach() for i in grad_group]
        key_tensor=comm_helpers.flatten_tensors(val_grad).detach()

        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)




        new_grad_group=[torch.nn.functional.relu(i*self.scale_+self.bias_) for i in new_grad_group]
        key_tensor=torch.nn.functional.relu(key_tensor*self.scale_+self.bias_)

        

        new_grad_group=[comm_helpers.unflatten_tensors(i,val_grad) for i in new_grad_group]
        key_tensor=comm_helpers.unflatten_tensors(key_tensor,val_grad)
        print('scale=',self.scale_)
        print('bias=',self.bias_)
        return new_grad_group,key_tensor
    def preprocess_learn_layer(self,grad_group,val_grad):
        new_grad_group=[comm_helpers.flatten_tensors(i).detach() for i in grad_group]
        key_tensor=comm_helpers.flatten_tensors(val_grad).detach()

        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)

        new_grad_group=torch.stack(new_grad_group)
        new_grad_group=new_grad_group.view(-1,1)
        key_tensor=key_tensor.view(-1,1)


        new_grad_group=F.relu(self.linear1(new_grad_group))
        key_tensor=F.relu(self.linear1(key_tensor))

        new_grad_group=F.relu(self.linear2(new_grad_group))
        key_tensor=F.relu(self.linear2(key_tensor))

        new_grad_group=new_grad_group.view(self.size,-1)

        

        new_grad_group=[comm_helpers.unflatten_tensors(i,val_grad) for i in new_grad_group]
        key_tensor=comm_helpers.unflatten_tensors(key_tensor,val_grad)
        #print('scale=',self.linear1.weight)
        #print('bias=',self.linear1.bias)
        return new_grad_group,key_tensor
class Attention_layerwise_multilayer1(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, model, out_dim,rank, comm,size):
        super().__init__()


        #self.in_dim=in_dim
        self.out_dim=out_dim
        self.in_dim_group=[]
        self.w_qs_group=[]
        self.w_ks_group=[]
        self.num_weights=0
        for i in model.params():
            param_data=i.detach().data
            param_data=param_data.view(param_data.size(0),-1)
            param_size=param_data.size()
            in_dim=param_size[1]
            self.in_dim_group.append(in_dim)
            self.w_qs_group.append(nn.Linear(in_dim, out_dim, bias=False))
            self.w_ks_group.append(nn.Linear(in_dim, out_dim, bias=False))
            self.num_weights=self.num_weights+1
            '''
            self.w_qs_group[-1].weight.data=self.w_qs_group[-1].weight.data
            self.w_ks_group[-1].weight.data=self.w_ks_group[-1].weight.data*10
            '''
        self.w_qs_layers = nn.ModuleList(self.w_qs_group)
        self.w_ks_layers = nn.ModuleList(self.w_ks_group)
            


        self.rank=rank
        self.comm=comm
        self.size=size
        self.grad_group_buffer=[None for i in range(size)]
        self.key_tensor_buffer=None
        self.grad_group_buffer_layerwise=[[None for i in range(size)] for j in range(10)]
        self.key_tensor_buffer_layerwise=[None for i in range(10)]
        self.inner_product_buffer=[None for i in range(size)]
        self.inner_product_buffer_forward=[None for i in range(size)]
        self.mixing_weight_buffer=None
        self.momentum=0.99
        self.mixing_momentum=0.99
        #self.register_parameter(name='mixing_momentum', param=torch.nn.Parameter(0.5*torch.ones(1)))
        self.iteration=0
        '''
        self.register_parameter(name='scale_', param=torch.nn.Parameter(torch.ones(1)))
        self.register_parameter(name='bias_', param=torch.nn.Parameter(torch.zeros(1)))
        '''
        self.linear=nn.Linear(10,100)


    def forward1(self, grad_group,val_grad,model):
        q_group,k_group=self.preprocess(grad_group,val_grad)
        #q_group size=[num_weights,neighbour_num,num_filter,in_dim] list of tensors
        #k_group size=[num_weights,num_filter,in_dim] list of tensors

        '''
        q=q*50
        k=k*50
        '''
        
        prod_sum=torch.zeros(self.size).cuda()
        for i in range(self.num_weights):
            q=q_group[i]
            k=k_group[i]
            neighbour_num,num_filter,in_dim=q.size()
            q = self.w_qs_layers[i](q)
            k = self.w_qs_layers[i](k)

            q=q.view(neighbour_num,-1)
            k=k.view(-1,1)



            prod=torch.matmul(q,k).view(-1)/num_filter
            prod_sum=prod_sum+prod
            #mixing_weight=mixing_weight/(self.out_dim**0.5)
        prod_sum=prod_sum/self.num_weights
        prod_sum=prod_sum.view(-1)
        #print('rank=',self.rank,prod_sum)
        #print(mixing_weight)
        mixing_weight=F.softmax(prod_sum,dim=0)

        mixing_weight=mixing_weight.view(-1)



        return mixing_weight
    def forward(self,grad_group,val_grad,model):
        q_group,k_group=self.get_gradient_norm2(grad_group,val_grad)
        q_group=q_group.view(self.size,-1)
        k_group=k_group.view(1,-1)

        q_group=torch.nn.functional.relu(self.linear(q_group))
        k_group=torch.nn.functional.relu(self.linear(k_group))

        k_group=k_group.view(-1,1)
        prod=torch.matmul(q_group,k_group).view(-1)
        mixing_weight=F.softmax(prod,dim=0).view(-1)
        #print('weight=',self.linear.weight)
        #print('bias=',self.linear.bias)
        return mixing_weight
    def preprocess(self,grad_group,val_grad):
        grad_group,val_grad=self.preprocess_learn_layer(grad_group,val_grad)
        new_grad_group=[]
        for i in range(self.num_weights):
            tmp_grad_group=[]
            for j in range(self.size):
                model_param=grad_group[j][i]
                model_param=model_param.view(model_param.size(0),-1)
                tmp_grad_group.append(model_param)
            tmp_grad_group=torch.stack(tmp_grad_group)
            new_grad_group.append(tmp_grad_group)
        key_tensor_group=[]
        for i in range(self.num_weights):
            model_param=val_grad[i]
            model_param=model_param.view(model_param.size(0),-1)
            key_tensor_group.append(model_param)
        



        '''
        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        '''
        #new_grad_group,key_tensor=self.update_momentum_buffer(new_grad_group,key_tensor)
        #print(new_grad_group)
        
        '''
        print('mean=',[torch.mean(i) for i in new_grad_group])
        print('abs=',[torch.mean(torch.abs(i)) for i in new_grad_group])
        '''
        return new_grad_group,key_tensor_group
    def get_gradient_norm2(self,grad_group,val_grad):
        new_grad_group=[i[-1] for i in grad_group]
        key_tensor=val_grad[-1]

        
     

        new_grad_group=[comm_helpers.flatten_tensors([i]).detach() for i in new_grad_group]
        key_tensor=comm_helpers.flatten_tensors([key_tensor]).detach()


        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        new_grad_group=torch.stack(new_grad_group)
        key_tensor=key_tensor/torch.norm(key_tensor)
        
        return new_grad_group,key_tensor
    
class Attention_elementwise_multilayer(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, model, out_dim,rank, comm,size):
        super().__init__()


        #self.in_dim=in_dim
        self.out_dim=out_dim
        self.in_dim_group=[]
        self.w_qs_group=[]
        self.w_ks_group=[]
        self.num_weights=0
        for i in model.params():
            param_data=i.detach().data
            param_data=param_data.view(param_data.size(0),-1)
            param_size=param_data.size()
            in_dim=param_size[1]
            self.in_dim_group.append(in_dim)
            self.w_qs_group.append(nn.Linear(in_dim, out_dim, bias=False))
            self.w_ks_group.append(nn.Linear(in_dim, out_dim, bias=False))
            self.num_weights=self.num_weights+1
            '''
            self.w_qs_group[-1].weight.data=self.w_qs_group[-1].weight.data
            self.w_ks_group[-1].weight.data=self.w_ks_group[-1].weight.data*10
            '''
        self.w_qs_layers = nn.ModuleList(self.w_qs_group)
        self.w_ks_layers = nn.ModuleList(self.w_ks_group)
            


        self.rank=rank
        self.comm=comm
        self.size=size
        self.grad_group_buffer=[None for i in range(size)]
        self.key_tensor_buffer=None
        self.grad_group_buffer_layerwise=[[None for i in range(size)] for j in range(10)]
        self.key_tensor_buffer_layerwise=[None for i in range(10)]
        self.inner_product_buffer=[None for i in range(size)]
        self.inner_product_buffer_forward=[None for i in range(size)]
        self.mixing_weight_buffer=None
        self.momentum=0.99
        self.mixing_momentum=0.99
        #self.register_parameter(name='mixing_momentum', param=torch.nn.Parameter(0.5*torch.ones(1)))
        self.iteration=0
        '''
        self.register_parameter(name='scale_', param=torch.nn.Parameter(torch.ones(1)))
        self.register_parameter(name='bias_', param=torch.nn.Parameter(torch.zeros(1)))
        '''
        self.linear1=nn.Linear(1,2,bias=False)
        self.linear2=nn.Linear(2,1,bias=False)


    def forward1(self, grad_group,val_grad,model):
        q_group,k_group=self.preprocess(grad_group,val_grad)
        #q_group size=[num_weights,neighbour_num,num_filter,in_dim] list of tensors
        #k_group size=[num_weights,num_filter,in_dim] list of tensors

        '''
        q=q*50
        k=k*50
        '''
        
        prod_sum=torch.zeros(self.size).cuda()
        for i in range(self.num_weights):
            q=q_group[i]
            k=k_group[i]
            neighbour_num,num_filter,in_dim=q.size()
            q = self.w_qs_layers[i](q)
            k = self.w_qs_layers[i](k)

            q=q.view(neighbour_num,-1)
            k=k.view(-1,1)



            prod=torch.matmul(q,k).view(-1)/num_filter
            prod_sum=prod_sum+prod
            #mixing_weight=mixing_weight/(self.out_dim**0.5)
        prod_sum=prod_sum/self.num_weights
        prod_sum=prod_sum.view(-1)
        #print('rank=',self.rank,prod_sum)
        #print(mixing_weight)
        mixing_weight=F.softmax(prod_sum,dim=0)

        mixing_weight=mixing_weight.view(-1)



        return mixing_weight
    def forward(self,grad_group,val_grad,model):
        q_group,k_group=self.get_gradient_norm2(grad_group,val_grad)
        q_group=q_group.view(-1,1)
        k_group=k_group.view(-1,1)

        q_group=torch.nn.functional.relu(self.linear1(q_group))
        k_group=torch.nn.functional.relu(self.linear1(k_group))

        q_group=torch.nn.functional.relu(self.linear2(q_group))
        k_group=torch.nn.functional.relu(self.linear2(k_group))

        q_group=q_group.view(self.size,-1)
        k_group=k_group.view(-1,1)
        prod=torch.matmul(q_group,k_group).view(-1)
        mixing_weight=F.softmax(prod,dim=0).view(-1)
        if self.rank==0:
            print('weight1=',self.linear1.weight)
            print('weight2=',self.linear2.weight)
        return mixing_weight
    def preprocess(self,grad_group,val_grad):
        grad_group,val_grad=self.preprocess_learn_layer(grad_group,val_grad)
        new_grad_group=[]
        for i in range(self.num_weights):
            tmp_grad_group=[]
            for j in range(self.size):
                model_param=grad_group[j][i]
                model_param=model_param.view(model_param.size(0),-1)
                tmp_grad_group.append(model_param)
            tmp_grad_group=torch.stack(tmp_grad_group)
            new_grad_group.append(tmp_grad_group)
        key_tensor_group=[]
        for i in range(self.num_weights):
            model_param=val_grad[i]
            model_param=model_param.view(model_param.size(0),-1)
            key_tensor_group.append(model_param)
        



        '''
        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        '''
        #new_grad_group,key_tensor=self.update_momentum_buffer(new_grad_group,key_tensor)
        #print(new_grad_group)
        
        '''
        print('mean=',[torch.mean(i) for i in new_grad_group])
        print('abs=',[torch.mean(torch.abs(i)) for i in new_grad_group])
        '''
        return new_grad_group,key_tensor_group
    def get_gradient_norm2(self,grad_group,val_grad):
        key_tensor=grad_group[self.rank]
        
        

        new_grad_group=[comm_helpers.flatten_tensors(i).detach() for i in grad_group]
        key_tensor=comm_helpers.flatten_tensors(key_tensor).detach()


        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        new_grad_group=torch.stack(new_grad_group)
        key_tensor=key_tensor/torch.norm(key_tensor)
        
        return new_grad_group,key_tensor
class Attention_layerwise_model_update(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, model, out_dim,rank, comm,size):
        super().__init__()


        #self.in_dim=in_dim
        self.out_dim=out_dim
        self.in_dim_group=[]
        self.w_qs_group=[]
        self.w_ks_group=[]
        self.num_weights=0
        for i in model.params():
            param_data=i.detach().data
            param_data=param_data.view(param_data.size(0),-1)
            param_size=param_data.size()
            in_dim=param_size[1]
            self.in_dim_group.append(in_dim)
            self.w_qs_group.append(nn.Linear(in_dim, out_dim, bias=False))
            self.w_ks_group.append(nn.Linear(in_dim, out_dim, bias=False))
            self.num_weights=self.num_weights+1
            '''
            self.w_qs_group[-1].weight.data=self.w_qs_group[-1].weight.data
            self.w_ks_group[-1].weight.data=self.w_ks_group[-1].weight.data*10
            '''
        self.w_qs_layers = nn.ModuleList(self.w_qs_group)
        self.w_ks_layers = nn.ModuleList(self.w_ks_group)
            


        self.rank=rank
        self.comm=comm
        self.size=size
        self.grad_group_buffer=[None for i in range(size)]
        self.key_tensor_buffer=None
        self.grad_group_buffer_layerwise=[[None for i in range(size)] for j in range(10)]
        self.key_tensor_buffer_layerwise=[None for i in range(10)]
        self.inner_product_buffer=[None for i in range(size)]
        self.inner_product_buffer_forward=[None for i in range(size)]
        self.mixing_weight_buffer=None
        self.momentum=0.99
        self.mixing_momentum=0.99
        #self.register_parameter(name='mixing_momentum', param=torch.nn.Parameter(0.5*torch.ones(1)))
        self.iteration=0
        


    def forward(self, model_update_group):
        q_group,k_group=self.preprocess(model_update_group)
        #q_group size=[num_weights,neighbour_num,num_filter,in_dim] list of tensors
        #k_group size=[num_weights,num_filter,in_dim] list of tensors

        '''
        q=q*50
        k=k*50
        '''
        
        prod_sum=torch.zeros(self.size).cuda()
        for i in range(self.num_weights):
            q=q_group[i]
            k=k_group[i]
            neighbour_num,num_filter,in_dim=q.size()
            q = self.w_qs_layers[i](q)
            k = self.w_qs_layers[i](k)

            q=q.view(neighbour_num,-1)
            k=k.view(-1,1)



            prod=torch.matmul(q,k).view(-1)/num_filter
            prod_sum=prod_sum+prod
            #mixing_weight=mixing_weight/(self.out_dim**0.5)
        prod_sum=prod_sum.view(-1)
        #print('rank=',self.rank,prod_sum)
        #print(mixing_weight)
        mixing_weight=F.softmax(prod_sum,dim=0)

        mixing_weight=mixing_weight.view(-1)



        return mixing_weight
    def update_mixing_momentum_buffer(self,mixing_weight):
        if self.mixing_weight_buffer==None:
            self.mixing_weight_buffer=mixing_weight.detach()
            new_mixing_weight=mixing_weight
        else:
            self.mixing_weight_buffer=self.mixing_momentum*self.mixing_weight_buffer+\
            (1-self.mixing_momentum)*mixing_weight.detach()
            new_mixing_weight=self.mixing_momentum*self.mixing_weight_buffer+\
            (1-self.mixing_momentum)*mixing_weight
        #print('.......',new_mixing_weight)
        return new_mixing_weight
    def update_innner_product_momentum_buffer(self,inner_product_group):
        for ind,grad,grad_buffer in zip(range(self.size),inner_product_group,self.inner_product_buffer): 
            if grad_buffer==None:
                grad_buffer=grad
            else:
                grad_buffer=self.mixing_momentum*grad_buffer+(1-self.mixing_momentum)*grad
            self.inner_product_buffer[ind]=grad_buffer.detach()
            self.inner_product_buffer_forward[ind]=grad_buffer
        print('.......',self.inner_product_buffer_forward)

        return inner_product_group
    def update_momentum_buffer(self,grad_group,key_tensor):
        for ind,grad,grad_buffer in zip(range(self.size),grad_group,self.grad_group_buffer): 
            if grad_buffer==None:
                grad_buffer=grad
            else:
                grad_buffer=self.momentum*grad_buffer+(1-self.momentum)*grad
            self.grad_group_buffer[ind]=grad_buffer

        if self.key_tensor_buffer==None:
            self.key_tensor_buffer=key_tensor
        else:
            self.key_tensor_buffer=self.momentum*self.key_tensor_buffer+(1-self.momentum)*key_tensor

        return self.grad_group_buffer,self.key_tensor_buffer
    def preprocess(self,model_update_group):
        #grad_group,val_grad=self.preprocess_moemntum(grad_group,val_grad)
        model_update_group1=self.normalize_model_update_group(model_update_group)
        grad_group=model_update_group1
        val_grad=model_update_group1[self.rank]
        new_grad_group=[]
        for i in range(self.num_weights):
            tmp_grad_group=[]
            for j in range(self.size):
                model_param=grad_group[j][i].detach()
                model_param=model_param.view(model_param.size(0),-1)
                tmp_grad_group.append(model_param)
            tmp_grad_group=torch.stack(tmp_grad_group)
            new_grad_group.append(tmp_grad_group)
        key_tensor_group=[]
        for i in range(self.num_weights):
            model_param=val_grad[i].detach()
            model_param=model_param.view(model_param.size(0),-1)
            key_tensor_group.append(model_param)
        



        '''
        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        '''
        #new_grad_group,key_tensor=self.update_momentum_buffer(new_grad_group,key_tensor)
        #print(new_grad_group)
        
        '''
        print('mean=',[torch.mean(i) for i in new_grad_group])
        print('abs=',[torch.mean(torch.abs(i)) for i in new_grad_group])
        '''
        return new_grad_group,key_tensor_group
    def preprocess1(self,grad_group,val_grad):
        new_grad_group=[]
        for i in range(self.num_weights):
            tmp_grad_group=[]
            for j in range(self.size):
                model_param=grad_group[j][i].detach()
                model_param=model_param.view(model_param.size(0),-1)
                tmp_grad_group.append(model_param/torch.norm(model_param))
            tmp_grad_group=torch.stack(tmp_grad_group)
            new_grad_group.append(tmp_grad_group)
        key_tensor_group=[]
        for i in range(self.num_weights):
            model_param=val_grad[i].detach()
            model_param=model_param.view(model_param.size(0),-1)
            key_tensor_group.append(model_param)
        



        '''
        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        '''
        #new_grad_group,key_tensor=self.update_momentum_buffer(new_grad_group,key_tensor)
        #print(new_grad_group)
        
        '''
        print('mean=',[torch.mean(i) for i in new_grad_group])
        print('abs=',[torch.mean(torch.abs(i)) for i in new_grad_group])
        '''
        return new_grad_group,key_tensor_group
    def preprocess2(self,grad_group,val_grad):
        new_grad_group=[]
        for i in range(self.num_weights):
            tmp_grad_group=[]
            for j in range(self.size):
                model_param=grad_group[j][i].detach()
                model_param=model_param.view(model_param.size(0),-1)
                tmp_grad_group.append(model_param/torch.norm(model_param))
            tmp_grad_group=torch.stack(tmp_grad_group)
            new_grad_group.append(tmp_grad_group)
        key_tensor_group=[]
        for i in range(self.num_weights):
            model_param=val_grad[i].detach()
            model_param=model_param.view(model_param.size(0),-1)
            key_tensor_group.append(model_param)
        



        '''
        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        '''
        #new_grad_group,key_tensor=self.update_momentum_buffer(new_grad_group,key_tensor)
        #print(new_grad_group)
        
        '''
        print('mean=',[torch.mean(i) for i in new_grad_group])
        print('abs=',[torch.mean(torch.abs(i)) for i in new_grad_group])
        '''
        return new_grad_group,key_tensor_group
    def preprocess_moemntum(self,grad_group,val_grad):
        new_grad_group=[comm_helpers.flatten_tensors(i).detach() for i in grad_group]
        key_tensor=comm_helpers.flatten_tensors(val_grad).detach()
        new_grad_group,key_tensor=self.update_momentum_buffer(new_grad_group,key_tensor)
        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)

        new_grad_group=[comm_helpers.unflatten_tensors(i,val_grad) for i in new_grad_group]
        key_tensor=comm_helpers.unflatten_tensors(key_tensor,val_grad)
        return new_grad_group,key_tensor
    def normalize_model_update_group(self,model_update_group):
        new_grad_group=[comm_helpers.flatten_tensors(i).detach() for i in model_update_group]
        new_grad_group=[i/torch.norm(i) for i in new_grad_group]

        new_grad_group=[comm_helpers.unflatten_tensors(i,model_update_group[0]) for i in new_grad_group]
        return new_grad_group
    
class Attention_layerwise_model_update_multilayer(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, model, out_dim,rank, comm,size):
        super().__init__()


        #self.in_dim=in_dim
        self.out_dim=out_dim
        out_dim1=10
        out_dim2=10
        self.in_dim_group=[]
        self.w_qs_group1=[]
        self.w_ks_group1=[]
        self.w_qs_group2=[]
        self.w_ks_group2=[]
        self.num_weights=0
        for i in model.params():
            param_data=i.detach().data
            param_data=param_data.view(param_data.size(0),-1)
            param_size=param_data.size()
            in_dim=param_size[1]
            self.in_dim_group.append(in_dim)
            
            self.w_qs_group1.append(nn.Linear(in_dim, out_dim1))
            #self.w_ks_group1.append(nn.Linear(in_dim, out_dim1))
            self.w_qs_group2.append(nn.Linear(out_dim1, out_dim2))
            #self.w_ks_group2.append(nn.Linear(out_dim1, out_dim2))
            
            '''
            self.w_qs_group1.append(nn.Linear(in_dim, out_dim1,bias=False))
            self.w_ks_group1.append(nn.Linear(in_dim, out_dim1,bias=False))
            self.w_qs_group2.append(nn.Linear(out_dim1, out_dim2,bias=False))
            self.w_ks_group2.append(nn.Linear(out_dim1, out_dim2,bias=False))
            '''
            self.num_weights=self.num_weights+1
            '''
            self.w_qs_group[-1].weight.data=self.w_qs_group[-1].weight.data
            self.w_ks_group[-1].weight.data=self.w_ks_group[-1].weight.data*10
            '''
        self.w_qs_layers1 = nn.ModuleList(self.w_qs_group1)
        #self.w_ks_layers1 = nn.ModuleList(self.w_ks_group1)

        self.w_qs_layers2 = nn.ModuleList(self.w_qs_group2)
        #self.w_ks_layers2 = nn.ModuleList(self.w_ks_group2)
            


        self.rank=rank
        self.comm=comm
        self.size=size
        self.grad_group_buffer=[None for i in range(size)]
        self.key_tensor_buffer=None
        self.grad_group_buffer_layerwise=[[None for i in range(size)] for j in range(10)]
        self.key_tensor_buffer_layerwise=[None for i in range(10)]
        self.inner_product_buffer=[None for i in range(size)]
        self.inner_product_buffer_forward=[None for i in range(size)]
        self.mixing_weight_buffer=None
        self.momentum=0.99
        self.mixing_momentum=0.99
        #self.register_parameter(name='mixing_momentum', param=torch.nn.Parameter(0.5*torch.ones(1)))
        self.iteration=0
        self.leaky_relu=nn.LeakyReLU(0.1)
        


    def forward(self, model_update_group):
        q_group,k_group=self.preprocess(model_update_group)
        #q_group size=[num_weights,neighbour_num,num_filter,in_dim] list of tensors
        #k_group size=[num_weights,num_filter,in_dim] list of tensors

        '''
        q=q*50
        k=k*50
        '''
        
        prod_sum=torch.zeros(self.size).cuda()
        for i in range(self.num_weights):

            q=q_group[i]
            k=k_group[i]
            neighbour_num,num_filter,in_dim=q.size()

            '''
            q = self.w_qs_layers2[i](F.relu(self.w_qs_layers1[i](q)))
            k = self.w_qs_layers2[i](F.relu(self.w_qs_layers1[i](k)))
            '''
            
            '''
            q = self.leaky_relu(self.w_qs_layers2[i](F.relu(self.w_qs_layers1[i](q))))
            k = self.leaky_relu(self.w_qs_layers2[i](F.relu(self.w_qs_layers1[i](k))))
            '''
            q = F.tanh(self.w_qs_layers2[i](F.tanh(self.w_qs_layers1[i](q))))
            k = F.tanh(self.w_qs_layers2[i](F.tanh(self.w_qs_layers1[i](k))))

            q=q.view(neighbour_num,-1)
            k=k.view(-1,1)
            '''
            if self.rank==0:
                print(q.size())
            '''



            prod=torch.matmul(q,k).view(-1)/num_filter
            prod_sum=prod_sum+prod
            #mixing_weight=mixing_weight/(self.out_dim**0.5)
        prod_sum=prod_sum.view(-1)/self.num_weights
        #print('rank=',self.rank,prod_sum)
        #print(mixing_weight)
        mixing_weight=F.softmax(prod_sum,dim=0)


        mixing_weight=mixing_weight.view(-1)



        return mixing_weight
    def update_mixing_momentum_buffer(self,mixing_weight):
        if self.mixing_weight_buffer==None:
            self.mixing_weight_buffer=mixing_weight.detach()
            new_mixing_weight=mixing_weight
        else:
            self.mixing_weight_buffer=self.mixing_momentum*self.mixing_weight_buffer+\
            (1-self.mixing_momentum)*mixing_weight.detach()
            new_mixing_weight=self.mixing_momentum*self.mixing_weight_buffer+\
            (1-self.mixing_momentum)*mixing_weight
        #print('.......',new_mixing_weight)
        return new_mixing_weight
    def update_innner_product_momentum_buffer(self,inner_product_group):
        for ind,grad,grad_buffer in zip(range(self.size),inner_product_group,self.inner_product_buffer): 
            if grad_buffer==None:
                grad_buffer=grad
            else:
                grad_buffer=self.mixing_momentum*grad_buffer+(1-self.mixing_momentum)*grad
            self.inner_product_buffer[ind]=grad_buffer.detach()
            self.inner_product_buffer_forward[ind]=grad_buffer
        print('.......',self.inner_product_buffer_forward)

        return inner_product_group
    def update_momentum_buffer(self,grad_group,key_tensor):
        for ind,grad,grad_buffer in zip(range(self.size),grad_group,self.grad_group_buffer): 
            if grad_buffer==None:
                grad_buffer=grad
            else:
                grad_buffer=self.momentum*grad_buffer+(1-self.momentum)*grad
            self.grad_group_buffer[ind]=grad_buffer

        if self.key_tensor_buffer==None:
            self.key_tensor_buffer=key_tensor
        else:
            self.key_tensor_buffer=self.momentum*self.key_tensor_buffer+(1-self.momentum)*key_tensor

        return self.grad_group_buffer,self.key_tensor_buffer
    def preprocess3(self,model_update_group):
        #grad_group,val_grad=self.preprocess_moemntum(grad_group,val_grad)
        model_update_group1=self.normalize_model_update_group(model_update_group)
        grad_group=model_update_group1
        val_grad=model_update_group1[self.rank]
        new_grad_group=[]
        for i in range(self.num_weights):

            tmp_grad_group=[]
            for j in range(self.size):
                model_param=grad_group[j][i].detach()
                model_param=model_param.view(model_param.size(0),-1)
                tmp_grad_group.append(model_param)
            tmp_grad_group=torch.stack(tmp_grad_group)
            new_grad_group.append(tmp_grad_group)
        key_tensor_group=[]
        for i in range(self.num_weights):
            model_param=val_grad[i].detach()
            model_param=model_param.view(model_param.size(0),-1)
            key_tensor_group.append(model_param)
        



        '''
        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        '''
        #new_grad_group,key_tensor=self.update_momentum_buffer(new_grad_group,key_tensor)
        #print(new_grad_group)
        
        '''
        print('mean=',[torch.mean(i) for i in new_grad_group])
        print('abs=',[torch.mean(torch.abs(i)) for i in new_grad_group])
        '''
        return new_grad_group,key_tensor_group
    def preprocess(self,model_update_group):
        #grad_group,val_grad=self.preprocess_moemntum(grad_group,val_grad)
        model_update_group1=self.normalize_model_update_group(model_update_group)
        grad_group=model_update_group1
        val_grad=model_update_group1[self.rank]
        new_grad_group=[]
        for i in range(self.num_weights):
            tmp_grad_group=[]
            for j in range(self.size):
                model_param=grad_group[j][i].detach()
                model_param=model_param.view(model_param.size(0),-1)
                #model_param=model_param/(torch.norm(model_param,dim=1,keepdim=True)+1e-10)
                tmp_grad_group.append(model_param)
            tmp_grad_group=torch.stack(tmp_grad_group)
            new_grad_group.append(tmp_grad_group)
        key_tensor_group=[]
        for i in range(self.num_weights):
            model_param=val_grad[i].detach()
            model_param=model_param.view(model_param.size(0),-1)
            key_tensor_group.append(model_param)
        



        '''
        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        '''
        #new_grad_group,key_tensor=self.update_momentum_buffer(new_grad_group,key_tensor)
        #print(new_grad_group)
        
        '''
        print('mean=',[torch.mean(i) for i in new_grad_group])
        print('abs=',[torch.mean(torch.abs(i)) for i in new_grad_group])
        '''
        return new_grad_group,key_tensor_group
    def preprocess1(self,grad_group,val_grad):
        new_grad_group=[]
        for i in range(self.num_weights):
            tmp_grad_group=[]
            for j in range(self.size):
                model_param=grad_group[j][i].detach()
                model_param=model_param.view(model_param.size(0),-1)
                tmp_grad_group.append(model_param/torch.norm(model_param))
            tmp_grad_group=torch.stack(tmp_grad_group)
            new_grad_group.append(tmp_grad_group)
        key_tensor_group=[]
        for i in range(self.num_weights):
            model_param=val_grad[i].detach()
            model_param=model_param.view(model_param.size(0),-1)
            key_tensor_group.append(model_param)
        



        '''
        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        '''
        #new_grad_group,key_tensor=self.update_momentum_buffer(new_grad_group,key_tensor)
        #print(new_grad_group)
        
        '''
        print('mean=',[torch.mean(i) for i in new_grad_group])
        print('abs=',[torch.mean(torch.abs(i)) for i in new_grad_group])
        '''
        return new_grad_group,key_tensor_group
    def preprocess2(self,grad_group,val_grad):
        new_grad_group=[]
        for i in range(self.num_weights):
            tmp_grad_group=[]
            for j in range(self.size):
                model_param=grad_group[j][i].detach()
                model_param=model_param.view(model_param.size(0),-1)
                tmp_grad_group.append(model_param/torch.norm(model_param))
            tmp_grad_group=torch.stack(tmp_grad_group)
            new_grad_group.append(tmp_grad_group)
        key_tensor_group=[]
        for i in range(self.num_weights):
            model_param=val_grad[i].detach()
            model_param=model_param.view(model_param.size(0),-1)
            key_tensor_group.append(model_param)
        



        '''
        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        '''
        #new_grad_group,key_tensor=self.update_momentum_buffer(new_grad_group,key_tensor)
        #print(new_grad_group)
        
        '''
        print('mean=',[torch.mean(i) for i in new_grad_group])
        print('abs=',[torch.mean(torch.abs(i)) for i in new_grad_group])
        '''
        return new_grad_group,key_tensor_group
    def preprocess_moemntum(self,grad_group,val_grad):
        new_grad_group=[comm_helpers.flatten_tensors(i).detach() for i in grad_group]
        key_tensor=comm_helpers.flatten_tensors(val_grad).detach()
        new_grad_group,key_tensor=self.update_momentum_buffer(new_grad_group,key_tensor)
        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)

        new_grad_group=[comm_helpers.unflatten_tensors(i,val_grad) for i in new_grad_group]
        key_tensor=comm_helpers.unflatten_tensors(key_tensor,val_grad)
        return new_grad_group,key_tensor
    def normalize_model_update_group(self,model_update_group):
        #print('aaa')
        new_grad_group=copy.deepcopy(model_update_group)
        new_grad_group=[comm_helpers.flatten_tensors(i).detach() for i in new_grad_group]
        new_grad_group=[i/(torch.norm(i)+1e-10) for i in new_grad_group]
        #print([torch.mean(i) for i in new_grad_group])
        #print([torch.sqrt(torch.var(i)) for i in new_grad_group])

        new_grad_group=[comm_helpers.unflatten_tensors(i,model_update_group[0]) for i in new_grad_group]
        return new_grad_group
















class Attention_layerwise_model_update_multilayer_mask(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, model, out_dim,rank, comm,size):
        super().__init__()


        #self.in_dim=in_dim
        self.out_dim=out_dim
        out_dim1=10
        out_dim2=20
        self.in_dim_group=[]
        self.w_qs_group1=[]
        self.w_ks_group1=[]
        self.w_qs_group2=[]
        self.w_ks_group2=[]
        self.num_weights=0
        for i in model.params():
            param_data=i.detach().data
            param_data=param_data.view(param_data.size(0),-1)
            param_size=param_data.size()
            in_dim=param_size[1]
            self.in_dim_group.append(in_dim)
            
            self.w_qs_group1.append(nn.Linear(in_dim, out_dim1))
            #self.w_ks_group1.append(nn.Linear(in_dim, out_dim1))
            self.w_qs_group2.append(nn.Linear(out_dim1, out_dim2))
            #self.w_ks_group2.append(nn.Linear(out_dim1, out_dim2))
            
            '''
            self.w_qs_group1.append(nn.Linear(in_dim, out_dim1,bias=False))
            self.w_ks_group1.append(nn.Linear(in_dim, out_dim1,bias=False))
            self.w_qs_group2.append(nn.Linear(out_dim1, out_dim2,bias=False))
            self.w_ks_group2.append(nn.Linear(out_dim1, out_dim2,bias=False))
            '''
            self.num_weights=self.num_weights+1
            '''
            self.w_qs_group[-1].weight.data=self.w_qs_group[-1].weight.data
            self.w_ks_group[-1].weight.data=self.w_ks_group[-1].weight.data*10
            '''
        self.w_qs_layers1 = nn.ModuleList(self.w_qs_group1)
        #self.w_ks_layers1 = nn.ModuleList(self.w_ks_group1)

        self.w_qs_layers2 = nn.ModuleList(self.w_qs_group2)
        #self.w_ks_layers2 = nn.ModuleList(self.w_ks_group2)
            


        self.rank=rank
        self.comm=comm
        self.size=size
        self.grad_group_buffer=[None for i in range(size)]
        self.key_tensor_buffer=None
        self.grad_group_buffer_layerwise=[[None for i in range(size)] for j in range(10)]
        self.key_tensor_buffer_layerwise=[None for i in range(10)]
        self.inner_product_buffer=[None for i in range(size)]
        self.inner_product_buffer_forward=[None for i in range(size)]
        self.mixing_weight_buffer=None
        self.momentum=0.99
        self.mixing_momentum=0.99
        #self.register_parameter(name='mixing_momentum', param=torch.nn.Parameter(0.5*torch.ones(1)))
        self.iteration=0
        self.leaky_relu=nn.LeakyReLU(0.1)
        


    def forward(self, model_update_group,mixing_weight_mask=None):
        q_group,k_group,neighbour_num=self.preprocess(model_update_group,mixing_weight_mask)
        #q_group size=[num_weights,neighbour_num,num_filter,in_dim] list of tensors
        #k_group size=[num_weights,num_filter,in_dim] list of tensors

        '''
        q=q*50
        k=k*50
        '''
        
        prod_sum=torch.zeros(neighbour_num).cuda()
        for i in range(self.num_weights):

            q=q_group[i]
            k=k_group[i]
            neighbour_num,num_filter,in_dim=q.size()

            '''
            q = self.w_qs_layers2[i](F.relu(self.w_qs_layers1[i](q)))
            k = self.w_qs_layers2[i](F.relu(self.w_qs_layers1[i](k)))
            '''
            
            '''
            q = self.leaky_relu(self.w_qs_layers2[i](F.relu(self.w_qs_layers1[i](q))))
            k = self.leaky_relu(self.w_qs_layers2[i](F.relu(self.w_qs_layers1[i](k))))
            '''
            q = F.tanh(self.w_qs_layers2[i](F.tanh(self.w_qs_layers1[i](q))))
            k = F.tanh(self.w_qs_layers2[i](F.tanh(self.w_qs_layers1[i](k))))

            q=q.view(neighbour_num,-1)
            k=k.view(-1,1)
            '''
            if self.rank==0:
                print(q.size())
            '''



            prod=torch.matmul(q,k).view(-1)/num_filter
            #print(prod)
            prod_sum=prod_sum+prod
            #mixing_weight=mixing_weight/(self.out_dim**0.5)
        prod_sum=prod_sum.view(-1)/self.num_weights
        #print('rank=',self.rank,prod_sum)
        #print(mixing_weight)
        mixing_weight=F.softmax(prod_sum,dim=0)


        mixing_weight=mixing_weight.view(-1)



        return mixing_weight
   
       
   

    def preprocess(self,model_update_group,mixing_weight_mask=None):
        #grad_group,val_grad=self.preprocess_moemntum(grad_group,val_grad)
        model_update_group1,val_grad=self.normalize_model_update_group(model_update_group)
        if mixing_weight_mask==None:
            grad_group=model_update_group1
        else:
            grad_group=[]
            for i in mixing_weight_mask:
                grad_group.append(model_update_group1[i])
        
        neighbour_num=len(grad_group)
        new_grad_group=[]
        for i in range(self.num_weights):
            tmp_grad_group=[]
            for j in range(neighbour_num):
                model_param=grad_group[j][i].detach()
                model_param=model_param.view(model_param.size(0),-1)
                #model_param=model_param/(torch.norm(model_param,dim=1,keepdim=True)+1e-10)
                tmp_grad_group.append(model_param)
            tmp_grad_group=torch.stack(tmp_grad_group)
            new_grad_group.append(tmp_grad_group)
        key_tensor_group=[]
        for i in range(self.num_weights):
            model_param=val_grad[i].detach()
            model_param=model_param.view(model_param.size(0),-1)
            key_tensor_group.append(model_param)
        



        '''
        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        '''
        #new_grad_group,key_tensor=self.update_momentum_buffer(new_grad_group,key_tensor)
        #print(new_grad_group)
        
        '''
        print('mean=',[torch.mean(i) for i in new_grad_group])
        print('abs=',[torch.mean(torch.abs(i)) for i in new_grad_group])
        '''
        return new_grad_group,key_tensor_group,neighbour_num
    
    
    
    def normalize_model_update_group(self,model_update_group):
        new_grad_group=copy.deepcopy(model_update_group)
        new_grad_group=[comm_helpers.flatten_tensors(i).detach() for i in new_grad_group]
        new_grad_group=[i/(torch.norm(i)+1e-10) for i in new_grad_group]
        #print([torch.mean(i) for i in new_grad_group])
        #print([torch.sqrt(torch.var(i)) for i in new_grad_group])

        new_grad_group=[comm_helpers.unflatten_tensors(i,model_update_group[0]) for i in new_grad_group]
        val_grad=new_grad_group[self.rank]
        return new_grad_group,val_grad


      
class Attention_dot_model_update_mask1(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, model, out_dim,rank, comm,size):
        super().__init__()


        #self.in_dim=in_dim
        self.out_dim=out_dim
        in_dim=1
        out_dim1=10
        out_dim2=3
        
        out_dim1=1
        out_dim2=1
        
        self.in_dim_group=[]
        self.w_qs_group1=[]
        self.w_ks_group1=[]
        self.w_qs_group2=[]
        self.w_ks_group2=[]
        self.num_weights=0

        self.channel_sum=0

        for i in model.params():
            
            
            self.num_weights=self.num_weights+1
            self.channel_sum=self.channel_sum+i.size(0)
        self.channel_avg=float(self.channel_sum)/self.num_weights
            
        self.w_qs_layers1 = nn.Linear(in_dim, out_dim1,bias=False)
        init1=np.sqrt(1.0/out_dim1)
        self.w_qs_layers1.weight.data=torch.tensor([1.0]).view(1,1)
        #self.w_ks_layers1 = nn.ModuleList(self.w_ks_group1)

        self.w_qs_layers2 = nn.Linear(out_dim1, out_dim2,bias=False)
        init2=np.sqrt(1.0/out_dim2)*10.0
        self.w_qs_layers2.weight.data=torch.tensor([1.0]).view(1,1)
        #self.w_ks_layers2 = nn.ModuleList(self.w_ks_group2)
            


        self.rank=rank
        self.comm=comm
        self.size=size
        
        


    def forward(self, model_update_group,mixing_weight_mask=None):
        q_group,k_group,neighbour_num=self.preprocess(model_update_group,mixing_weight_mask)
        #q_group size=[num_weights,neighbour_num,num_filter,in_dim] list of tensors
        #k_group size=[num_weights,num_filter,in_dim] list of tensors
        length=float(q_group.size(1))
        '''
        q=q*50
        k=k*50
        '''
        q_group=q_group.view(neighbour_num,-1,1)
        k_group=k_group.view(-1,1)
        #print(self.num_weights)
        q1=F.tanh(self.w_qs_layers1(q_group))
        q = F.tanh(self.w_qs_layers2(q1))#/self.num_weights
        #print('q1=',q1)
        #print('q2=',q)
        k1=F.tanh(self.w_qs_layers1(k_group))
        k = F.tanh(self.w_qs_layers2(k1))#/self.num_weights
        
        q=q.view(neighbour_num,-1)
        k=k.view(-1,1)

        prod_sum=torch.matmul(q,k).view(-1)#/float(length)
        #print(prod_sum)
        mixing_weight=F.softmax(prod_sum,dim=0)


       


        return mixing_weight
   
       
   

    def preprocess(self,model_update_group,mixing_weight_mask=None):
        #grad_group,val_grad=self.preprocess_moemntum(grad_group,val_grad)
        model_update_group1,val_grad=self.normalize_model_update_group(model_update_group)
        if mixing_weight_mask==None:
            grad_group=model_update_group1
        else:
            grad_group=[]
            for i in mixing_weight_mask:
                grad_group.append(model_update_group1[i])
        
        neighbour_num=len(grad_group)
        grad_group=torch.stack(grad_group)
        


        
        return grad_group,val_grad,neighbour_num
    
    
    
    def normalize_model_update_group(self,model_update_group):
        new_grad_group=copy.deepcopy(model_update_group)
        new_grad_group=[comm_helpers.flatten_tensors(i).detach() for i in new_grad_group]
        new_grad_group=[i/(torch.norm(i)+1e-10) for i in new_grad_group]
        #print([torch.mean(i) for i in new_grad_group])
        #print([torch.sqrt(torch.var(i)) for i in new_grad_group])

        
        val_grad=new_grad_group[self.rank]
        return new_grad_group,val_grad

class Attention_dot_model_update_mask(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, model, out_dim,rank, comm,size):
        super().__init__()


        #self.in_dim=in_dim
        self.out_dim=out_dim
        in_dim=1
        
        
        out_dim1=5
        out_dim2=5
        
        self.in_dim_group=[]
        self.w_qs_group1=[]
        self.w_ks_group1=[]
        self.w_qs_group2=[]
        self.w_ks_group2=[]
        self.num_weights=0

        self.channel_sum=0

        for i in model.params():
            
            
            self.num_weights=self.num_weights+1
            self.channel_sum=self.channel_sum+i.size(0)
        self.channel_avg=float(self.channel_sum)/self.num_weights
            
        self.w_qs_layers1 = nn.Linear(in_dim, out_dim1,bias=False)
        '''
        init1=np.sqrt(1.0/out_dim1)
        self.w_qs_layers1.weight.data=torch.tensor([1.0]).view(1,1)
        '''
       

        self.w_qs_layers2 = nn.Linear(out_dim1, out_dim2,bias=False)
        '''
        init2=np.sqrt(1.0/out_dim2)*10.0
        self.w_qs_layers2.weight.data=torch.tensor([1.0]).view(1,1)
        '''
        
            


        self.rank=rank
        self.comm=comm
        self.size=size
        
        


    def forward(self, model_update_group,mixing_weight_mask=None):
        q_group,k_group,neighbour_num=self.preprocess(model_update_group,mixing_weight_mask)
        #q_group size=[num_weights,neighbour_num,num_filter,in_dim] list of tensors
        #k_group size=[num_weights,num_filter,in_dim] list of tensors
        length=float(q_group.size(1))
        '''
        q=q*50
        k=k*50
        '''
        q_group=q_group.view(neighbour_num,-1,1)
        k_group=k_group.view(-1,1)
        #print(self.num_weights)
        q1=F.tanh(self.w_qs_layers1(q_group))
        q = F.tanh(self.w_qs_layers2(q1))#/self.num_weights
        #print('q1=',q1)
        #print('q2=',q)
        k1=F.tanh(self.w_qs_layers1(k_group))
        k = F.tanh(self.w_qs_layers2(k1))#/self.num_weights
        
        q=q.view(neighbour_num,-1)
        k=k.view(-1,1)

        prod_sum=torch.matmul(q,k).view(-1)#/float(length)
        #print(prod_sum)
        mixing_weight=F.softmax(prod_sum,dim=0)


       


        return mixing_weight
   
       
   

    def preprocess(self,model_update_group,mixing_weight_mask=None):
        #grad_group,val_grad=self.preprocess_moemntum(grad_group,val_grad)
        model_update_group1,val_grad=self.normalize_model_update_group(model_update_group)
        if mixing_weight_mask==None:
            grad_group=model_update_group1
        else:
            grad_group=[]
            for i in mixing_weight_mask:
                grad_group.append(model_update_group1[i])
        
        neighbour_num=len(grad_group)
        grad_group=torch.stack(grad_group)
        


        
        return grad_group,val_grad,neighbour_num
    
    
    
    def normalize_model_update_group(self,model_update_group):
        new_grad_group=copy.deepcopy(model_update_group)
        new_grad_group=[comm_helpers.flatten_tensors(i).detach() for i in new_grad_group]
        new_grad_group=[i/(torch.norm(i)+1e-10) for i in new_grad_group]
        #print([torch.mean(i) for i in new_grad_group])
        #print([torch.sqrt(torch.var(i)) for i in new_grad_group])

        
        val_grad=new_grad_group[self.rank]
        return new_grad_group,val_grad


      




class Attention_layerwise_model_update_multilayer_mask_randchoose_channel(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, model, out_dim,rank, comm,size,device_name,selected_num_filter):
        super().__init__()


        #self.in_dim=in_dim
        self.out_dim=out_dim
        out_dim1=10
        out_dim2=20
        self.in_dim_group=[]
        self.w_qs_group1=[]
        self.w_ks_group1=[]
        self.w_qs_group2=[]
        self.w_ks_group2=[]
        self.num_weights=0
        for i in model.params():
            param_data=i.detach().data
            param_data=param_data.view(param_data.size(0),-1)
            param_size=param_data.size()
            in_dim=param_size[1]
            self.in_dim_group.append(in_dim)
            
            self.w_qs_group1.append(nn.Linear(in_dim, out_dim1))
            #self.w_ks_group1.append(nn.Linear(in_dim, out_dim1))
            self.w_qs_group2.append(nn.Linear(out_dim1, out_dim2))
            #self.w_ks_group2.append(nn.Linear(out_dim1, out_dim2))
            
            '''
            self.w_qs_group1.append(nn.Linear(in_dim, out_dim1,bias=False))
            self.w_ks_group1.append(nn.Linear(in_dim, out_dim1,bias=False))
            self.w_qs_group2.append(nn.Linear(out_dim1, out_dim2,bias=False))
            self.w_ks_group2.append(nn.Linear(out_dim1, out_dim2,bias=False))
            '''
            self.num_weights=self.num_weights+1
            '''
            self.w_qs_group[-1].weight.data=self.w_qs_group[-1].weight.data
            self.w_ks_group[-1].weight.data=self.w_ks_group[-1].weight.data*10
            '''
        self.w_qs_layers1 = nn.ModuleList(self.w_qs_group1)
        #self.w_ks_layers1 = nn.ModuleList(self.w_ks_group1)

        self.w_qs_layers2 = nn.ModuleList(self.w_qs_group2)
        #self.w_ks_layers2 = nn.ModuleList(self.w_ks_group2)
            


        self.rank=rank
        self.comm=comm
        self.size=size
        self.grad_group_buffer=[None for i in range(size)]
        self.key_tensor_buffer=None
        self.grad_group_buffer_layerwise=[[None for i in range(size)] for j in range(10)]
        self.key_tensor_buffer_layerwise=[None for i in range(10)]
        self.inner_product_buffer=[None for i in range(size)]
        self.inner_product_buffer_forward=[None for i in range(size)]
        self.mixing_weight_buffer=None
        self.momentum=0.99
        self.mixing_momentum=0.99
        #self.register_parameter(name='mixing_momentum', param=torch.nn.Parameter(0.5*torch.ones(1)))
        self.iteration=0
        self.leaky_relu=nn.LeakyReLU(0.1)
        self.device_name=device_name
        self.selected_num_filter=selected_num_filter


    def forward(self, model_update_group,mixing_weight_mask=None):
        q_group,k_group,neighbour_num=self.preprocess(model_update_group,mixing_weight_mask)
        #q_group size=[num_weights,neighbour_num,num_filter,in_dim] list of tensors
        #k_group size=[num_weights,num_filter,in_dim] list of tensors

        '''
        q=q*50
        k=k*50
        '''
        
        prod_sum=torch.zeros(neighbour_num).cuda()
        for i in range(self.num_weights):

            q=q_group[i]
            k=k_group[i]
            neighbour_num,num_filter,in_dim=q.size()
            #print(num_filter)
            if self.selected_num_filter>=num_filter:
                selcted_ind=np.random.choice(num_filter,num_filter,replace=False)
            else:
                selcted_ind=np.random.choice(num_filter,self.selected_num_filter,replace=False)
            if self.device_name=='cpu':
                selcted_ind1=torch.tensor(selcted_ind)
            if self.device_name=='cuda':
                selcted_ind1=torch.tensor(selcted_ind).cuda()
            q=torch.index_select(q,1,selcted_ind1)
            k=torch.index_select(k,0,selcted_ind1)

           

            '''
            q = self.w_qs_layers2[i](F.relu(self.w_qs_layers1[i](q)))
            k = self.w_qs_layers2[i](F.relu(self.w_qs_layers1[i](k)))
            '''
            
            '''
            q = self.leaky_relu(self.w_qs_layers2[i](F.relu(self.w_qs_layers1[i](q))))
            k = self.leaky_relu(self.w_qs_layers2[i](F.relu(self.w_qs_layers1[i](k))))
            '''
            q = F.tanh(self.w_qs_layers2[i](F.tanh(self.w_qs_layers1[i](q))))
            k = F.tanh(self.w_qs_layers2[i](F.tanh(self.w_qs_layers1[i](k))))

            q=q.view(neighbour_num,-1)
            k=k.view(-1,1)
            '''
            if self.rank==0:
                print(q.size())
            '''



            prod=torch.matmul(q,k).view(-1)/self.selected_num_filter
            #print(prod)
            prod_sum=prod_sum+prod
            #mixing_weight=mixing_weight/(self.out_dim**0.5)
        prod_sum=prod_sum.view(-1)/self.num_weights
        #print('rank=',self.rank,prod_sum)
        #print(mixing_weight)
        mixing_weight=F.softmax(prod_sum,dim=0)


        mixing_weight=mixing_weight.view(-1)



        return mixing_weight
   
       
   

    def preprocess(self,model_update_group,mixing_weight_mask=None):
        #grad_group,val_grad=self.preprocess_moemntum(grad_group,val_grad)
        model_update_group1,val_grad=self.normalize_model_update_group(model_update_group)
        if mixing_weight_mask==None:
            grad_group=model_update_group1
        else:
            grad_group=[]
            for i in mixing_weight_mask:
                grad_group.append(model_update_group1[i])
        
        neighbour_num=len(grad_group)
        new_grad_group=[]
        for i in range(self.num_weights):
            tmp_grad_group=[]
            for j in range(neighbour_num):
                model_param=grad_group[j][i].detach()
                model_param=model_param.view(model_param.size(0),-1)
                #model_param=model_param/(torch.norm(model_param,dim=1,keepdim=True)+1e-10)
                tmp_grad_group.append(model_param)
            tmp_grad_group=torch.stack(tmp_grad_group)
            new_grad_group.append(tmp_grad_group)
        key_tensor_group=[]
        for i in range(self.num_weights):
            model_param=val_grad[i].detach()
            model_param=model_param.view(model_param.size(0),-1)
            key_tensor_group.append(model_param)
        



        '''
        new_grad_group=[i/torch.norm(i) for i in new_grad_group]
        key_tensor=key_tensor/torch.norm(key_tensor)
        '''
        #new_grad_group,key_tensor=self.update_momentum_buffer(new_grad_group,key_tensor)
        #print(new_grad_group)
        
        '''
        print('mean=',[torch.mean(i) for i in new_grad_group])
        print('abs=',[torch.mean(torch.abs(i)) for i in new_grad_group])
        '''
        return new_grad_group,key_tensor_group,neighbour_num
    
    
    
    def normalize_model_update_group(self,model_update_group):
        new_grad_group=copy.deepcopy(model_update_group)
        new_grad_group=[comm_helpers.flatten_tensors(i).detach() for i in new_grad_group]
        new_grad_group=[i/(torch.norm(i)+1e-10) for i in new_grad_group]
        #print([torch.mean(i) for i in new_grad_group])
        #print([torch.sqrt(torch.var(i)) for i in new_grad_group])

        new_grad_group=[comm_helpers.unflatten_tensors(i,model_update_group[0]) for i in new_grad_group]
        val_grad=new_grad_group[self.rank]
        return new_grad_group,val_grad


class GCN(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, size):
        super().__init__()

        self.register_parameter(name='model_weight', param=torch.nn.Parameter(torch.zeros(size)))


    def forward(self, model_update_group,mixing_weight_mask=None):
        mixing_weight=F.softmax(self.model_weight)
        return mixing_weight

    