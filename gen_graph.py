import collections
import random
import numpy as np
def gen_graph(size,args):
    if args.graph_name=='full':
        mixing_mat=np.ones((size,size))/size
    
    if args.graph_name=='single':
        mixing_mat=np.identity(size)

    if args.graph_name=='connect_same':
        split_num=args.split_label_group_num
        mat=np.zeros((size,size))
        every_split_num=size//split_num
        for i in range(size):
                for j in range(size):
                        i_=i//every_split_num
                        j_=j//every_split_num
                        if i_==j_:
                                mat[i][j]=1.0/every_split_num
        mixing_mat=mat
    return mixing_mat


