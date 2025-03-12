# ------------------------------------------------------------------------------
# Title:        Causality Calculation Script
# Description:  This is the main file that calls causality and psi1 and psi2 tests functions and bootstrap.
# Author:       Sanaz Panahandeh , Seulgi Lee
# Date:         2023-09-01
# Version:      2025-03-10
# ------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import tools, entropy, output
import cProfile
from scipy.sparse import issparse
import torch
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def causation(DataTensor,g1,g2,args):
    TensSize = DataTensor.size()[0]
    
    zeros_tensor = torch.zeros((DataTensor.size(0), 11), device=device)
    NewTensor = torch.cat((DataTensor, zeros_tensor), dim=1)

    m = args.neighbors
    
    #if m_variable:
    #    m=int((TensSize/5)**(1/3))
    #else:
    #    from param import m
        
    NewTensor,w , NN_Array =tools.neighbors(NewTensor,TensSize,m)
    NewTensor=tools.lag(NewTensor,w)
    NewTensor=tools.map(NewTensor,TensSize,NN_Array,args)
    
    bl = 6
    buoys=tools.buoys_finding(NewTensor)
    ordi=tools.blocks(NewTensor,buoys,TensSize,bl) ## n : / bl : number of block
    
    delh,YtoX=entropy.hm(NewTensor,TensSize)
    #pvalue,YtoX_pvalue=tools.bootstrap(NewTensor,w,delh,YtoX,TensSize,NN_Array,args)
    pvalue,YtoX_pvalue=tools.block_bootstrap(NewTensor,w,delh,YtoX,TensSize,NN_Array,bl,ordi,args)

    return pvalue,YtoX_pvalue, delh,YtoX,TensSize,m


def psi1test(DataTensor,g1,g2,args):
    TensSize = DataTensor.size()[0]
    
    zeros_tensor = torch.zeros((DataTensor.size(0), 11), device=device)
    NewTensor = torch.cat((DataTensor, zeros_tensor), dim=1)

    m = args.neighbors
    
    #if m_variable:
    #    m=int((TensSize/5)**(1/3))
    #else:
    #    from param import m
        
    NewTensor,w , NN_Array =tools.neighbors(NewTensor,TensSize,m)
    NewTensor=tools.map_xy(NewTensor,TensSize,NN_Array,args)
    entxy = entropy.hm_xy(NewTensor,TensSize)
    si = entropy.permut(NewTensor,TensSize,m)
    two = torch.tensor(2.0, device=device) 
    psi1=2*(m-1)*torch.log(two)-si-entxy
    #delh,YtoX=entropy.hm(NewTensor,TensSize)
    pvalue=tools.bootstrap_xy(NewTensor,psi1,TensSize,NN_Array,args)
    
    return pvalue, psi1, TensSize,m


def psi2test(DataTensor,g1,g2,args):
    TensSize = DataTensor.size()[0]
    
    zeros_tensor = torch.zeros((DataTensor.size(0), 11), device=device)
    NewTensor = torch.cat((DataTensor, zeros_tensor), dim=1)

    m = args.neighbors
    
    #if m_variable:
    #    m=int((TensSize/5)**(1/3))
    #else:
    #    from param import m
        
    NewTensor,w , NN_Array =tools.neighbors(NewTensor,TensSize,m)
    NewTensor=tools.map_xy(NewTensor,TensSize,NN_Array,args)
    entxy = entropy.hm_xy(NewTensor,TensSize)
    entx = entropy.hm_x(NewTensor,TensSize)
    enty = entropy.hm_y(NewTensor,TensSize)
    psi2 = entx + enty - entxy
    
    bl = 6
    buoys=tools.buoys_finding(NewTensor)
    ordi=tools.blocks(NewTensor,buoys,TensSize,bl) ## n : / bl : number of block

    pvalue=tools.block_bootstrap_pre(NewTensor,psi2,TensSize,NN_Array,bl,ordi,args)

    return pvalue, psi2, TensSize,m
