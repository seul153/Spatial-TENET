# ------------------------------------------------------------------------------
# Title:        Causality Calculation Script
# Description:  This is the main file that calls causality and psi1 and psi2 tests functions and bootstrap.
# Author:       Sanaz Panahandeh , Seulgi Lee
# Date:         2023-09-01
# Version:      2024-11.07
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
    delh,YtoX=entropy.hm(NewTensor,TensSize)
    pvalue,YtoX_pvalue=tools.bootstrap(NewTensor,w,delh,YtoX,TensSize,NN_Array,args)

    return pvalue,YtoX_pvalue, delh,YtoX,TensSize,m

