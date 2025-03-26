# ------------------------------------------------------------------------------
# Title:        Tools Box Script
# Description:  This script contains tools for calculating different things.
# Author:       Seulgi Lee , Sanaz Panahandeh
# Date:         2024-11-07
# Version:      2025-03-10
# ------------------------------------------------------------------------------

import entropy, output
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
import multiprocessing
import time
from scipy.sparse import csr_matrix
import torch
import os
#import faiss
# Set the environment variable
os.environ['TORCH_USE_CUDA_DSA'] = '1'


def weightm(nnlist,n,m):
    # Create sparse weights matrix
    nnlist = torch.tensor(nnlist)
    
    data = torch.ones(n * (m - 1)) / (m - 1)
    row_indices = torch.repeat_interleave(torch.arange(n), m - 1)
    col_indices = nnlist[:, 1:].flatten()
    indices = torch.stack([row_indices, col_indices])
    w = torch.sparse_coo_tensor(indices, data, size=(n, n)).cuda()

    return w

def neighbors(DataTensor,TensSize,m):

    coord = DataTensor[:,[0,1]].to('cpu').numpy()

    nbrs = NearestNeighbors(n_neighbors=m, algorithm='ball_tree').fit(coord)
    dist, nnlist = nbrs.kneighbors(coord)
    w=weightm(nnlist,TensSize,m)
    
    NNTensor = torch.tensor([arr.tolist() for arr in nnlist[:, 1:]])

   
    return DataTensor, w , NNTensor

def lag(DataTensor,w):

    DataTensor[:,4] = w @ DataTensor[:,2]
    DataTensor[:,5] = w @ DataTensor[:,3]
    
    return DataTensor

def map(DataTensor, n,NN_Array,args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    # Convert Pandas columns to PyTorch tensors and move to GPU
    x = DataTensor[:,2].to(device)
    y = DataTensor[:,3].to(device)
    xw = DataTensor[:,4].to(device)
    yw = DataTensor[:,5].to(device)
    nn = NN_Array.to(device)
    
    bins = args.symbolizing

    if bins == 3:
        # Quantile calculation using PyTorch
        q25_x = torch.quantile(x, 0.25)
        q75_x = torch.quantile(x, 0.75)
        q25_y = torch.quantile(y, 0.25)
        q75_y = torch.quantile(y, 0.75)
        q25_xw = torch.quantile(xw, 0.25)
        q75_xw = torch.quantile(xw, 0.75)
        q25_yw = torch.quantile(yw, 0.25)
        q75_yw = torch.quantile(yw, 0.75)

        # Vectorized operations
        tx = torch.where(x <= q25_x, 0, torch.where(x <= q75_x, 1, 2)).to(device)
        ty = torch.where(y <= q25_y, 0, torch.where(y <= q75_y, 1, 2)).to(device)
        txw = torch.where(xw <= q25_xw, 0, torch.where(xw <= q75_xw, 1, 2)).to(device)
        tyw = torch.where(yw <= q25_yw, 0, torch.where(yw <= q75_yw, 1, 2)).to(device)
        
    elif bins == 2:
        # Median calculation using PyTorch
        med_x = torch.median(x)
        med_y = torch.median(y)
        med_xw = torch.median(xw)
        med_yw = torch.median(yw)
        

        # Vectorized operations
        tx = (x > med_x).to(device).long()
        ty = (y > med_y).to(device).long()
        txw = (xw > med_xw).to(device).long()
        tyw = (yw > med_yw).to(device).long()
        
        
    elif bins ==1:
        # not Vectorized operations
        tx = x 
        ty = y 
        txw = xw 
        tyw = yw 
        
        
    elif bins == 4: 
        # Median calculation using PyTorch
        med_x = torch.median(x)
        med_y = torch.median(y)
        med_xw = torch.median(xw)
        med_yw = torch.median(yw)
        
        # Vectorized operations
        
        tx = torch.where(x == 0, torch.tensor(0, device=device), torch.where(x <= med_x, torch.tensor(1, device=device), torch.tensor(2, device=device)))
        ty= torch.where(y == 0, torch.tensor(0, device=device), torch.where(y <= med_y, torch.tensor(1, device=device), torch.tensor(2, device=device)))
        txw = torch.where(xw == 0, torch.tensor(0, device=device), torch.where(xw <= med_xw, torch.tensor(1, device=device), torch.tensor(2, device=device)))
        tyw = torch.where(yw == 0, torch.tensor(0, device=device), torch.where(yw <= med_yw, torch.tensor(1, device=device), torch.tensor(2, device=device)))

    # Initialize the results arrays
    sigx = torch.zeros(n, device=device)
    sigy = torch.zeros(n, device=device)
    sigxw = torch.zeros(n, device=device)
    sigyw = torch.zeros(n, device=device)
    
    
    # 벡터화된 연산
    sigx = torch.sum(tx[nn] == tx.unsqueeze(1), dim=1)
    sigy = torch.sum(ty[nn] == ty.unsqueeze(1), dim=1)
    sigxw = torch.sum(txw[nn] == txw.unsqueeze(1), dim=1)
    sigyw = torch.sum(tyw[nn] == tyw.unsqueeze(1), dim=1)
    
    DataTensor[:,7] = tx
    DataTensor[:,8] = ty
    DataTensor[:,9] = txw
    DataTensor[:,10] = tyw

    DataTensor[:,11] = sigx
    DataTensor[:,12] = sigy
    DataTensor[:,13] = sigxw
    DataTensor[:,14] = sigyw
    
    
    return DataTensor

def map_xy(DataTensor, n,NN_Array,args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    # Convert Pandas columns to PyTorch tensors and move to GPU
    x = DataTensor[:,2].to(device)
    y = DataTensor[:,3].to(device)
    nn = NN_Array.to(device)
    
    bins = args.symbolizing

    if bins == 3:
        # Quantile calculation using PyTorch
        q25_x = torch.quantile(x, 0.25)
        q75_x = torch.quantile(x, 0.75)
        q25_y = torch.quantile(y, 0.25)
        q75_y = torch.quantile(y, 0.75)
        
        # Vectorized operations
        tx = torch.where(x <= q25_x, 0, torch.where(x <= q75_x, 1, 2)).to(device)
        ty = torch.where(y <= q25_y, 0, torch.where(y <= q75_y, 1, 2)).to(device)
        
        
    elif bins == 2:
        # Median calculation using PyTorch
        med_x = torch.median(x)
        med_y = torch.median(y)
    
        # Vectorized operations
        tx = (x > med_x).to(device).long()
        ty = (y > med_y).to(device).long()
    
        
    elif bins ==1:
        # not Vectorized operations
        tx = x 
        ty = y 
        
    elif bins == 4: 
        # Median calculation using PyTorch
        med_x = torch.median(x)
        med_y = torch.median(y)
        
        # Vectorized operations
        
        tx = torch.where(x == 0, torch.tensor(0, device=device), torch.where(x <= med_x, torch.tensor(1, device=device), torch.tensor(2, device=device)))
        ty= torch.where(y == 0, torch.tensor(0, device=device), torch.where(y <= med_y, torch.tensor(1, device=device), torch.tensor(2, device=device)))
       
    # Initialize the results arrays
    sigx = torch.zeros(n, device=device)
    sigy = torch.zeros(n, device=device)
    
    # 벡터화된 연산
    sigx = torch.sum(tx[nn] == tx.unsqueeze(1), dim=1)
    sigy = torch.sum(ty[nn] == ty.unsqueeze(1), dim=1)
    
    DataTensor[:,7] = tx
    DataTensor[:,8] = ty

    DataTensor[:,11] = sigx
    DataTensor[:,12] = sigy
    
    return DataTensor

def buoys_finding(DataTensor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    xc = DataTensor[:,0].to(device)
    yc = DataTensor[:,1].to(device)

    # calculation min, max of each axis 
    xmin = torch.min(xc)
    xmax = torch.max(xc)
    ymin = torch.min(yc)
    ymax = torch.max(yc)

    # buoys
    buoys = torch.tensor([[xmin, ymin],
                        [xmin + ((xmax - xmin) / 2), ymin],
                        [xmax, ymin],
                        [xmin, ymax],
                        [xmin + ((xmax - xmin) / 2), ymax],
                        [xmax, ymax]], device='cuda')
    return(buoys)
        
def blocks(DataTensor,buoys,n,bl):
    nbl = int(n / bl)

    # generate tensor for containing each points
    points = DataTensor[:,[0,1]]
    indices = torch.arange(n, device='cuda').reshape(-1, 1)

    # ord :: bl X nbl size of new_tensor 
    ordi = torch.empty((nbl, bl), dtype=torch.long, device='cuda')

    # calc KNN
    knn = NearestNeighbors(n_neighbors=nbl)
    for i in range(bl):
            knn.fit(points.cpu().numpy())  
            distances, nearest_indices = knn.kneighbors([buoys[i].cpu().numpy()])
            
            distances = torch.tensor(distances).to('cuda')
            nearest_indices = torch.tensor(nearest_indices).to('cuda')
            # sorting indices by distance
            sorted_indices = nearest_indices[0, torch.argsort(distances[0])]
    
            # get indices  : get index of sorted_indices
            occupied_indices = indices[sorted_indices].squeeze()
         
            # updating ord_tensor
            ordi[:,i] = occupied_indices
        
            # updating points
            mask = ~torch.isin(indices, occupied_indices).squeeze()
            points = points[mask]
            indices = indices[mask]
    return(ordi)


def bootstrap(DataTensor_B,w,delh,YtoX,n,NN_Array,args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ## sampling x,y
    DataSamp = DataTensor_B[:,[2,3]]
    
    ## xc,yc but except nn. (we're gonna deal 'nn' as seperated array(NN_Array)) 
    DataCop = DataTensor_B[:,[0,1]]
    
    
    zeros_tensor = torch.zeros(DataTensor_B.size()[0], 13, device=device)  
    NewTensor_B = torch.cat((DataCop,zeros_tensor),dim=1)
    
    B = args.bootstrap
    
    bi=1
    t=0
    Yt=0
    while bi<B+1:

        # random indexes
        indices_x = torch.randint(0, n, (n,),  device = device)
        indices_y = torch.randint(0, n, (n,),  device = device)

        NewTensor_B[:,2] = DataSamp[:,0][indices_x]
        NewTensor_B[:,3] = DataSamp[:,1][indices_y]
     
        NewTensor_B=lag(NewTensor_B,w)

        NewTensor_B=map(NewTensor_B,n,NN_Array,args)

        delhb,YtoXb=entropy.hm(NewTensor_B,n)

        if delhb>delh: t+=1
        if YtoXb>YtoX: Yt+=1
        bi+=1

    
    pvalue=t/B
    YtoX_pvalue=Yt/B
    return pvalue ,YtoX_pvalue



def bootstrap_xy(DataTensor_B,delh,n,NN_Array,args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ## sampling x,y
    DataSamp = DataTensor_B[:,[2,3]]
    
    ## xc,yc but except nn. (we're gonna deal 'nn' as seperated array(NN_Array)) 
    DataCop = DataTensor_B[:,[0,1]]
    
    zeros_tensor = torch.zeros(DataTensor_B.size()[0], 13, device=device)  
    NewTensor_B = torch.cat((DataCop,zeros_tensor),dim=1)
    
    B = args.bootstrap
    m = args.neighbors
    
    bi=1
    t=0
    Yt=0
    while bi<B+1:
        

        # permutation indexes (not random)
        indices_x = torch.randperm(n, device=device) 
        indices_y = torch.randperm(n, device=device)
        
        # random indexes
        #indices_x = torch.randint(0, n, (n,),  device = device)
        #indices_y = torch.randint(0, n, (n,),  device = device)

        NewTensor_B[:,2] = DataSamp[:,0][indices_x]
        NewTensor_B[:,3] = DataSamp[:,1][indices_y]
     

        NewTensor_B=map_xy(NewTensor_B,n,NN_Array,args)
        entxy = entropy.hm_xy(NewTensor_B,n)
        si = entropy.permut(NewTensor_B,n,m)
        psi1b=2*(m-1)*torch.log(torch.tensor(2.0, device=device))-si-entxy
        

        if psi1b>delh: t+=1
        bi+=1
    
    pvalue=t/B

    return pvalue 


def block_bootstrap_pre(DataTensor_B,delh,n,NN_Array,bl,ordi,args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ## sampling x,y
    DataSamp = DataTensor_B[:,[2,3]]
    
    ## xc,yc but except nn. (we're gonna deal 'nn' as seperated array(NN_Array)) 
    DataCop = DataTensor_B[:,[0,1]]
    
    
    zeros_tensor = torch.zeros(DataTensor_B.size()[0], 13, device=device)  
    NewTensor_B = torch.cat((DataCop,zeros_tensor),dim=1)
    
    B = args.bootstrap
    buoys_n = [i for i in range(bl)]
    
    bi=1
    t=0

    while bi<B+1:
        
        # selecting buoys randomly and true of replacement 
        indices_x = torch.randint(0, ordi.shape[1] , (ordi.shape[0],)) ## bl = ordi.shape[1]
        indices_y = torch.randint(0, ordi.shape[1], (ordi.shape[0],))
              
        for j in range(bl):
            # selecting row of ord by indexing 
            from_rows_x = ordi[torch.arange(ordi.shape[0]),indices_x]
            from_rows_y = ordi[torch.arange(ordi.shape[0]),indices_y]
            to_rows = ordi[:,j]

            NewTensor_B[to_rows, 2] = DataSamp[from_rows_x, 0]
            NewTensor_B[to_rows, 3] = DataSamp[from_rows_y, 1]
  
        
        NewTensor_B=map_xy(NewTensor_B,n,NN_Array,args)
        entxy = entropy.hm_xy(NewTensor_B,n)
        entx = entropy.hm_x(NewTensor_B,n)
        enty = entropy.hm_y(NewTensor_B,n)
        psi2b = entx + enty - entxy
        

        if psi2b>delh: t+=1    
        bi+=1

    pvalue=t/B

    return pvalue 


def block_bootstrap(DataTensor_B,w,delh,YtoX,n,NN_Array,bl,ordi,args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ## sampling x,y
    DataSamp = DataTensor_B[:,[2,3]]
    
    ## xc,yc but except nn. (we're gonna deal 'nn' as seperated array(NN_Array)) 
    DataCop = DataTensor_B[:,[0,1]]
    
    
    zeros_tensor = torch.zeros(DataTensor_B.size()[0], 13, device=device)  
    NewTensor_B = torch.cat((DataCop,zeros_tensor),dim=1)
    
    B = args.bootstrap
    buoys_n = [i for i in range(bl)]
    
    bi=1
    t=0
    Yt=0
    while bi<B+1:
        
        # selecting buoys randomly and true of replacement 
        indices_x = torch.randint(0, ordi.shape[1] , (ordi.shape[0],)) ## bl = ordi.shape[1]
        indices_y = torch.randint(0, ordi.shape[1], (ordi.shape[0],))
              
        for j in range(bl):
            # selecting row of ord by indexing 
            from_rows_x = ordi[torch.arange(ordi.shape[0]),indices_x]
            from_rows_y = ordi[torch.arange(ordi.shape[0]),indices_y]
            to_rows = ordi[:,j]

            NewTensor_B[to_rows, 2] = DataSamp[from_rows_x, 0]
            NewTensor_B[to_rows, 3] = DataSamp[from_rows_y, 1]
            
     
        NewTensor_B=lag(NewTensor_B,w)

        NewTensor_B=map(NewTensor_B,n,NN_Array,args)

        delhb,YtoXb=entropy.hm(NewTensor_B,n)

        if delhb>delh: t+=1
        if YtoXb>YtoX: Yt+=1
        
        
        bi+=1

    pvalue=t/B
    YtoX_pvalue=Yt/B
    return pvalue ,YtoX_pvalue

