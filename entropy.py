# ------------------------------------------------------------------------------
# Title:        Entropy Calculation Script
# Description:  This script calculates the conditional entropy with and without the 
#               info of second variable(x) and passes the difference between them
#               as a metric for precision improvement on prediction of y by x.
# Author:       Seulgi Lee, Sanaz Panahandeh
# Date:         2024-11-07
# Version:      2025-03-10
# ------------------------------------------------------------------------------


import math
import time
import torch
import warnings


warnings.filterwarnings("ignore", category=UserWarning)


def hm(s, n):
    # ******** delh(yw;xw)=hm(y|ym)-hm(y|yw;xw)=[hm(y;yw)-hm(yw)]-[hm(y;yw;xw)-hm(yw;xw)] # ********
    # ******** hm(x;y)=- sum (p(xi,yi)*ln(p(xi,yi))) This is mixed or joint entropy of x and y # ********
    # ******** hm(x)= - sum (p(xi)*ln(p(xi))) This is marginal entropy # ********
    # ******** So conditional entropy(x|y) is: mixed entropy of x and y (x;y) minus marginal entropy of y # ********

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s = s.to(device)
    
    hyw = entropy(ClusterCounts(s[:,14]),n)
    hyyw = entropy(ClusterCounts(s[:,[12,14]]),n)
    hywxw = entropy(ClusterCounts(s[:,[14,13]]),n)
    hyywxw = entropy(ClusterCounts(s[:,[12,14,13]]),n)
    
    delh = (hyyw - hyw) - (hyywxw - hywxw)

    # this is for y to x
    
    hxw = entropy(ClusterCounts(s[:,13]),n)
    hxxw = entropy(ClusterCounts(s[:,[11,13]]),n)
    hxwyw = hywxw ## entropy(ClusterCounts(s[:,[13,14]]),n)
    hxxwyw = entropy(ClusterCounts(s[:,[11,13,14]]),n)
    

    ytox = (hxxw - hxw) - (hxxwyw - hxwyw)


    return delh, ytox


def entropy(counts, n):
    """Helper function to calculate entropy given counts and total number of samples."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    
    counts = torch.tensor(counts,dtype = torch.float32).clone().detach().float().to(device)
    probs = counts / n
    return (-torch.sum(probs * torch.log(probs)))

    # Use groupby and size to get the counts more efficiently
    
def ClusterCounts(Sub):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    uniq_rows, ClustCount = torch.unique(Sub,return_counts=True, dim=0)
        
    return ClustCount


def hm_xy(s,n): #NewTensor,TensSize
    entxy =  entropy(ClusterCounts(s[:,[12,11]]),n)
    return(entxy)

def hm_x(s,n): #NewTensor,TensSize
    entx =  entropy(ClusterCounts(s[:,[11]]),n)
    return(entx)

def hm_y(s,n): #NewTensor,TensSize
    enty =  entropy(ClusterCounts(s[:,[12]]),n)
    return(enty)


def permut(s,n,m): #NewTensor,TensSize
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s = s.to(device)
    entxy =  entropy(ClusterCounts(s[:,[12,11]]),n)
    nxyepk = entxy / n

    def log_comb(n, k):
        """Compute log of combinations using lgamma function."""
        n = torch.tensor(n, dtype=torch.float32, device=device)
        k = torch.tensor(k, dtype=torch.float32, device=device)
        return torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)

    permut_value = torch.log(torch.exp(log_comb(m - 1, entxy)))
    si = torch.sum(nxyepk * permut_value)

    
    return(si)
