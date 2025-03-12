# ------------------------------------------------------------------------------
# Title:        Output Files Management Script
# Description:  ------------
# Author:       Sanaz Panahandeh , Seulgi Lee
# Date:         2023-09-01
# Version:      2025-03-10
# ------------------------------------------------------------------------------


import pandas as pd
import csv

def pvalue(file_name,g1_name,g2_name,delh,n,m,pvalue,ytox_pval,cuz):
	with open(file_name,'a') as c: # a means append mode
            #c.write("cell_type:{},g1:{},g2:{},delh:{},n:{},m:{},medx:{},medy:{},medxw:{},medyw:{},pvalue:{},YtoX_pval:{},Causality:{}\n".format(cell_type,g1_name,g2_name,delh,n,m,medx,medy,medxw,medyw,pvalue,ytox_pval,cuz))
            c.write("{},{},{},{},{},{},{},{}\n".format(g1_name,g2_name,delh,n,m,pvalue,ytox_pval,cuz))

def Psitest(file_name,g1_name,g2_name,psi,n,m,pvalue):
	with open(file_name,'a') as c: # a means append mode
            c.write("{},{},{},{},{},{}\n".format(g1_name,g2_name,psi,n,m,pvalue))