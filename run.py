# ------------------------------------------------------------------------------
# Title:        Excecuting Script
# Description:  Running causality code starts from here. 
# Author:       Seulgi Lee , Sanaz Panahandeh
# Date:         2024-11-07
# Version:      2025-03-10
# ------------------------------------------------------------------------------
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import argparse
import tools,entropy,output,calc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from sklearn.neighbors import NearestNeighbors
import multiprocessing
import time
import cudf
import torch

def running(query_data,output_data,args):
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## read a input file
    if query_data.endswith('csv'):
        df = cudf.read_csv(query_data)
        try:
            if 'barcode' in df.columns.tolist():
                df = df.drop(columns=['barcode'])
            df = df.drop(columns=['Unnamed: 0'])
        except:
            pass
        
        ## get x(cell type or gene) , y (cell type or gene) information.
        cols = df.columns.tolist()
        DataTensor = torch.tensor(df.to_numpy(), dtype=torch.float32,device = device)
        ColSize = DataTensor.size()[1]
        
        ## get coordinates
        CoordTense = DataTensor[:,[0,1]]
        GPoint = args.gene_col
        TypeLevel = args.Type_Level
        XSize = ColSize
        YSize = ColSize
        
        PsiTests = args.pretest
        
            
        if TypeLevel in ["g", "c"]:
            XSize = ColSize if TypeLevel == "g" else GPoint
            YSize = GPoint if TypeLevel == "g" else XSize
        else:
            raise ValueError("Insert correct TypeLevel ('g' or 'c').")
        
        ## need to set by length of x,y 
        for cell_type in range(2, XSize):
            CellID = cols[cell_type]

            for g1 in range(YSize,ColSize):
                    GeneID = cols[g1]
                    if CellID==GeneID:
                        continue
                    g_start_time = time.time()
                    
                    ## setted Cuz be x, Res be y.
                    Cuz = DataTensor[:, cell_type].unsqueeze(1)
                    Res = DataTensor[:, g1].unsqueeze(1)
                    
                    ## making tensor
                    PairTens = torch.cat((CoordTense, Cuz, Res), dim=1)

                    PsiCut = False

                    for p in PsiTests:
                        if p ==1:
                            pvalue, psi1,n,m=calc.psi1test(PairTens,CellID,GeneID,args)
                            output.Psitest(output_data+"_psi1-test.csv",CellID,GeneID,psi1,n,m,pvalue)
                            if pvalue >= 0.05:
                                PsiCut = True
                                break
                            #output.Psitest(output_data+"_psi1-test.csv",CellID,GeneID,psi1,n,m,pvalue)
                        elif p==2 : 
                            pvalue, psi2,n,m=calc.psi2test(PairTens,CellID,GeneID,args)
                            output.Psitest(output_data+"_psi2-test.csv",CellID,GeneID,psi2,n,m,pvalue)
                            if pvalue >= 0.05:
                                PsiCut = True
                                break
                            #output.Psitest(output_data+"_psi2-test.csv",CellID,GeneID,psi2,n,m,pvalue)
                    if PsiCut == True: 
                        print(CellID, " - " ,GeneID, 'pretest is not passed')
                        continue ## if psi test is not passed, skip the causality test.
                            
                    pvalue,YtoX_pvalue,delh,YtoX,n,m = calc.causation(PairTens,CellID,GeneID,args)
                    g_end_time = time.time()
                    print(CellID, " - " ,GeneID, " : gene_pair time ",g_end_time-g_start_time)
                    cuz = 0
                        
                    ## check causality between x and y.
                    if pvalue < 0.05 and YtoX_pvalue >=0.05:
                        cuz = 1

                    output.pvalue(output_data,CellID,GeneID,delh,n,m,pvalue,YtoX_pvalue,cuz)

    end_time = time.time()
    print("total time: ",end_time-start_time)
    
def argv(query,ondir,args):
    Start_time = time.time()
    
    output_suffix = args.suffix
    
    
    if not os.path.exists(ondir):
        os.makedirs(ondir)
        
    if os.path.isdir(query):
        for f in os.listdir(query):
            query_data = os.path.join(query,f)
            output_data = ondir + output_suffix + query_data.split('/')[-1]
            if os.path.isfile(output_data):
                continue
            
            running(query_data,output_data,args)
            
    elif os.path.isfile(query):
        query_data = query
        output_data = ondir + output_suffix + query_data.split('/')[-1]
        running(query_data,output_data,args)
    else:
        print('Please check your input-file format.')
    
    End_time = time.time()
    total_time = End_time - Start_time
    
    print('****** CALCULATION COMPLETE ******')
    print("Total execution time: ",total_time ,"seconds")
    


def main():
    argparser = argparse.ArgumentParser(description='Spatial-TENET')
    argparser.add_argument("-f", "--file_path", help="Path to the Spatial data for input. should be csv format" ,type = str , required= True)
    argparser.add_argument("-o", "--output_path", help="Path to saved" ,type = str , required= True)
    argparser.add_argument("-l" , "--Type_Level" , help="causality between 'c' (=Cell to Gene) | 'g' (=Gene to Gene)." , type = str, required = True)
    argparser.add_argument("-g", "--gene_col" , help="start column of gene." , type=int, default=2)
    argparser.add_argument("-m", "--neighbors", help="number of neighbors. default is 6" ,type = int , default = 6)
    argparser.add_argument("-s", "--symbolizing", help="how symbolize from the raw data. 2 is median , 3 is quantile. (2 is default)" ,type = int, default=2)
    argparser.add_argument("-b", "--bootstrap", help="number of bootstrapping. defatuls is 199" ,type = int , default = 199)
    argparser.add_argument("-sf", "--suffix", help="suffix of output file" ,type = str , default = 'causality_GPU_')
    argparser.add_argument("-t", "--pretest", type=int, nargs='+',help='an integer for the accumulator',default= [1])
    args = argparser.parse_args()
    argv(args.file_path , args.output_path , args)

if __name__ == "__main__":
    main()
