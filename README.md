# Spatial-TENET
A tool for effectively detecting dependencies between two processes in spatial data while preserving spatial patterns and structure.

## Dependency
** This tool relies on GPU computation, and therefore requires an environment with CUDA as well as compatible versions of PyTorch and cuDF.
```
  cuda 
  torch
  cudf <https://github.com/rapidsai/cudf>
  matplotlib
```

## 1. Run Spa-TENET using expression data in a csv file
* **1-2nd column** contains x,y **coordinates**.   
* **from 3nd columns** contain **gene expression** data.

#### usage
```
python run.py -f FILE_PATH -o OUTPUT_PATH [-m NEIGHBORS] [-s SYMBOLIZING] [-b BOOTSTRAP] [-sf SUFFIX]
```

* options:   
&nbsp; -h, --help &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; show this help message and exit<br>
&nbsp; -f FILE_PATH, --file_path FILE_PATH<br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Path to the Spatial data for input. should be csv format <br>
&nbsp; -o OUTPUT_PATH, --output_path OUTPUT_PATH<br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Path to saved<br>
&nbsp; -m NEIGHBORS, --neighbors NEIGHBORS<br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; number of neighbors. default is 6<br>
&nbsp; -s SYMBOLIZING, --symbolizing SYMBOLIZING<br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; how symbolize from the raw data. 1 is median, 2 is quantile. (1 is default)<br>
&nbsp; -b BOOTSTRAP, --bootstrap BOOTSTRAP<br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; number of bootstrapping. defaults is 199<br>
&nbsp; -sf SUFFIX, --suffix SUFFIX<br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; suffix of output file
