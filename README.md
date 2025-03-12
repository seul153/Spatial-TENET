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
python run.py -f FILE_PATH -o OUTPUT_PATH [-m NEIGHBORS] [-s SYMBOLIZING] [-b BOOTSTRAP] [-sf SUFFIX] [-t --pretest]
```

### Configuring Arguments and Parameters

| Short Option | Long Option               | Description                                                          | Default        |
|--------------|---------------------------|----------------------------------------------------------------------|----------------|
| `-f`         | `--file_path FILE_PATH`   | Path to the spatial data for input (CSV format).                     | Required       |
| `-o`         | `--output_path OUTPUT_PATH`| Path to save the output.                                             | Required       |
| `-l`         | `--Type_Level TYPE_LEVEL` | Specify causality: `'c'` (Cell to Gene) or `'g'` (Gene to Gene).     | Required       |
| `-g`         | `--gene_col GENE_COL`     | Start column index of gene.                                          | `2`            |
| `-m`         | `--neighbors NEIGHBORS`   | Number of neighbors to consider.                                     | `6`            |
| `-s`         | `--symbolizing SYMBOLIZING`| How to symbolize raw data: `1` (median), `2` (quantile).             | `1`            |
| `-b`         | `--bootstrap BOOTSTRAP`   | Number of bootstrapping iterations.                                  | `199`          |
| `-sf`        | `--suffix SUFFIX`         | Suffix for output files.                                             | `'causality'`  |
| `-t`         | `--pretest`               | preliminirary test.(2 step. if you want to run all steps. insert 1 2)| `1`            |

---

### Notes:
- All required arguments must be specified when executing the script.
- The default values for optional parameters can be overridden as needed.