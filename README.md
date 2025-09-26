# SeqTrainer

ML package for generating tabular and graph datasets from Synthetic Biology data in SBOL, preprocesing, ML models, trainning and metrics.


## High Performance Computing (HPC) Usage

first you need to get the original datasets

navigate to https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE144621

download GSE144621_U00096.2_frag-rLP5_LB_expression.txt.gz and GSE144621_U00096.2_frag-rLP5_M9_expression.txt.gz

unzip those files and add them to the data/original_data folder

the names that we used for those files is frag-rLP5-LB_expression.txt and frag-rLP5-M9_expression.txt

navigate to the folder hpc

run 300K_preprocessing.py


