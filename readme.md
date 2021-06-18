# **DGPN**

Source code and dataset for KDD 2021 paper: [Zero-shot Node Classification with Decomposed Graph Prototype Network] (https://arxiv.org/abs/2106.08022).

## **Dataset and experimental setting**

Three datasets (Cora, Citeseer and C-M10-M) are used in this example. 

## **Usage** 

`python main.py --dataset cora --train-val-class 3 0 --n-epochs 1000 --k 3 --beta 0.7 --alpha 1.0`

`python main.py --dataset citeseer --train-val-class 2 0 --n-epochs 1000 --k 3 --beta 0.7 --alpha 1.0`

`python main.py --dataset C-M10-M --train-val-class 3 0 --n-epochs 500 --k 3 --beta 0.7 --alpha 0.1` 
