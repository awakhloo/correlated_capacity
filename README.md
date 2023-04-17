# correlated_capacity
Code to reproduce the experiments presented in Wakhloo, Sussman, and Chung, 2022

## Reproducing the Experiments 

First install all required dependencies using

```
pip install -r requirements.txt 
```

The Gaussian cloud and sphere experiments can be reproduced by executing the `cloud-exper.py` and `ratio_sphere_simcap.py` scripts. To reproduce the experiments with the deep network, run:

```
image-net-experiment.py --imagenetpath /path/to/IMGNET/ILSVRC/Data/CLS-LOC/train --samp s
``` 

where s is an integer between 0 to 4 indicating which of the five runs should be executed. 

Jupyter notebooks reproducing the relevant figures can then be found in the figs directory. 
