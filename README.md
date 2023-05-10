# correlated_capacity
Code to reproduce the experiments presented in Wakhloo, Sussman, and Chung, Physical Review Letters (in press, 2023)



## Reproducing the Experiments 

First, in a new anaconda environment, install the required dependencies using

```
pip install -r requirements.txt 
```

The Gaussian cloud and sphere experiments can be reproduced by executing the `cloud-exper.py` and `ratio_simcap_sphere.py` scripts. To reproduce the experiments with the deep network, run:

```
image-net-experiment.py --imagenetpath /path/to/IMGNET/ILSVRC/Data/CLS-LOC/train --samp s
``` 

where s is an integer between 0 to 4 indicating which of the five runs should be executed. 

Jupyter notebooks reproducing the relevant figures can then be found in the figs directory. 

## DOI 

[![DOI](https://zenodo.org/badge/595802352.svg)](https://zenodo.org/badge/latestdoi/595802352)
