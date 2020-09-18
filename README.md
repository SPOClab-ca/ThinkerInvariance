# Thinker Invariance

This repository contains all of the source code used to train, test, evaluate and compare 
all of the deep learning models that make up the work from 
_Thinker Invariance: Enabling Deep Neural Networks for BCI Across More People_ written by 
Kostas and Rudzicz (under review).

To reproduce results from each of the datasets, the *./experiments* directory contains BASH
scripts that should be sufficient. Similarly, the *./analysis* directory should provide all
the tools needed to produce plots and figures.

If you would like to extend this work as is, or run the tests with different hyperparameters 
or options, run:

```python3 main.py --help```

This should provide many options for trying something new and or different.

# Adding Datasets
If you would simply like to run the exact same tests with a new dataset, adding a new file to the
directory *./datasets* and using one of the existing files as a template should be enough to
get up and running quickly.

# Rough Requirements
This was originally written, and heavily relies on using:
  * Specifically
    * python >= 3.5
    * mne ~= 20.0
    * pytorch >= 1.0
  * More generally
    * tqdm
    * numpy
    * matplotlib
    * pandas
