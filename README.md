# MambaAAI
Source code and data for "Bio-inspired Mamba for antibody-antigen interaction prediction"

![Framework of MambaAAI](https://github.com/liuxuan666/MambaAAI/blob/main/p1.png)  

# Requirements
* Python >= 3.10
* PyTorch >= 2.2
* PyTorch Geometry >= 1.8
* hickle >= 3.4
* DeepChem >= 2.4
* RDkit >= 2020.09

# Usage
* First, `pretrained.py` needs to be run to obtain the pretrained features of the antigen and antibody.
* Next, the following scenarios can be tested:
* python Main_5cv.py \<parameters\>  #---Binary classification task with 5-fold CV
* python Main_indep.py \<parameters\> #---Independent testing with 9(traing):1(testing) split of the dataset
