
### On Adversarial Bias and the Robustness of Fair Machine Learning
This repo provides the code for [our paper]([https://arxiv.org/abs/2006.08669](https://arxiv.org/abs/2006.08669)) "On Adversarial Bias and the Robustness of Fair Machine Learning".

We design and implement data poisoning attack algorithms (algorithm1 and algorithm2) against machine learning models trained with equalized odds as the fairness constraint. We assume the attacker who can control the sampling process and (in the stronger case, also) the labeling process for some of the training data. 

##### Set up

- Require: python 3.6

  1. Install virtualenv: `pip install virtualenv`

  2. Create a virtual environment
     - `mkdir ~/env/ # this directory contains all virtual environment`
     - `virtualenv -p python3 ~/env/(name)` # replace (name) with your naming of the virtual environment

  3. Install packages
     - `source ~/env/(name)/bin/activate` # activate the environment
     - `pip install -r requirements.txt `
	 
- Note: For reproducibility, we include [fairlearn repo]([https://github.com/fairlearn/fairlearn](https://github.com/fairlearn/fairlearn)) (Microsoft, v0.3.0).

##### Datasets

- We evaluate the code on the COMPAS and Adult dataset. We generate 4 datasets in the dataset folder based on the preprocessing step which we mentioned in the paper. Each dataset includes:

- `x_train`,`y_train` clean training dataset.
- `x_noise`,`y_noise` hard examples from nature.
- `x_attacker`,` y_attacker` attacker dataset which is from the same distribution as clean training dataset but no overlap.
- `x_test`,` y_test` test dataset.

##### Run the code

- Implementation of Algorithm 1 and 2 are in attacks.py. Call the corresponding functions and provide required arguments to run the attacks.
- Follow Example notebook to see an example.

##### Citation
To cite the arxiv version, please use the following bibtex
```
@article{chang2020adversarial,
  title={On Adversarial Bias and the Robustness of Fair Machine Learning},
  author={Chang, Hongyan and Nguyen, Ta Duy and Murakonda, Sasi Kumar and Kazemi, Ehsan and Shokri, Reza},
  journal={arXiv preprint arXiv:2006.08669},
  year={2020}
}
```