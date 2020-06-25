### On Adversarial Bias and the Robustness of Fair Machine Learning

##### Set up

- Require: python 3.6

  1. Install virtualenv: `pip install virtualenv`

  2. Create a virtual environment
     - `mkdir ~/env/ # this directory contains all virtual environment`
     - `virtualenv -p python3 ~/env/(name)` # replace (name) with your naming of the virtual environment

  3. Install packages
     - `source ~/env/(name)/bin/activate` # activate the environment
     - `pip install -r requirements.txt `

##### Datasets

- We evaluate the code on the COMPAS and Adult dataset. We generate 4 datasets in the dataset folder based on the preprocessing step which we mentioned in the paper. Each dataset includes:

- `x_train`,`y_train` clean training dataset.
- `x_noise`,`y_noise` hard examples from nature.
- `x_attacker`,` y_attacker` attacker dataset which is from the same distribution as clean training dataset but no overlap.
- `x_test`,` y_test` test dataset.

##### Run the code

- Implementation of Algorithm 1 and 2 are in attacks.py. Call the corresponding functions and provide required arguments to run the attacks.
- Follow Example notebook to see an example.
