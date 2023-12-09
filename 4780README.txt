4780 Final Project:
Luc Dube, Brycen Wright
Dec 8th, 2023

CODE:
Our modified code is integrated into the CIFAR-Zoo library, I have provided two .py files containing this code.
labelPoisoner.py and imagePoisoner.py, they are non functional individually and simply act as code storage.



Instructions:
- Clone the repo git clone https://github.com/BrWri4227/4780FinalProject.git
- Run pip install -r requirements.txt to download the required dependencies
- Use Terminal to navigate to CIFAR-Zoo Directory
- Run program by typing python -u train.py --work-path ./experiments/cifar10/lenet
- Replace 'lenet' with the architecture you wish to run.
- Modify the parameters by navigating to experiments/cifar10/lenet but lenet is the network you wish to modify
- To poison labels, navigate to utils.py in the CIFAR-ZOO root directory, find trainset.targets = poisonLabels(trainset.targets, 0.0) on line 136, modify 0.0 to the preferred % poison from 0-1, 0 being 0%, 1 being 100%
- To change epsilon values, navigate to train.py in the CIFAR-ZOO Root directory, find epsilons on line 219 and modify to your desired values.