# Decentralized Mean Field Games 

Implementation of DMFG-QL and DMFG-AC in the AAAI-2022 paper - [Decentralized Mean Field Games](https://arxiv.org/pdf/2112.09099.pdf). 


Our first set of environments pertain to simulated domains using the Petting Zoo simulator. The second set of experiments contains a real-world ride-sharing environment. 

 
 
## Code structure

- See folder battle for training and testing scripts of the Battle environment. 

- See folder combinedarms for training and testing scripts of the Combined Arms environment. 

- See folder gather for training and testing scripts of the Gather environment. 

- See folder tiger for training and testing scripts of the Tiger-Deer environment. 

- See folder for waterworld for training and testing scripts of the Waterworld environment. 

- See folder for ridesharing for training and testing scripts of the Ride sharing environment. 


### In each of directories, the files most relevant to our research are:

- /battle/pettingzoomagentDQNdecentralizedparallel.py: This file contains the training script for IL in the battle game. Similarly change the name of the algorithm to see the training of the 5 different algorithms.

 
- /battle/pettingzoomagentexecutionDMFGQLvsIL.py: This file contains the execution script for the battle between DMFG-QL and IL. Change the name of the algorithms to see the execution experiments between different algorithms. 

Similarly the training and execution scripts for all the different environments can be found in the respective folders. 


- /battle/RL_brainDQN.py: This file contains the IL implementation. 

- /battle/RL_braindmfgql.py: This file contains the DMFG-QL implementation.

- /battle/RL_dmfgac.py: This file contains the DMFG-AC implementation.

- /battle/RL_brainMFQ.py: This file contains the MFQ implementation.

- /battle/RL_actorcritic.py: This file contains the MFAC implementation.


These algorithmic files are almost the same across the different environmental domains. 



- /ridesharing/src2 - This folder contains the files for running DMFG-QL in the ride sharing environment. In this folder the file ValueFunction.py contains the algorithmic details. 

- /ridesharing/src - This folder contains the files for running NeurADP and CO in the ride sharing environment. In this folder the file ValueFunction.py contains the algorithmic details.

In addition, we have a set of python files in the main directory pertaining to modifications of the environmental domains as compared to the defaults given in the Petting Zoo environments. 


## Installation Instructions of synthetic (Petting Zoo) environments for Ubuntu 18.04


### Requirements


- `python==3.7.1`

```shell
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.7
```



- `Gym environment (0.18.0)`

```shell
pip install gym==0.18.0
```

- `MAgent environment`

```shell
pip install magent
```

Replace the gridworld.py file in the magent installation with the file provided in this repository. 


- `Petting zoo`

Please do not install petting zoo from source. Our implementation contains some changes from the souce implementation. The important files that have been changed are provided in the root directory for simplicity and easy reference. 


```shell
cd pettingzoo
pip install ./ 
```




- `tensorflow 2`

```shell
pip install --upgrade pip
pip install tensorflow
``` 


## Installation Instructions of ride sharing environment for Ubuntu 18.04


### Requirements


- `python==3.6.0`

```shell
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.6
```



- Tensorflow 1.12

```shell
pip install tensorflow==1.12
```

- Keras 2.2.4

```shell
pip install keras==2.2.4
```

- CPLEX 12.8 (Note: Commercial license is required): Follow the instructions [here](https://www.ibm.com/docs/pl/icos/20.1.0?topic=cplex-setting-up-gnulinuxmacos)  



## Synthetic (Petting Zoo) environments

- Training

```shell
cd battle
python pettingzoomagentDQNdecentralizedparallel.py 
```

This will run the training for the battle domain with the IL algorithm. Similarly run the other files for training other algorithms. For training other environments, change to the respective folders. 

- Testing

```shell
python pettingzoomagentexecutionDMFGQLvsIL.py      
```

The above command is for running the test battles between the DFMG-QL and IL algorithms. Similarly run the respective execution script for the battles. Make sure all the training is conducted first and the training files are stored in a suitable folder. Include the saved file information while running the respective scripts. 

## Ride Sharing environment 

- For running CO or NeurADP algorithms. 

```shell
cd ridesharing/src
python main.py 
```

- For running DMFG-QL algorithm 

```shell
cd ridesharing/src2
python main.py 
```

## Obtaining data for ride sharing experiments 

Due to copyright and size restrictions, the processed dataset used for the ride sharing experiments cannot be released here. Please email ***s2ganapa@uwaterloo.ca*** for obtaining this data. 


## Random Seeds

We use a set of 30 random seeds (1 -- 30) for all training experiments and a new set of 30 random seeds (31 -- 60) for all execution experiments. 


## Note

This is research code and will not be actively maintained. Please send an email to ***s2ganapa@uwaterloo.ca*** for questions or comments. 


## Code citations 

We would like to cite the [MAgent](https://github.com/geek-ai/MAgent) for the MAgent simulator on which most of our virtual experiments are based. We are using the modified version of the games from the [Pettingzoo](https://www.pettingzoo.ml/#) environment. We would also like to cite [Reinforcement Learning and Tensorflow](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow) from which we framed the structure of our algorithms and obtained the required baselines. Importantly, we would like to acknowledge [Sanket Shah](https://github.com/sanketkshah/NeurADP-for-Ride-Pooling) for data used in the ride sharing experiments. Our algorithmic implementations for the ride sharing experiments are based on the architectures and code base provided by Sanket Shah.   



## Paper citation

If you found this helpful, please cite the following paper:

<pre>



@InProceedings{Sriramdmfg2022,
  title = 	 {Decentralized Mean Field Games},
  author = 	 {Ganapathi Subramanian, Sriram and Taylor, Matthew and Crowley, Mark and Poupart, Pascal} 
  booktitle = 	 {Proceedings of the Association for the Advancement of Artificial Intelligence (AAAI-2022)},
  year = 	 {2022},
  editor = 	 {Katia Sycara},
  address = 	 {Vancouver, Canada},
  month = 	 {Feb 22 -- 1 March},
  publisher = 	 {AAAI press}
}
</pre>


