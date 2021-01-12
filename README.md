# LISA Dataset for Classification

[LISA](http://cvrr.ucsd.edu/LISA/lisa-traffic-sign-dataset.html) is a dataset, mainly used for detection tasks.
It contains images of traffic signs taken from video shots of driving vehicles.
During my research, I came across [Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering](https://arxiv.org/pdf/1811.03728.pdf), 
which uses clustering as a defense against backdoor attacks. 
So in order to reproduce their results, I needed this dataset.
So I thought I should make it available for other people in community whom may require the dataset as well. 
It is mainly tailored for use using Pytorch. 


### Usage
You could simply copy and paste lisa.py into your project and then use it same way you would use CIFAR10. For instance:

```python
from lisa import LISA

dataset = LISA(root='path_to_data', download=True, train=True)

```
To reproduce the results of Activation Clustering paper, look at the `activation_clustering_example.py` code. 


### Dataset Properties
To train networks, one might need `mean` and `std` of the dataset. It's as following: 
* mean: `[0.4563, 0.4076, 0.3895]`
* std: `[0.2298, 0.2144, 0.2259]`

### Examples 
There are some figures of the images presented in the examples folder. 
Here are the 5 classes that (I guess) authors of Activation Clustering paper use for classification:

Stop signs:

![Stop Signs](/examples/png/stop.png)

Yield Signs:

![Yield Signs](/examples/png/yield.png)

Warning Signs:

![Warning Signs](/examples/png/warning.png)

Speed Limit Signs:

![Speed Limit Signs](/examples/png/speed.png)

Regulatory Signs:

![Regulatory Signs](/examples/png/regulatory.png)