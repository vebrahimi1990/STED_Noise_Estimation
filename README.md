# STED_Noise_Estimation
This is a python script for estimation of Poisson noise and generating a noisy data set from a high SNR data set. This is usefule for training a supervised deep learning
network to denoise fast confocal/STED microscopy data since the dominant source of noise in those cases is Poisson. The algorithm simply compares the histogram of the intensity of a generated noisy data to a noisy data that is captured with a microscope and finds the right Poisson noise parameters that minimizes the comparison error.

# Requirements
```pip install -r requirements.txt```

# Installation
```!git clone https://github.com/vebrahimi1990/STED_Noise_Estimation.git``` 

Import all the necessary parameters and direcotries into the ```config.py``` file.

```python poisson_noise_estimator.py```

# Results
![plot](https://github.com/vebrahimi1990/STED_Noise_Estimation/blob/master/image_files/Results.png)

# Contact
Should you have any question, please contact vebrahimi1369@gmail.com. 

