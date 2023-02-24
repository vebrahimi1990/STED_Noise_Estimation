# STED_Noise_Estimation
This is a python script for estimation of Poisson noise and generating a noisy data set from a high SNR data set. This is usefule for training a supervised deep learning
network to denoise fast confocal/STED microscopy data since the dominant source of noise in those cases is Poisson. The algorithm simply compares the histogram of the intensity of a generated noisy data to a noisy data that is captured with a microscope and finds the right Poisson noise parameters that minimizes the comparison error.

# Results 
