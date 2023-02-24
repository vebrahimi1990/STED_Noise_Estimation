import numpy as np
from tifffile import imread, imwrite
import matplotlib.pyplot as plt
from config import CFG

path_gt = CFG['highSNR_dr']
path_noisy = CFG['noisy_dr']
n_bins = CFG['num_bins']
range_hist = CFG['hist_range']
iteration = CFG['iterations']
lp_range = sorted(CFG['lamb_poisson_range'])
lp = np.linspace(lp_range[0], lp_range[1], CFG['num_lamb'])

gt = imread(path_gt).astype(np.float64)
gt = gt / gt.max()
noisy = imread(path_noisy).astype(np.float64)
noisy = noisy / noisy.max()
h_noisy = np.histogram(noisy, bins=n_bins, range=range_hist)[0]
h_noisy = h_noisy / h_noisy.max()

mlp = 0
for q in range(iteration):
    error = np.zeros(lp.shape)
    for i, l in enumerate(lp):
        image = np.random.poisson(gt / l, size=gt.shape)
        image = image / image.max()
        h_image = np.histogram(image, bins=n_bins, range=range_hist)[0]
        h_image = h_image / h_image.max()
        err = (abs(h_image - h_noisy)) ** 0.2
        error[i] = np.mean(err)
    error = error / error.max()
    arg = np.argmin(error)
    ind = np.unravel_index(arg, error.shape)
    mlp = lp[ind[0]] + mlp
mlp = mlp / iteration

print('lamb_poisson=', mlp)
gen_noisy = np.random.poisson(gt / mlp, gt.shape)
gen_noisy = gen_noisy / gen_noisy.max()
gen_noisy = np.uint16(gen_noisy * (2 ** 16 - 1))

# imwrite(r'D:\Projects\STED_Noise_Estimation\image_files\generated_noisy.tif',gen_noisy)

fig = plt.figure(figsize=(15, 10))
fig.add_subplot(1, 3, 1)
plt.imshow(noisy, cmap='magma')
plt.axis('off')
plt.title('Noisy data')
fig.add_subplot(1, 3, 2)
plt.imshow(gt, cmap='magma')
plt.axis('off')
plt.title('High SNR data')
fig.add_subplot(1, 3, 3)
plt.imshow(gen_noisy, cmap='magma')
plt.axis('off')
plt.title('Generated noisy data')
plt.show()
