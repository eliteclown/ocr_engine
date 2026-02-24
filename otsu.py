import numpy as np
import cv2

def otsu_thresholding_manual(image):

    #1. compute Histogram

    hist = cv2.calcHist([image], None)
    hist_norm = hist.ravel()/hist.sum()

    # 2. Compute cumulative sums (omega) and cumulative means (mu)
    # NumPy's cumsum is highly optimized for this
    Q = hist_norm.cumsum()
    bins = np.arange(256)
    fn_min = np.inf
    thresh = -1

    #Global mean (mu_T)
    mu_T = (bins * hist_norm).sum()

    sigma_b_squared_max = -1
    optimal_k =0

    #3. Iterate through all possible thresholds
    for k in range(256):
        omega_0 = Q[k]
        omega_1 = 1 - omega_0
        
        if omega_0 == 0 or omega_1==0:
            continue

        #mu(k)
        mu_k= (bins[:k+1] * hist_norm[:k+1]).sum()

        # Mean of class 0 and class 1
        mu_0 = mu_k/omega_0
        mu_1 = (mu_T - mu_k)/omega_1

        # Inter-class variance: sigma_B^2 = w0 * w1 * (mu0 - mu1)^2
        sigma_b_squared = omega_0*omega_1*((mu_0-mu_1)**2)

        if sigma_b_squared > sigma_b_squared_max:
            sigma_b_squared_max = sigma_b_squared
            optimal_k = k
    

    #4. Apply Threshold
    binary_img = image.copy()
    binary_img[binary_img > optimal_k] = 255
    binary_img[binary_img <= optimal_k] = 0

    return optimal_k , binary_img

