import numpy as np

def perrycioc(startValue, endValue, location, slope, x):
    return ((startValue * np.exp(location * slope) + endValue * np.exp(slope * x))/
            (np.exp(location * slope) + np.exp(slope * x)))

copper_rho_s_sigma = (8.5e28 * -1.6e-19, -1.6e-19 / 9e-31, 8e7)