import numpy as np

def perrycioc(startValue, endValue, location, slope, x):
    return startValue + (endValue-startValue)/(np.exp(slope*(location-x))+1)

def gaussian(mu, sigma, amp, x):
    return amp * np.exp(-0.5*(x-mu)**2/sigma**2)

#copper_rho_s_sigma = (8.5e28 * -1.6e-19, -1.6e-19 / 9e-31, 8e7)
copper_rho_s_sigma = (8.5e28 * -1.6e-19, -1.6e-19 / 9e-31, 8e0)