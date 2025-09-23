import numpy as np
from matplotlib import pyplot as plt

from fdtd_fun import Grid, Field
from main.util import ODR
from runTestSlab import k, runSlab, s, rho, sigma, w, T


def prediction(t):
    return np.exp(-s * rho / 2 / sigma * t) * s * rho * k / w * np.sin(w * t)


def errormodel(B, step):  # B = [A,exp]
    return np.pow(B[0] * step, B[1])

def noisemodel(B,step):
    return B[0]/step

steps = np.linspace(T / 2000, T/50, 40)
errors = []
noises = []

for stepsize in steps:
    Jzmean, t, Jzsig = [],[],[]
    def trigger(grid:Grid):
        surface = grid[:][Field.J.value].reshape(3,-1)[2]
        Jzmean.append(np.mean(surface))
        Jzsig.append(np.std(surface))
        t.append(grid.time())
    runSlab(stepsize, T,10,trigger)
    Jzmean = np.asarray(Jzmean)
    t = np.asarray(t)
    # region look at stuff
    """
    Jzsig = np.asarray(Jzsig)
    plt.title(f"Mean current at stepsize {stepsize:.2E} s")
    plt.plot(t,Jzmean)
    plt.xlabel("Time, s")
    plt.ylabel("Noise")
    plt.show()
    """
    # endregion
    noises.append(np.max(Jzsig))
    expected = prediction(t)
    chi_abs = np.abs(expected - Jzmean)
    errors.append(np.max(chi_abs))
errors = np.asarray(errors)/(s * rho * k / w)
noises = np.asarray(noises)
fit = ODR(errormodel, steps, errors, (8e18, 2))
print(f"Error power fit coefs: {fit.beta}")
plt.subplot(1,2,1)
plt.plot(steps, errors, "bx")
finesteps = np.linspace(np.min(steps)*0.99, np.max(steps), 500)
plt.plot(finesteps, errormodel(fit.beta, finesteps), "r", label = f"$(s\\cdot{fit.beta[0]:.3E})^"+"{"+f"{fit.beta[1]:.3E}"+"}$")
plt.xlabel("Stepsize, s")
plt.title("Cycle error over \n oscillation amplitude, unitless")
plt.legend()
plt.tight_layout()
plt.subplot(1,2,2)
inverses = 1/noises
inversefit = ODR(noisemodel,steps,noises,(1e25,))
print(f"Spatial noise inverse fit coefs: {inversefit.beta}")
plt.plot(steps,noises,"bx")
plt.plot(finesteps,noisemodel(inversefit.beta,finesteps), "r", label = "$\\frac{1}{" +f"{inversefit.beta[0]:.2E}"+"\\cdot s}$")
plt.title("Maximum noise over cycle, A$\\mathrm{m}^-2$")
plt.xlabel("Stepsize, s")
plt.legend()
plt.tight_layout()
plt.show()
# at higher stepsizes the simulated period time increases a lot, why?
