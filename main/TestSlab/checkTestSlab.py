import numpy as np
from matplotlib import pyplot as plt

from fdtd_fun import Grid, Field
from main.util import ODR
from runTestSlab import k, runSlab, s, rho, sigma, w, T


def prediction(t):
    return np.exp(-s * rho / 2 / sigma * t) * s * rho * k / w * np.sin(w * t)


def errormodel(B, step):  # B = [A,exp]
    return np.pow(B[0] * step, B[1])


steps = np.linspace(T / 2000, T / 40, 10)
errors = []


for stepsize in steps:
    Jz, t = [],[]
    def trigger(grid:Grid):
        Jz.append(grid[0,0,0][Field.J.value].reshape(3)[2])
        t.append(grid.time())
    runSlab(stepsize, T,10,trigger)
    Jz = np.asarray(Jz)
    t = np.asarray(t)
    expected = prediction(t)
    chi_abs = np.abs(expected - Jz)
    errors.append(np.max(chi_abs))
errors = np.asarray(errors)
fit = ODR(errormodel, steps, errors, (8e18, 2))
print(f"Fit coefs: {fit.beta}")
plt.plot(steps, errors, "bx")
finesteps = np.linspace(np.min(steps), np.max(steps), 500)
plt.plot(finesteps, errormodel(fit.beta, finesteps), "r", label = f"$(s\\cdot{fit.beta[0]:.3E})^"+"{"+f"{fit.beta[1]:.3E}"+"}$")
plt.xlabel("Stepsize, s")
plt.title("Cycle error")
plt.legend()
plt.tight_layout()
plt.show()
