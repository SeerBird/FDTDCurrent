from matplotlib import pyplot as plt

from main.TestSlab.runTestSlab import runSlab, T

plots = []
time = None
singleJz, _ = runSlab(T/200,T,1,(0,0,0))
for i in range(10):
    Jz,t = runSlab(T/200,T,i+2,(i//2,i//2,i//2))
    plt.plot(t,Jz-singleJz, label = f"{i+2}")
plt.legend()
print("It seems different between cells and between grid sizes, why?")
plt.show()
