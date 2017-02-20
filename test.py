from exp import Exp,Log
import numpy as np

def Test_Exp():
    Error = np.array([ abs((Exp(x)-np.exp(x))/np.exp(x)) for x in np.linspace(-709,709,100000) ])
    if max(Error)<1e-12 and Exp(-800) == 0.0 and Exp(800) == np.Inf :
        print("All test passed!")
    else:
        print("Test Failed")
Test_Exp()

def Test_Log():
    Error = np.array([ abs((Log(x)-np.log(x))/np.log(x)) for x in np.linspace(1e-10,1e20,100000) ])
    if max(Error)<1e-12 and Log(0.0) == -np.Inf and Log(1e600) == np.Inf :
        print("All test passed!")
    else:
        print("Test Failed")
Test_Log()
