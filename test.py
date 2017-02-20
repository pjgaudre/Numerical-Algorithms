from exp import Exp
import numpy as np
def Test_Exp():
    Error = np.array([ abs((Exp(x)-np.exp(x))/np.exp(x)) for x in np.linspace(-709,709,100000) ])
    if max(Error)<1e-12 and Exp(-800) == 0.0 and Exp(800) == np.Inf :
        print("All test passed!")
    else:
        print("Test Failed")
Test_Exp()
