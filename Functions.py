def RMSEP(pred, labels):
    # This function calculates the Root Mean Square Percentage Error
    import numpy as np
    import math 
    rmsep = 0
    for i in range(0,len(pred)):
        rmsep += ((np.exp(labels[i])-np.exp(pred[i]))/np.exp(labels[i]))**2
    rmsep = rmsep / len(pred)
    rmsep = math.sqrt(rmsep)
    return rmsep
    