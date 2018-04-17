#!/usr/bin/env python
import numpy as np

def wMSE(w, y_true, y_pred):
    return np.sum( w*(y_pred-y_true)**2 )
