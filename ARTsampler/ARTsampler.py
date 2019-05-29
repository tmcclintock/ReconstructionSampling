"""
This file will get completely rewritten soon. Disregard.
"""
import numpy as np
from george import kernels, GP
from scipy.optimize import minimize

class ARTsampler(object):
    #This constructor will be removed later
    def __init__(self, mean, covariance, lnprob_function, lnprob_args):
        self.mean = mean
        self.cov = covariance
        self.lnlikes = lnlikes

        self.points = points
        self.x = self._transform_data(points)

        
        _guess = 4.5 #kernel length guess
        ndim = len(points[0])
        kernel = kernels.ExpSquaredKernel(metric=_guess, ndim=ndim)
        lnPmin = np.min(lnlikes)
        gp = george.GP(kernel, mean=lnPmin-np.fabs(lnPmin*3))
        
        gp.compute(self.x)
        def neg_ln_likelihood(p):
            gp.set_parameter_vector(p)
            return -gp.log_likelihood(lnlikes)

        def grad_neg_ln_likelihood(p):
            gp.set_parameter_vector(p)
            return -gp.grad_log_likelihood(lnlikes)

        result = minimize(neg_ln_likelihood,
                          gp.get_parameter_vector(),
                          jac=grad_neg_ln_likelihood)
        print(result)
        gp.set_parameter_vector(result.x)
        self.gp = gp

    def _transform_data(self, x):
        C = self.cov
        w, RT = np.linalg.eig(C)
        
        #Get x into the eigenbasis
        R = RT.T
        rotated_means = np.dot(R, self.mean)
        rotated_stds = np.dot(R, np.sqrt(C.diagonal()))
        xR = np.array([np.dot(R, xi) for xi in x])
        xR_stds = np.std(xR, 0)
        xR -= rotated_means
        xR /= xR_stds
        return xR

                        
