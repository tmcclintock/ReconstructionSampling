#In development.

import numpy as np
import pyDOE2
import emcee
from george import kernels, GP
from scipy.optimize import minimize

class ARTsampler(object):
    #This constructor will be removed later
    def __init__(self, mean_guess, covariance_guess, lnprob, lnprob_args,
                 Ntraining_points=40, scale=8, Nwalkers=32, max_iter=2,
                 Nburn=100, Nsteps=1000, quiet=True):
        self.iteration = 0 #Haven't done anything yet
        training_points = self._make_training_points(self.iteration,
                                                     mean_guess,
                                                     covariance_guess,
                                                     Ntraining_points, scale)
        
        self.lnprob = lnprob
        self.lnprob_args = lnprob_args

        #Set quantities that have been computed so far
        self.mean_guesses = [mean_guess]
        self.covariance_guess = [covariance_guess]
        self.training_points = [training_points]

        #Initialize lists
        self.stages = []
        self._emcee_samplers = []
        self.chains = []

        #Save other attributes
        self.Ntraining_points = Ntraining_points
        self.scale = scale
        self.Nwalkers = Nwalkers
        self.Nburn = Nburn
        self.Nsteps = Nsteps
        self.max_iter = max_iter
        self.quiet = quiet

    def single_iteration(self):
        i = self.iteration

        if not self.quiet:
            print("Performing iteration {0}".format(i))
        
        if i > 0:
            chain = self.get_samples()
            mean_guess = np.mean(chain, 0)
            cov_guess = np.cov(chain.T)
            training_points = self._make_training_points(i, mean_guess,
                                                         cov_guess,
                                                         self.Ntraining_points,
                                                         self.scale)
            self.mean_guesses.append(mean_guess)
            self.covariance_guess.append(cov_guess)
            self.training_points.append(training_points)
            
        if i > self.max_iter:
            print("Iteration {0} reached, max_iter is {1}.".format(i, self.max_iter))
            return False
        mean_guess = self.mean_guesses[i]
        cov_guess = self.covariance_guess[i]
        points = self.training_points[i]

        if not self.quiet:
            print("Computing log-probability of training points.")
        lnlikes = np.array([self.lnprob(p, self.lnprob_args) for p in points]).flatten()

        #Make the current stage
        stage = ARTstage(mean_guess, cov_guess, points, lnlikes, self.quiet)
        
        #Perform MCMC
        initial = mean_guess
        ndim = len(initial)
        nwalkers = self.Nwalkers
        sampler = emcee.EnsembleSampler(nwalkers, ndim, stage.predict)
        if not self.quiet:
            print("Running first burn-in")
        p0 = np.random.multivariate_normal(mean_guess, cov_guess, size=nwalkers)
        #p0 = initial + 1e-4*np.random.randn(nwalkers, ndim)
        p0, lp, _ = sampler.run_mcmc(p0, self.Nburn)
        if not self.quiet:
            print("Running second burn-in")
        p0 = np.random.multivariate_normal(p0[np.argmax(lp)],
                                           cov_guess, size=nwalkers)
        #p0 = p0[np.argmax(lp)] + 1e-4*np.random.randn(nwalkers, ndim)
        p0, lp, _ = sampler.run_mcmc(p0, self.Nburn)
        sampler.reset()
        if not self.quiet:
            print("Running production...")
        sampler.run_mcmc(p0, self.Nsteps)

        self.stages.append(stage)
        self._emcee_samplers.append(sampler)
        self.chains.append(sampler.flatchain)
        
        self.iteration += 1
        return True

    def get_samples(self, index=-1):
        return self.chains[index]

    def get_training_points(self, index=-1):
        return self.training_points[index]
    
    def _make_training_points(self, iteration, mean, cov,
                             Ntraining_points=100, scale=8):
        #Create LH training samples
        x = pyDOE2.lhs(len(mean), samples=Ntraining_points,
                       criterion="center", iterations=5)
        
        #Transform them correctly
        x -= 0.5 #center the training points
        x = np.append(x, np.atleast_2d(np.zeros_like(x[0])), axis=0)
        w, RT = np.linalg.eig(cov)
        R = RT.T
        
        return np.dot(scale*x[:]*np.sqrt(w), R.T)[:] + mean

class ARTstage(object):
    #This constructor will be removed later
    def __init__(self, mean, covariance, points, lnlikes, quiet=True):
        self.mean = mean
        self.cov = covariance
        self.points = points
        self.lnlikes_true = lnlikes
        self.lnlike_max = np.max(lnlikes)
        self.lnlikes = lnlikes - self.lnlike_max #max is now at 0
        print(self.lnlikes.shape)
        self.x = self._transform_data(points)
        
        _guess = 4.5 #kernel length guess
        ndim = len(points[0])
        kernel = kernels.ExpSquaredKernel(metric=_guess, ndim=ndim)
        lnPmin = np.min(self.lnlikes)
        gp = GP(kernel, mean=lnPmin-np.fabs(lnPmin*3))
        
        gp.compute(self.x)
        def neg_ln_likelihood(p):
            gp.set_parameter_vector(p)
            return -gp.log_likelihood(self.lnlikes)

        def grad_neg_ln_likelihood(p):
            gp.set_parameter_vector(p)
            return -gp.grad_log_likelihood(self.lnlikes)

        result = minimize(neg_ln_likelihood,
                          gp.get_parameter_vector(),
                          jac=grad_neg_ln_likelihood)
        if not quiet:
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
        xR -= rotated_means
        xR /= rotated_stds
        return xR

    def predict(self, x):
        #Make it the correct format
        x2d = np.atleast_2d(x).copy()
        
        #Get x into the eigenbasis and predict
        pred, pred_var = self.gp.predict(self.lnlikes,
                                         self._transform_data(x2d))
        return pred + self.lnlike_max 
