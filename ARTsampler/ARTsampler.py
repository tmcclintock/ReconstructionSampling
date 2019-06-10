#In development.

import numpy as np
import pyDOE2
import emcee
from george import kernels, GP
from scipy.optimize import minimize

class ARTsampler(object):
    def __init__(self, prior_distributions, lnprob, lnprob_args,
                 Ntraining_points=40, scale=8, Nwalkers=32, max_iter=2,
                 Nburn=100, Nsteps=1000, quiet=True):
        #Enfore dimensionality
        #prior_volume = np.asarray(prior_volume)
        #priov_volume = np.atleast_2d(prior_volume)
        #assert prior_volume.ndim == 2 #array of min and maxes
        #assert len(prior_volume[0]) == 2 #only a min and a max
        
        self.prior_distributions = prior_distributions
        self.ndim = len(prior_distributions)
        self.iteration = 0 #Haven't done anything yet
        
        self.lnprob = lnprob
        self.lnprob_args = lnprob_args

        initial = np.zeros(self.ndim)
        for i in range(self.ndim):
            initial[i] = self.prior_distributions[i].mean()
        def nlp(params):
            return -self.lnprob(params, self.lnprob_args)
        if not quiet:
            print("Maximizing True lnprob:")
        result = minimize(nlp, initial, method="Nelder-Mead", tol=0.01)
        self.peak = result.x
        if not quiet:
            print(result)
        training_points = self._make_initial_training_points(prior_distributions, self.peak,
                                                             Ntraining_points)
        #training_points = self._make_training_points(0, result.x,
        #                                             result2.hess_inv,
        #                                             Ntraining_points,
        #                                             scale)
        #training_points = np.append(training_points, np.atleast_2d(self.peak), axis=0)

        #Set quantities that have been computed so far
        self._training_points = [training_points] #No prior clipping
        self.training_points = [training_points] #Prior clipping

        #Initialize lists
        self.mean_guesses = []
        self.covariance_guesses = []
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
        iteration = self.iteration
        
        if iteration > self.max_iter:
            print("Iteration {0} reached, max_iter is {1}.".format(iteration, self.max_iter))
            return False

        if not self.quiet:
            print("Performing iteration {0}".format(iteration))
        
        if iteration > 0:
            chain = self.get_chain()
            mean_guess = np.mean(chain, 0)
            cov_guess = np.cov(chain.T)
            _training_points = self._make_training_points(iteration, mean_guess,
                                                          cov_guess,
                                                          self.Ntraining_points-1,
                                                          self.scale)
            #Append on the peak point
            _training_points = np.append(_training_points,
                                         np.atleast_2d(self.peak), axis=0)

            self.mean_guesses.append(mean_guess)
            self.covariance_guesses.append(cov_guess)
            self._training_points.append(_training_points)

            #Clip points off that are outside of priors (i.e. have lnprob of -np.inf)
            _points = self._training_points[iteration]
            flags = []
            lnpriors = np.zeros(len(_points))
            for i in range(len(_points)):
                for j in range(self.ndim):
                    lnpriors[i] += self.prior_distributions[j].logpdf(_points[i, j])
                if np.isinf(lnpriors[i]) or np.isnan(lnpriors[i]):
                    flags.append(False)
                else:
                    flags.append(True)
            points = _points[flags]
            self.training_points.append(points)
            
        else:
            self.mean_guesses.append(np.mean(self._training_points[0], 0))
            self.covariance_guesses.append(np.diag(np.var(self._training_points[0], 0)))
        
        mean_guess = self.mean_guesses[iteration]
        cov_guess = self.covariance_guesses[iteration]
        points = self.training_points[iteration]
            
        if not self.quiet:
            print("Computing log-probability of training points.")

        lnlikes = np.array([self.lnprob(p, self.lnprob_args) for p in points]).flatten()

        #Make the current stage
        if not self.quiet:
            print("Reconstructing the distribution")
        stage = ARTstage(mean_guess, cov_guess, points, lnlikes, self.quiet)
        self.stages.append(stage)

        #Perform MCMC
        def lnprob(params):
            lnprior = 0
            for i in range(self.ndim):
                lnprior += self.prior_distributions[i].logpdf(params[i])
            if np.isinf(lnprior):
                return -1e99
            """
            lnlike = 0
            weights_sum = 0
            pred = 0
            for stage in self.stages:
                pll, pll_var = stage.predict(params, True)
                w = 1./pll_var
                pred += pll * w
                weights_sum += w
            return pred/weights_sum
            """
            return self.stages[-1].predict(params)# + lnprior
            
            
        #Initial should always be the peak of the lnprob, since we maximized earlier
        initial = points[np.argmax(lnlikes)]
        ndim = len(initial)
        nwalkers = self.Nwalkers
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
        if not self.quiet:
            print("Running first burn-in")
        #p0 = np.random.multivariate_normal(mean_guess, cov_guess, size=nwalkers)
        p0 = initial + 1e-2*np.random.randn(nwalkers, ndim)
        p0, lp, _ = sampler.run_mcmc(p0, self.Nburn)
        if not self.quiet:
            print("Running second burn-in")
        #p0 = np.random.multivariate_normal(p0[np.argmax(lp)],
        #                                   cov_guess, size=nwalkers)
        p0 = p0[np.argmax(lp)] + 1e-2*np.random.randn(nwalkers, ndim)
        p0, lp, _ = sampler.run_mcmc(p0, self.Nburn)
        sampler.reset()
        if not self.quiet:
            print("Running production...")
        sampler.run_mcmc(p0, self.Nsteps)

        self._emcee_samplers.append(sampler)
        self.chains.append(sampler.flatchain)
        
        self.iteration += 1
        return True

    def get_samples(self, index=-1):
        sampler = self._emcee_samplers[index]
        lnprob = sampler.flatlnprobability
        chain = self.chains[index]
        peak = np.max(lnprob)
        inds = (peak - lnprob) < 2*self.ndim #better than 5*chi^2 points
        return chain[inds]

    def get_chain(self, index=-1):
        return self.chains[index]

    def get_training_points(self, index=-1):
        return self.training_points[index]

    def _make_initial_training_points(self, prior_distributions, best_point,
                                      Ntraining_points=100):
        ndim = len(prior_distributions)
        x = np.zeros((Ntraining_points, ndim))
        x[0] = best_point
        for i in range(ndim):
            x[1:, i] = best_point[i] + 0.1 * np.random.randn(Ntraining_points-1) * \
                prior_distributions[i].std()
        return x
    
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

    def predict(self, x, with_var=False):
        #Make it the correct format
        x2d = np.atleast_2d(x).copy()
        
        #Get x into the eigenbasis and predict
        pred, pred_var = self.gp.predict(self.lnlikes,
                                         self._transform_data(x2d))
        if with_var:
            return pred + self.lnlike_max, pred_var
        else:
            return pred + self.lnlike_max 
