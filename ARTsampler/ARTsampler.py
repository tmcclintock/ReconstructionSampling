#In development.

import numpy as np
import scipy.optimize as op
import scipy.stats as ss
import george

class ARTsampler(object):
    """A sampler that uses PDF reconstruction.

    Args:
        lnprob (callable): the log PDF of interest
        lnprob_args (collection): arguments to lnprob function
        x0 (list or array-like): initial guess of the parameters
        temperature (float): TODO
        quiet (boolean): flag to indicate whether to print updates
        verbose (boolean): flag to get extra outputs

    """
    def __init__(self, lnprob, lnprob_args, x0, temperature=1.,
                 quiet=True, verbose=False):
        x0 = np.atleast_1d(x0)
        assert np.ndim(x0) == 1
        
        #Save the dimensionality of the parameter space
        self.ndim = len(x0)

        #Save everything
        self.lnprob = lnprob
        self.lnprob_args = lnprob_args
        self.T = temperature
        self.quiet = quiet
        self.verbose = verbose
        
        #Step 1 - find the MAP point
        nll = lambda *args: -lnprob(*args)
        if not quiet:
            print("Running the minimizer")
        result = op.minimize(nll, x0, args=lnprob_args, method="BFGS")
        if not quiet and verbose:
            print(result)

        #Step 2 - define the PDF called G -- the Gaussian guess
        self.G = ss.multivariate_normal(mean=result.x, cov=result.hess_inv)

        #Step 3 to 6 - call an update step
        self.g = None
        self.lnPs = None
        self.n_updates = 0
        self.update()

    def _domain_transform(self, x):
        w, R = np.linalg.eig(self.G.cov)
        xp = np.array([np.dot(R.T, xi) for xi in x])
        m = np.dot(R, self.G.mean)
        s = np.sqrt(w)
        return (xp - m)/s

    def update(self):
        """Perform steps 3 to 6 to update the reconstruction.
        """
        self.n_updates += 1
        
        #Draw samples at a 
        N = 200*int(self.ndim * (self.ndim + 3) / 2)
        #g = self.G.rvs(size=N)
        g = ss.multivariate_normal.rvs(mean=self.G.mean,
                                       cov=self.G.cov*self.T,
                                       size=N)

        #Evaluate the posterior
        lnPs = np.array([self.lnprob(gi, *self.lnprob_args) for gi in g])

        #Save the samples
        if self.g is None: #first update
            assert self.lnPs == None
            self.g = g
            self.lnPs = lnPs 
        else: #all other updates
            self.g = np.vstack((self.g, g))
            self.lnPs = np.hstack((self.lnPs, lnPs))

        def fill_lower_diag(a):
            n = int(np.sqrt(len(a)*2))
            mask = np.tri(n, dtype=bool)#, k=0)
            out = np.zeros((n,n), dtype=float)
            out[mask] = a
            #print(out)
            #print(mask, a)
            return out

        #Update the G distribution with the new weights using VI
        #we will minimize the KL divergence between G and P
        def DKL(params, x, lnP):
            mu = params[:self.ndim] #N items
            arr = params[self.ndim:] #values in L
            L = fill_lower_diag(arr)
            cov = np.dot(L, L.T)
            #Return sum G(g)*lnP - G(g)*lnG(g)
            lnG = ss.multivariate_normal.logpdf(x, mean=mu, cov=cov)
            return np.exp(lnG)*(lnP - lnG)
        #Create the initial guess
        mean = self.G.mean
        L = np.linalg.cholesky(self.G.cov)
        arr = L[np.tril_indices(len(L))]
        guess = np.hstack((mean, arr))
        if not self.quiet:
            print("Optimizing G on update {update}".format(update=self.n_updates))
        result = op.least_squares(DKL, guess, args=(self.g, self.lnPs))
        #if not self.quiet and self.verbose:
        #    print(result)
        if not result.success:
            print("Unable to optimize on update {update}".format(update=self.n_updates))
        mean = result.x[:self.ndim]
        L = fill_lower_diag(result.x[self.ndim:])
        self.G = ss.multivariate_normal(mean=mean, cov=np.dot(L,L.T))

        #Use a GP to model the difference
        self.f = self.lnPs - self.G.logpdf(self.g)
            
        kernel = george.kernels.ExpSquaredKernel(metric=4.5, ndim=self.ndim)
        gp = george.GP(kernel, mean=np.min(self.lnPs), fit_mean=False)
        gp.compute(self._domain_transform(self.g))
        def neg_ln_likelihood(p):
            gp.set_parameter_vector(p)
            return -gp.log_likelihood(self.f)
        def grad_neg_ln_likelihood(p):
            gp.set_parameter_vector(p)
            return -gp.grad_log_likelihood(self.f)
        result = op.minimize(neg_ln_likelihood, gp.get_parameter_vector(),
                             jac=grad_neg_ln_likelihood)
        if not self.quiet and self.verbose:
            print(result)
        gp.set_parameter_vector(result.x)
        self.gp = gp
        
        return

    def logpdf(self, x):
        """Predict the log PDF.
        """
        lnG = self.G.logpdf(x)
        GP = self.gp.predict(self.lnPs,
                             self._domain_transform(np.atleast_2d(x)),
                             return_cov=False)
        return np.asscalar(lnG + GP)
            
if __name__ == "__main__":
    print("nothing yet")
