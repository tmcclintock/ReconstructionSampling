#In development.

import numpy as np
import scipy.optimize as op
import scipy.stats as ss
import george
import pyDOE2

class ARTsampler(object):
    """A sampler that uses PDF reconstruction.

    Args:
        lnprob (callable): the log PDF of interest
        lnprob_args (collection): arguments to lnprob function
        x0 (list or array-like): initial guess of the parameters
        temperature (float): TODO
        multiplicity (int): TODO
        quiet (boolean): flag to indicate whether to print updates
        verbose (boolean): flag to get extra outputs

    """
    def __init__(self, lnprob, lnprob_args, x0, temperature=5., multiplicity=10,
                 quiet=True, verbose=False):
        x0 = np.atleast_1d(x0)
        assert np.ndim(x0) == 1
        
        #Save the dimensionality of the parameter space
        self.ndim = len(x0)

        #Save everything
        self.lnprob = lnprob
        self.lnprob_args = lnprob_args
        self.T = temperature
        self.M = multiplicity
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
        self.w, self.R = np.linalg.eig(self.G.cov)
        self.L = np.linalg.cholesky(self.G.cov)
        self.Li = np.linalg.cholesky(np.linalg.inv(self.G.cov))

        #Step 3 to 6 - call an update step
        self.g = None
        self.lnPs = None
        self.n_updates = 0
        self.update()

    def Gaussian_component_logpdf(self, x):
        return self.A + self.G.logpdf(x)
        
    def _domain_transform(self, x): #Goes from parameter space to a circular space
        m = self.G.mean
        Li = self.Li
        return np.dot((x - m)[:], Li) 
        
    def _inverse_domain_transform(self, x): #Circular space to parameter space
        m = self.G.mean
        L = self.L
        return np.dot(x[:], L) + m

    def update(self):
        """Perform steps 3 to 6 to update the reconstruction.
        """
        self.n_updates += 1

        #Draw samples from G in a clever way
        #Take nested samples in expanding Latin-hypercube
        N = int(self.ndim * (self.ndim + 3) / 2)
        g = pyDOE2.lhs(self.ndim, samples=N, criterion="center", iterations=5) - 0.5
        T_M = float(self.T)/(self.M - 1) #scale per nest
        for i in range(1, self.M):
            gnew = (pyDOE2.lhs(self.ndim, samples=N, criterion="center", iterations=5) - 0.5) * (i*T_M + 1)
            #Randomly rotate the LH
            gnew = np.dot(gnew[:], ss.special_ortho_group.rvs(self.ndim))
            g = np.vstack((g, gnew))
        """print(g.shape)
        print("temp = {T}".format(T=self.T))
        print(np.min(g[:,0]), np.max(g[:,0]))
        print(np.min(g[:,1]), np.max(g[:,1]))"""
        g = self._inverse_domain_transform(g)
        """print(g.shape)
        print(np.min(g[:,0]), np.max(g[:,0]))
        print(np.min(g[:,1]), np.max(g[:,1]))"""

        #Evaluate the posterior at all samples
        lnPs = np.array([self.lnprob(gi, *self.lnprob_args) for gi in g])

        #TODO check for infs

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
            return out

        #Update the G distribution with the new weights using VI
        #we will minimize the KL divergence between G and P
        def DKL(params, x, lnP):
            A = params[0]
            mu = params[1:self.ndim+1] #N items
            arr = params[1+self.ndim:] #values in L
            L = fill_lower_diag(arr)
            cov = np.dot(L, L.T)
            lnG = ss.multivariate_normal.logpdf(x, mean=mu, cov=cov)
            #return np.sum(np.exp(lnG)*(lnG - lnP + A)) #KL divergence
            l2 = (lnP - A - lnG)**2 #least squares
            #print(l2)
            #print(lnG)
            #print(lnP)
            return l2
        #Create the initial guess
        mean = np.hstack(([9.5], self.G.mean))
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
        self.A = result.x[0]
        mean = result.x[1:self.ndim+1]
        L = fill_lower_diag(result.x[self.ndim+1:])
        self.G = ss.multivariate_normal(mean=mean, cov=np.dot(L,L.T))
        self.w, self.R = np.linalg.eig(self.G.cov)
        self.L = L
        self.Li = np.linalg.cholesky(np.linalg.inv(self.G.cov))

        #Use a GP to model the difference
        self.f = self.lnPs - self.Gaussian_component_logpdf(self.g)
        
        metric = self.G.cov
        kernel = george.kernels.ExpSquaredKernel(metric=4.5, ndim=self.ndim)
        #kernel = george.kernels.ExpSquaredKernel(metric="isotropic", ndim=self.ndim)

        Gpart = self.Gaussian_component_logpdf
        _idt = self._inverse_domain_transform
        class MeanModel(george.modeling.Model):
            def get_value(self, x):
                return Gpart(_idt(x)) #x has a flat metric so transform it
        gp = george.GP(kernel, mean=MeanModel(), fit_mean=False)

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
        GP = self.gp.predict(self.lnPs,
                             self._domain_transform(np.atleast_2d(x)),
                             return_cov=False)
        return np.squeeze(GP)#np.asscalar(GP)

if __name__ == "__main__":
    print("nothing yet")
