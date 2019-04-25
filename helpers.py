import numpy as np


class LinearTracker():
    ''' Taken from PyHT, tracking only (x,xp), alpha = 0 '''

    def __init__(self, beta_s0, beta_s1, Q):
        I = np.zeros((2,2))
        J = np.zeros((2,2))

        # Sine component.
        I[0,0] = np.sqrt(beta_s1 / beta_s0)
        I[0,1] = 0.
        I[1,0] = 0.
        I[1,1] = np.sqrt(beta_s0 / beta_s1)

        # Cosine component.
        J[0,0] = 0.
        J[0,1] = np.sqrt(beta_s0 * beta_s1)
        J[1,0] = -np.sqrt(1. / (beta_s0 * beta_s1))
        J[1,1] = 0.

        # One-turn phase advance
        c_phi = np.cos(2.*np.pi*Q)
        s_phi = np.sin(2.*np.pi*Q)

        # Calculate the matrix M and transport the transverse phase
        # spaces through the segment.
        self.M00 = I[0,0] * c_phi + J[0,0] * s_phi
        self.M01 = I[0,1] * c_phi + J[0,1] * s_phi
        self.M10 = I[1,0] * c_phi + J[1,0] * s_phi
        self.M11 = I[1,1] * c_phi + J[1,1] * s_phi

    def track(self, x_in, xp_in):
        x_out = self.M00*x_in + self.M01*xp_in
        xp_out = self.M10*x_in + self.M11*xp_in
        return x_out, xp_out


def create_fake_dataset(
        n_samples=1000, beta_s0=1., beta_s1=1., Q=20.13, distr='Gaussian',
        n_turns=1):

    # Load simple linear tracker for 1D betatron
    lin_tracker = LinearTracker(beta_s0, beta_s1, Q)

    # Generate input 'bunch' coordinates
    if distr == 'Gaussian':
        x = np.random.randn(n_samples)
        xp = np.random.randn(n_samples) / beta_s0
    elif distr == 'Uniform':
        x = np.random.uniform(-6., 6., n_samples)
        xp = np.random.uniform(-10./beta_s0, 10./beta_s0, n_samples)

    # Do tracking for n_turns
    inp = np.array([x, xp])
    centroid = np.zeros((n_turns, 2))
    for i in range(n_turns):
        centroid[i,0] = np.mean(x)
        centroid[i,1] = np.mean(xp)
        x, xp = lin_tracker.track(x, xp)
    out = np.array([x, xp])

    return inp, out, centroid
