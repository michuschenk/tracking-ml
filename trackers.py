import numpy as np
import pandas as pd
from cpymad.madx import Madx
import pysixtracklib as pyst


class PySixTrackLibTracker:
    """ Using fixed SPS sequence for now ... """

    def __init__(self):
        # Prepare MAD-X, load SPS sequence (just an example)
        mad = Madx()
        mad.options.echo = False
        mad.call(file="SPS_Q20_thin.seq")
        mad.use(sequence='sps')
        twiss = mad.twiss()
        q1mad = twiss.summary['q1']
        q2mad = twiss.summary['q2']
        print('q1mad', q1mad)
        print('q2mad', q2mad)

        # Build elements for SixTrackLib
        self.elements = pyst.Elements.from_mad(mad.sequence.sps)

    def create_dataset(self, n_particles=100000, n_turns=1, xsize=5e-5,
                       ysize=5e-5, distribution='Gaussian'):
        # Add a beam monitor to elements
        # (given kwargs values will produce only turn by turn data for
        # all particles)
        pyst.append_beam_monitors_to_lattice(
            self.elements.cbuffer, until_turn_elem_by_elem=0,
            until_turn_turn_by_turn=n_turns, until_turn=0, skip_turns=0)

        # Initialise particle distribution
        # TODO: Change possible initial particle distributions
        particles = pyst.Particles.from_ref(
            num_particles=n_particles, p0c=26e9)
        if distribution == 'Gaussian':
            particles.x[:] = xsize * np.random.randn(n_particles)
            particles.px[:] = xsize/10. * np.random.randn(n_particles)
            particles.y[:] = ysize * np.random.randn(n_particles)
            particles.py[:] = ysize/10. * np.random.randn(n_particles)
        elif distribution == 'Uniform':
            particles.x[:] = np.random.uniform(
                low=-xsize, high=xsize, size=n_particles)
            particles.px[:] = np.random.uniform(
                low=-xsize/10., high=xsize/10., size=n_particles)
            particles.y[:] = np.random.uniform(
                low=-ysize, high=ysize, size=n_particles)
            particles.py[:] = np.random.uniform(
                low=-ysize/10., high=ysize/10., size=n_particles)

        df_init = pd.DataFrame(
            data={
                'x': particles.x,
                'xp': particles.px,
                'y': particles.y,
                'yp': particles.py,
                'particle_id': particles.particle_id,
                'turn': particles.at_turn
            })

        # Create track job with initialised lattice and particle distr.
        # and perform tracking job
        # (job.collect() required for OpenCL, CUDA, to sync. device and
        # host memory. Here only CPU, but better to keep it in anyway
        # according to M. Schwinzerl).
        job = pyst.TrackJob(self.elements, particles)
        job.track(n_turns)
        job.collect()

        # Turn-by-turn particle information for training dataset
        pdata = job.output.particles[0]
        df = pd.DataFrame(
            data={
                'x': pdata.x,
                'xp': pdata.px,
                'y': pdata.y,
                'yp': pdata.py,
                'particle_id': pdata.particle_id,
                'turn': pdata.at_turn + 1
            })

        df = df_init.append(df, ignore_index=True)
        return df


class LinearTracker:
    """ Taken from PyHT, tracking only (x,xp), alpha = 0 """

    def __init__(self, beta_s0, beta_s1, Q):
        self.beta_s0 = beta_s0
        self.beta_s1 = beta_s1
        self.Q = Q

        i_matrix = np.zeros((2, 2))
        j_matrix = np.zeros((2, 2))

        # Sine component.
        i_matrix[0, 0] = np.sqrt(self.beta_s1 / self.beta_s0)
        i_matrix[0, 1] = 0.
        i_matrix[1, 0] = 0.
        i_matrix[1, 1] = np.sqrt(self.beta_s0 / self.beta_s1)

        # Cosine component.
        j_matrix[0, 0] = 0.
        j_matrix[0, 1] = np.sqrt(self.beta_s0 * self.beta_s1)
        j_matrix[1, 0] = -np.sqrt(1. / (self.beta_s0 * self.beta_s1))
        j_matrix[1, 1] = 0.

        # One-turn phase advance
        c_phi = np.cos(2.*np.pi*self.Q)
        s_phi = np.sin(2.*np.pi*self.Q)

        # Calculate the matrix M and transport the transverse phase
        # spaces through the segment.
        self.M00 = i_matrix[0, 0] * c_phi + j_matrix[0, 0] * s_phi
        self.M01 = i_matrix[0, 1] * c_phi + j_matrix[0, 1] * s_phi
        self.M10 = i_matrix[1, 0] * c_phi + j_matrix[1, 0] * s_phi
        self.M11 = i_matrix[1, 1] * c_phi + j_matrix[1, 1] * s_phi

    def track(self, x_in, xp_in):
        x_out = self.M00*x_in + self.M01*xp_in
        xp_out = self.M10*x_in + self.M11*xp_in
        return x_out, xp_out

    def create_dataset(self, n_particles=1000, distribution='Gaussian', n_turns=1):
        # Generate input 'bunch' coordinates
        if distribution == 'Gaussian':
            x = np.random.randn(n_particles)
            xp = np.random.randn(n_particles) / self.beta_s0
        elif distribution == 'Uniform':
            # TODO: Choice of grid very arbitrary ...
            x = np.random.uniform(-6., 6., n_particles)
            xp = np.random.uniform(
                -10./self.beta_s0, 10./self.beta_s0, n_particles)
        else:
            raise ValueError('Distributions other than "Unform" or "Gaussian"'
                             'not supported')

        # Do 1D betatron tracking for n_turns
        inp = np.array([x, xp])
        centroid = np.zeros((n_turns, 2))
        for i in range(n_turns):
            centroid[i, 0] = np.mean(x)
            centroid[i, 1] = np.mean(xp)
            x, xp = self.track(x, xp)
        out = np.array([x, xp])

        data_dict = {
            'x_in': inp[0, :], 'xp_in': inp[1, :],
            'x_out': out[0, :], 'xp_out': out[1, :]}
        df = pd.DataFrame(data_dict)

        return df


def generate_tracking_data(tracker, n_particles, n_turns, xsize=1e-6,
                           ysize=1e-6, filename="temp",
                           distribution='Gaussian'):
    if tracker == 'linear':
        lin_tracker = LinearTracker(beta_s0=25., beta_s1=25., Q=20.13)
        tracking_data = lin_tracker.create_dataset(
            n_particles=n_particles, distribution=distribution,
            n_turns=n_turns)
    elif tracker == 'nonlinear':
        nonlin_tracker = PySixTrackLibTracker()
        tracking_data = nonlin_tracker.create_dataset(
            n_particles=n_particles, n_turns=n_turns, xsize=xsize,
            ysize=ysize, distribution=distribution)
    hdf_file = pd.HDFStore(filename, mode='w')
    hdf_file['data'] = tracking_data
    hdf_file.close()
    
    return tracking_data
