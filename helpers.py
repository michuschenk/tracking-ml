import numpy as np
import pandas as pd
from cpymad.madx import Madx
import pysixtracklib as pyst


class PySixTrackLibTracker:
    """ Using fixed SPS sequence for now ... """

    def __init__(self):
        # Prepare MAD-X
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

    def create_dataset(self, n_particles=40000, n_turns=1):
        # Add a beam monitor to elements
        # (given kwargs values will produce only turn by turn data for all
        # particles)
        pyst.append_beam_monitors_to_lattice(
            self.elements.cbuffer, until_turn_elem_by_elem=0,
            until_turn_turn_by_turn=n_turns, until_turn=0, skip_turns=0)

        # Produce particle set
        particles = pyst.Particles.from_ref(num_particles=n_particles, p0c=26e9)

        # TODO: Change possible initial particle distributions
        # x = np.random.uniform(-1e-6, 1e-6, n_particles)
        # px = np.random.uniform(-1e-8, 1e-8, n_particles)
        x = 3e-6 * np.random.randn(n_particles)
        px = 1e-7 * np.random.randn(n_particles)
        particles.x[:] = x[:]
        particles.px[:] = px[:]

        # Dataframe for input and output data
        idx_srt_in = particles.particle_id
        pdata_df = {
            'x_in': particles.x.take(idx_srt_in),
            'xp_in': particles.px.take(idx_srt_in)}

        # Create track job with initialised lattice and particle distr.
        # and start tracking.
        job = pyst.TrackJob(self.elements, particles)
        job.track(n_turns)
        job.collect()

        # Process output, bring into 'correct' form and save to file
        pdata = job.output.particles[0]
        msk_last_turn = (pdata.at_turn == (n_turns - 1))
        pdata_x = pdata.x[msk_last_turn]
        pdata_px = pdata.px[msk_last_turn]
        pdata_pid = pdata.particle_id[msk_last_turn]

        idx_srt_out = np.argsort(pdata_pid)
        pdata_df['x_out'] = pdata_x.take(idx_srt_out)
        pdata_df['xp_out'] = pdata_px.take(idx_srt_out)

        pdata_df = pd.DataFrame(data=pdata_df)
        # TODO: save data to h5 file -- allow for possibilty to load from
        #       file or regenerate data
        # hdf_file = pd.HDFStore('sixtrack_training_set.h5')
        # hdf_file['tracking_data'] = pdata_df

        return pdata_df


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

    def create_dataset(self, n_particles=1000, distr='Gaussian', n_turns=1):
        # Generate input 'bunch' coordinates
        if distr == 'Gaussian':
            x = np.random.randn(n_particles)
            xp = np.random.randn(n_particles) / self.beta_s0
        elif distr == 'Uniform':
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

        return df, centroid
