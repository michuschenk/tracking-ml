# EXAMPLE BASED ON RICCARDO DE MARIA'S:
# https://github.com/rdemaria/sixtracklib/blob/mad_example/examples/python/test_sps_mad/test_sps_mad.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


from cpymad.madx import Madx
import pysixtracklib as pyst

# Prepare MAD-X
mad = Madx()
mad.options.echo = False
mad.call(file="SPS_Q20_thin.seq")
mad.use(sequence='sps')
twiss = mad.twiss()
q1mad = twiss.summary['q1']
q2mad = twiss.summary['q2']
# print('q1mad', q1mad)
# print('q2mad', q2mad)

# Particle tracking
n_particles = 1000
n_turns = 50

# Build elements for SixTrackLib
elements = pyst.Elements.from_mad(mad.sequence.sps)

# Add a beam monitor to elements
# (given kwargs values will produce only turn by turn data for all
# particles)
pyst.append_beam_monitors_to_lattice(
    elements.cbuffer, until_turn_elem_by_elem=0,
    until_turn_turn_by_turn=n_turns, until_turn=0, skip_turns=0)

# Produce particle set
particles = pyst.Particles.from_ref(num_particles=n_particles, p0c=26e9)
# particles.px += np.linspace(-5e-5, 5e-5, n_particles)
# x = np.random.uniform(-1e-6, 1e-6, n_particles)
# px = np.random.uniform(-1e-8, 1e-8, n_particles)
x = 3e-6 * np.random.randn(n_particles)
px = 1e-7 * np.random.randn(n_particles)
particles.x[:] = x[:]
particles.px[:] = px[:]

idx_srt_in = particles.particle_id
pdata_init = {
    'x_in': particles.x.take(idx_srt_in),
    'xp_in': particles.px.take(idx_srt_in)}
init_distr_df = pd.DataFrame(data=pdata_init)

# Create track job with initialised lattice and particle distr.
# and start tracking.
job = pyst.TrackJob(elements, particles)
job.track(n_turns)
job.collect()

# Process output, bring into 'right' form and save to file
plot_turn = 1
plot_n_particles = 1000

# Test StandardScaler -- be careful with shape! Best to use DataFrame
# where features are the columns.
scaler = StandardScaler()
# init_transf = scaler.fit_transform(init_distr_df)

for plot_turn in range(1, 11, 1):
    plt.close('all')
    pdata = job.output.particles[0]
    msk_last_turn = (pdata.at_turn == (plot_turn - 1))
    pdata_x = pdata.x[msk_last_turn]
    pdata_px = pdata.px[msk_last_turn]
    pdata_pid = pdata.particle_id[msk_last_turn]

    idx_srt_out = np.argsort(pdata_pid)
    outp_distr_df = pd.DataFrame(
        data={
            'x_out': pdata_x.take(idx_srt_out),
            'xp_out': pdata_px.take(idx_srt_out)})
    # outp_transf = scaler.transform(outp_distr_df)

    # hdf_file = pd.HDFStore('sixtrack_training_set.h5')
    # hdf_file['tracking_data'] = pdata_df

    plt.suptitle('Turn {:d}'.format(plot_turn))
    # plt.scatter(init_transf[:plot_n_particles, 0],
    #             init_transf[:plot_n_particles, 1],
    #             c='r')
    plt.scatter(init_distr_df['x_in'][:plot_n_particles],
                init_distr_df['xp_in'][:plot_n_particles],
                c='r')

    # plt.scatter(outp_transf[:plot_n_particles, 0],
    #             outp_transf[:plot_n_particles, 1])
    plt.scatter(outp_distr_df['x_out'][:plot_n_particles],
                outp_distr_df['xp_out'][:plot_n_particles])
    # plt.ylim(-5e-7, 5e-7)
    # plt.xlim(-5e-5, 5e-5)
    plt.show()
