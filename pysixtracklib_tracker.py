# EXAMPLE BASED ON RICCARDO DE MARIA'S:
# https://github.com/rdemaria/sixtracklib/blob/mad_example/examples/python/test_sps_mad/test_sps_mad.py

import numpy as np
import pandas as pd
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
print('q1mad', q1mad)
print('q2mad', q2mad)

# Particle tracking
n_particles = 10000
n_turns = 1

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
px = np.random.uniform(-1e-5, 1e-5, n_particles)
x = np.random.uniform(-1e-5, 1e-5, n_particles)
particles.x[:] = x[:]
particles.px[:] = px[:]

idx_srt_in = particles.particle_id
pdata_df = {
    'x_in': particles.x.take(idx_srt_in),
    'xp_in': particles.px.take(idx_srt_in)}


# Create track job with initialised lattice and particle distr.
# and start tracking.
job = pyst.TrackJob(elements, particles)
job.track(n_turns)
job.collect()

# Process output, bring into 'right' form and save to file
pdata = job.output.particles[0]
idx_srt_out = np.argsort(pdata.particle_id)
pdata_df['x_out'] = pdata.x.take(idx_srt_out)
pdata_df['xp_out'] = pdata.px.take(idx_srt_out)
pdata_df = pd.DataFrame(data=pdata_df)
hdf_file = pd.HDFStore('sixtrack_training_set.h5')
hdf_file['tracking_data'] = pdata_df