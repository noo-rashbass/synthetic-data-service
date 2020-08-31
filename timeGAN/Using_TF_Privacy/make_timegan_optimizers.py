import tensorflow as tf
from dp_optimizers_NEW import make_gaussian_optimizer_class


DPOptimizer = make_gaussian_optimizer_class(tf.compat.v1.train.AdamOptimizer)

## Just some parameters for checking the optimizers can be used without errors
## I'm finishing other tools for estimating better performance values and will update here when done
embedder0_clip = 1
embedder_clip = 1
disc_clip = 1
embedder0_static_clip = 1
embedder_static_clip = 1
disc_static_clip = 1

embedder0_noise_multiplier = 0.0001
embedder_noise_multiplier = 0.0001
disc_noise_multiplier = 0.0001
embedder0_static_noise_multiplier = 0.0001
embedder_static_noise_multiplier = 0.0001
disc_static_noise_multiplier = 0.0001


# Non-static optimizers
def make_timegan_optimizers(learning_rate = 1e-4, num_microbatches = None):

	embedder0_optimizer = DPOptimizer(
	    l2_norm_clip=embedder0_clip,
	    noise_multiplier=embedder0_noise_multiplier,
	    num_microbatches=num_microbatches,
	    learning_rate=learning_rate)
	embedder_optimizer = DPOptimizer(
	    l2_norm_clip=embedder_l2_norm_clip,
	    noise_multiplier=embedder_noise_multiplier,
	    num_microbatches=num_microbatches,
	    learning_rate=learning_rate)
	gen_s_optimizer = tf.keras.optimizers.Adam() # Gen does not need DP
	generator_optimizer = tf.keras.optimizers.Adam() # Gen does not need DP
	discriminator_optimizer = DPOptimizer(
	    l2_norm_clip=disc_clip,
	    noise_multiplier=disc_noise_multiplier,
	    num_microbatches=num_microbatches,
	    learning_rate=learning_rate)

	# Static optimizers
	embedder0_static_optimizer = DPOptimizer(
	    l2_norm_clip=embedder0_static_clip,
	    noise_multiplier=embedder0_static_noise_multiplier,
	    num_microbatches=num_microbatches,
	    learning_rate=learning_rate)
	embedder_static_optimizer = DPOptimizer(
	    l2_norm_clip=embedder_static_clip,
	    noise_multiplier=embedder_static_noise_multiplier,
	    num_microbatches=num_microbatches,
	    learning_rate=learning_rate)
	generator_static_optimizer = tf.keras.optimizers.Adam() # Gen does not need DP
	discriminator_static_optimizer = DPOptimizer(
	    l2_norm_clip=disc_static_clip,
	    noise_multiplier=disc_static_noise_multiplier,
	    num_microbatches=num_microbatches,
	    learning_rate=learning_rate)