# General Goal:

- want to predict poverty using satellite images
- poverty prediction currently done through day and night images since this is less expensive than on ground surveys (day + night -> poor)
- can model using just day imagery (day -> night, day + generated night -> poor). benefit here is less data needed.
- but even better, maybe can do day -> latent -> poverty and directly learn what the latent is instead of assuming night captures the latent
	- train the latent based on the night. experimentation here.

# Baseline Methods:

1) day -> poverty (discriminative)
2) night -> poverty (discriminative)
2) day / night -> poor or not (discriminative)
	imgs -> VGG features -> fully connected -> sigmoid

# Notes

- for now doing binary classification, can do 5-class, or regression.
- should get more data (right now 1k, we want 41k).
- You may wish to consider certain semi-supervised learning baselines such as self-consistency to compare against the efficacy of the GAN-based approach.

# Actual Models:

1) day -> night, night-> poverty (generative)
	train two step
	day -> night is generative (pix2pix)
	day + generated night -> poverty is discriminative
2) take night, train autoencoder, take those latents
	take day, train to generate those latents
	take latent, train to get poverty
	then given day, can generate latents, 
3) conditional multiclass gan
	(adversarially learned inference)
	generator conditioned on daytime image
	generates latent embedding
	discriminator classifies embedding as fake/poor/rich
4) generator -> conditioned on poverty, generates day (or night or day & night)
   discriminator -> given day / night / both, predicts poverty (fake | poor | rich)
   see if match up
