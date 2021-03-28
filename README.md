# EEG-GAN

This is a simple generative adversarial network (GAN) that was trained to synthesize single-epoch electroenphalographic data from one channel. The individual synthetic epochs (light purple in the figure below) look qualitatively different from one another and their average at each time point closely follows the actual average of data recorded from human participants (i.e., the event-related potential). This indicates that the model has correctly approximated the underlying distribution of single-epoch EEGs.

![EEG GAN samples](https://github.com/samern92/EEG-GAN/blob/master/EEG-GAN_samples.png)

My code is adapted from a handwritten digit-generating tutorial by Balint Gersey (the code is at the end of his [MS thesis](https://www.researchgate.net/publication/326676131_Generative_Adversarial_Networks)).
