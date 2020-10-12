# EEG-GAN

This is a simple generative adversarial network (GAN) that was trained to synthesize single-epoch electroenphalographic data from one channel. The average of the synthesized single-epoch data also closely follows the average of data recorded from human participants (i.e., the event-related potential); this means that the model did not merely learn a small variety of realistic-looking EEG epochs (i.e., mode collapse), but has learned the true underlying distribution of single-epoch EEGs. The output looks like this:

![EEG GAN samples](https://github.com/samern92/EEG-GAN/blob/master/EEG-GAN_samples.png)

My code is adapted from a handwritten digit-generating tutorial by Balint Gersey (the code is at the end of his [MS thesis](https://www.researchgate.net/publication/326676131_Generative_Adversarial_Networks)).
