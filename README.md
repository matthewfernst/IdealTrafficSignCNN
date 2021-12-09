# Ideal Traffic Sign Images Classification For Convolutional Neural Networks

### [Matthew Ernst](https://github.com/matthewfernst) and [James Yost](https://github.com/Jeyost)

### Abstract

Images easily fool convolutional neural networks with noise. They are not as secure as previously thought. We show this by training a VGG16 model on the Mapillary data set on over 200,000 images. We were able to trick this model into a classification of images that appear to be random noise as 100% confidence prediction of a particular class. Ultimately, we found these ideal images as a result of a random search over the image space. With our small images 32x32x3, we were able to find these relatively quickly and prove the draw- backs of convolutional neural networks and their security.
