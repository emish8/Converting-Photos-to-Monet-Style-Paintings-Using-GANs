import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from scipy.linalg import sqrtm

def calculate_fid(images_real, images_fake, batch_size=50):
    module = hub.load('https://tfhub.dev/google/imagenet/inception_v3/feature_vector/5')
    
    # Compute mean and covariance of real images
    real_activations = get_activations(images_real, module, batch_size)
    real_mean = np.mean(real_activations, axis=0)
    real_cov = np.cov(real_activations, rowvar=False)
    
    # Compute mean and covariance of generated images
    fake_activations = get_activations(images_fake, module, batch_size)
    fake_mean = np.mean(fake_activations, axis=0)
    fake_cov = np.cov(fake_activations, rowvar=False)
    
    # Calculate FID
    mean_diff = real_mean - fake_mean
    cov_product = real_cov.dot(fake_cov)
    cov_sqrt = sqrtm(cov_product).real
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real
    fid = mean_diff.dot(mean_diff) + np.trace(real_cov + fake_cov - 2 * cov_sqrt)
    
    return fid
    
 def get_activations(images, module, batch_size):
    num_images = len(images)
    activations = np.zeros((num_images, 2048))
    
    for i in range(0, num_images, batch_size):
        batch = images[i:i+batch_size]
        batch = tf.image.resize(batch, (299, 299))  # Resize to Inception-v3 input size
        batch = (batch - 0.5) * 2  # Normalize to [-1, 1]
        activations[i:i+batch_size] = module(batch)['default'].numpy()
    
    return activations
