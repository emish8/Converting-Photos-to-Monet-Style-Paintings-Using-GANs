# Converting Photos to Monet Style Paintings Using cycleGAN And Neural Style Transfer

## Problem Definition
- Monet paintings are named after French artist Claude Monet, he was one of the  most popular Impressionist Era Artist. This project aims to convert normal photos to Monet Style paintings.
- I have used two types of deep learning models -
1.   CycleGANs 
2.   Neural Style Transfer



## Dataset
This is a kaggle competition Dataset. The aim of the competition is to convert photos to monet style paintings. 

https://www.kaggle.com/competitions/gan-getting-started/data?select=monet_jpg

There are two types of photos:

1. Monet Paintings (jpg) - 300 samples
2. Photo (jpg) - 1000 samples

## Data Visualization

### Sample Monets

![image](https://user-images.githubusercontent.com/83595196/226116470-1a12dd3c-f74e-4756-ad8b-9bce395968aa.png)

### Sample Photos

![image](https://user-images.githubusercontent.com/83595196/226116557-4ec98383-db18-4e0d-8535-823683f6fa31.png)

### Visualizing Gray and RGB Channels
- Monets

![image](https://user-images.githubusercontent.com/83595196/226116591-0dc61e00-cabc-4d55-8f9f-27bf26b19208.png)

- Photos

![image](https://user-images.githubusercontent.com/83595196/226116620-7fdf25f8-4694-4ef4-87d9-1358fe0452f3.png)

### Data Preprocessing 
- CycleGAN
Train Data -  Normalize, Add Random Noise, Random Flip
Test Data - Normalize

- Neural Style Transfer - resize images (if different shapes)

- Adding Random Noise

![image](https://user-images.githubusercontent.com/83595196/226116676-13b0d8eb-99b8-4b3a-aab7-e305dba669c6.png)

- Random Flip
-- Flipping image at random.

![image](https://user-images.githubusercontent.com/83595196/236402244-7cc9e9b7-6aa4-45a2-aa22-d6d72ba82ddb.png)

## Models

- CycleGAN - CycleGAN uses a cycle consistency loss to enable training without the need for paired data. In other words, it can translate from one domain to another without a one-to-one mapping between the source and target domain.
Image-to-Image Translation of Unpaired Data.

- Neural Style Transfer - Style transfer is a computer vision technique that takes two images—a content image and a style reference image—and blends them together so that the resulting output image retains the core elements of the content image, but appears to be “painted” in the style of the style reference image.

## Output

### Pix2pix 
This is a GAN used for paired image-to-image transaltion. This is repurposed for paired image-to-image transalation. The output has lot of noise.

![image](https://user-images.githubusercontent.com/83595196/236399965-6182dd25-53e9-447e-bbf4-5b54b02a013b.png)

###  CycleGAN
The cycle Gans has been trained for 10 epochs gave this output. One ecoch training time exceeds 1 hour on google collab TPU. Repurposed pix2pix gave much better output. Two pix2pix model were used (monet-to-photo and image-to-photo). 

![2ep 6](https://user-images.githubusercontent.com/83595196/226182005-777aab60-edc9-4b79-a81c-98b2dc073b1c.png)

### Neural Style transfer output

The model takes two inputs. The content image is the one we want to transform and the style image is the one whose style we want to capture.
Training time is very less, as we use few layers of pretrained VGG19 to extract features from both the images.

![image](https://user-images.githubusercontent.com/83595196/236398793-1706a978-1dd2-4579-8a3c-d61bb997eaa8.png)

### FID score 
Related file name: calculate_fid.ipynb
Sample of 190 images was taken to generate this score. The Frechet Inception Distance score, or FID for short, is a metric that calculates the distance between feature vectors calculated for real and generated images. The FID score is used to evaluate the quality of images generated by generative adversarial networks, and lower scores have been shown to correlate well with higher quality images.

![image](https://user-images.githubusercontent.com/83595196/236401967-3c8a96a3-af48-4e9b-a8dc-71defa3a9c18.png)


| Model                  | FID score    |
|:-----------------------|-------------:|
| pix2pix                | 19,17,342.13 |
| cycleGAN               | 11,99,486.18 |
| neural style transfer  | 2,08,174.26  |


## Conclusion
Visual output is much better for Neural style transfer than cycleGAN. Neural style transfer is better for a very  specific style  but not for scaling output. The cycleGAn is does not produce the best result thus we can try using other types of GAN for this problem.


## References

CycleGAN  |  TensorFlow Core

https://www.tensorflow.org/tutorials/generative/cyclegan

Generative Adversarial Networks (GANs) - YouTube

https://www.youtube.com/playlist?list=PLdxQ7SoCLQAMGgQAIAcyRevM8VvygTpCu

Deep Learning 46: Unpaired Image to Image translation Network (Cycle GAN) and DiscoGAN - YouTube

https://www.youtube.com/watch?v=nB8uVGbesZ4

Neural style transfer  |  TensorFlow Core

https://www.tensorflow.org/tutorials/generative/style_transfer#build_the_model

Neural Style Transfer - YouTube

https://www.youtube.com/playlist?list=PLBoQnSflObcmbfshq9oNs41vODgXG-608






