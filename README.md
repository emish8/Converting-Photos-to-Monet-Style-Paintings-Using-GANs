# Converting photos to Monet Style Paintings Using GANs And Neural Style Transfer

## Problem Definition
- Monet paintings are named after French artist Claude Monet, he was one of the  most popular Impressionist Era Artist. This project aims to convert normal photos to Monet Style paintings.
- I have used two types of deep learning models -
1.   CycleGANs 
2.   Neural Style Transfe



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

![image](https://user-images.githubusercontent.com/83595196/226117008-ed535b2a-9f56-455b-b749-5257e7abff62.png)

## Models

- CycleGAN - CycleGAN uses a cycle consistency loss to enable training without the need for paired data. In other words, it can translate from one domain to another without a one-to-one mapping between the source and target domain.
Image-to-Image Translation of Unpaired Data.

- Neural Style Transfer - Style transfer is a computer vision technique that takes two images—a content image and a style reference image—and blends them together so that the resulting output image retains the core elements of the content image, but appears to be “painted” in the style of the style reference image.


###  CycleGAN Output
The cycle Gans has been trained for 10 epochs gave this output. one ecpoch training time exceeds 1 hour on google collab TPU.

![2ep 6](https://user-images.githubusercontent.com/83595196/226182005-777aab60-edc9-4b79-a81c-98b2dc073b1c.png)

### Neural Style transfer output
- Input
The model takes two inputs. The content image is the one we want to transform and the style image is the one whose style we want to capture.
Training time is very less, as we use few layers of pretrained VGG19 to extract features from both the images.

![download (1)](https://user-images.githubusercontent.com/83595196/226182084-e40e104b-9a83-4b9f-8c65-509d4c42af88.png)

- Output
![image](https://user-images.githubusercontent.com/83595196/226182192-7ff85ca5-b406-40a6-a2ff-8be7c1960363.png)








