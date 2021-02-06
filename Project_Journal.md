For this project, the user will upload an image of a person and chose whether they want one-to-all clothing items or the whole person subtracted from the image.

### Day 1:
I chose the [clothing-co-parsing](https://github.com/bearpaw/clothing-co-parsing) dataset to build a background subtraction model. I downloaded the dataset and visualized masked images.

### Day 2:
The masked images are only 1004, while the total number of images is 2098. Since the dataset will be much smaller, I'll consider data augmentation. I still need to understand the dataset better to decide what to do with it. After all, this project is an exercise.

### Day 3:
The goal is still to develop an understanding of the dataset. The file [label_list.mat](https://github.com/bearpaw/clothing-co-parsing/blob/master/label_list.mat) maps label numbers to label names. 

I'm not sure I understand what is meant by 'image-level annotations'. It's probably treating the dataset as though it's for classification purposes rather than segmentation.

We can see from plotting the first image in every folder they are arranged in a different order for every folder. I need to find a way to map all the information in each folder/file to its corresponding image.

### Day 4:
It shouldn't have taken me this long to figure out how images are annotated. Half the images are segmented pixel-to-pixel, up t image number 1004. The other half is annotated by image; that is, the file for each image contains an array representing the classes in the image.

I was hoping to be able to build a model that can subtract whatever clothing item the user decides from the image. Not gonna happen. 1004 images are not enough to train a model to classify 59 different elements and what pixels they occupy. I think I can use them to subtract the person -including her/his clothes- from the background.

I'll keep searching for datasets for extracting clothes that are large enough.

The next step is to treat all 59 elements as one and the background, denoted as 'null', as the other. I can benefit from using grayscale now that the task is much less complex. Black and white it is.

Well, since this is a clothing dataset, maybe I'll feel better if, instead of subtracting the model, I subtract the clothes. Meaning I'll add hair and skin to the background as one class and all clothes as the other class. This will only affect preparing the data.

Reading raw images using `os.listdir(path)` returns an unordered list, so I used `sorted()` to order them.

### Day 5:
Now that the decision is made, The next step is to build the final dataset. My dataset will consist of the raw image and another image where the clothes and accessories are 1, and background skin and hair are 0.

Since the dataset is so small, it's better to use a pre-trained model. I was hoping to train a model end-to-end, but I'll try with some other dataset. For now, this project is still worth completing.

### Day 6:
Small problem; raw images are read as RGB, and there's no way to tell what colorspace they originally have. I tried changing the color space from BGR, the default for `cv.imread`, to many other color spaces. I couldn't find the right format. I could ignore it if I'm feeding raw images in Grayscale. However, I'd like to have the option of feeding colored images to the network. The problem was solved by using `mpimg.imread` instead of `cv.imread`.

### Day 7:
Now is the time to build the network. VGG seems to suffice for this task. However, I want to try something new, so I'll search for alternatives. The paper [A SURVEY ON DEEP LEARNING-BASED ARCHITECTURES FOR SEMANTIC SEGMENTATION ON 2D IMAGES](https://arxiv.org/pdf/1912.10230.pdf) is a great source; I'll use it to decide on the network or try them all 🤷🏽‍♀️.

Ok. Good insight from the paper. Many schemes are too advances for this project. Choosing between U-NET and SEG-NET is not so difficult as the latter is faster and delivers similar quality. On to building the network.

### Day 8:
After reading the [SegNet paper](https://arxiv.org/pdf/1511.00561.pdf), I can build the network. The decoder is based on [VGG16](https://arxiv.org/pdf/1409.1556.pdf). Therefore, I can download it pre-trained, as the SegNet paper suggests. Then, I'll build the corresponding decoder for each of the 13 Convolutional Layers in the encoder. Finally, a pixel-wise classification layer, Softmax. SegNet discards the final Fully-Connected Layer in VGG; to retain higher resolution feature maps at the deepest encoder output and reduce the size.  Thus, computations and complexity are reduced.

In SegNet, the output of each `MaxPool` in the encoder is the input to the corresponding upsampling layers in the decoder. In the VGG16 model downloaded from Pytorch, the MaxPooling layers are operations number 6, 13, 23, 33, 43. The outputs of these operations can be fetched using 

Now the model is completed, it's time to train.

### Day 9:
It's been a while. I noticed I did not save the modified parts of the dataset; Labls. So, I added the required part to `Prep-Visualise-Data`.
 
I worked out the necessary transformations needed to feed the raw images to the network. I tried it by feeding an image to VGG_16_bn, not my model, as there may be problems with the model.

### Day 10:
As usual, feeding our data to the model is not straightforward. I've been stuck here for longer than I wished. Yet, it's expected. 

The problem is mostly picking up the pooling layer's outputs. It turns out the VGG16 built-in model does not take the indices as an output. So the next step is rebuilding the model and initializing the encoder part with the VGG16 weights.

### Day 11:
The model is adjusted as needed. While the code runs with no error, I need a better understanding of the output data before training.

### Day 12:
It's been a while. Now I understand what everything means. The output is random, as shown with the output image. Initializing the encoder part of the network with VGG16 weights is no longer straight forward, as the network is slightly changed. Now I need to find a different way to incorporate these weights. I remember passing by something similar. Time for Research 🧐
