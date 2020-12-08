For this project, the user will upload an image of a person and chose whether they want one-to-all clothing items or the whole person subtracted from the image.

### Day 1:
I chose the [clothing-co-parsing](https://github.com/bearpaw/clothing-co-parsing) dataset to build a background subtraction model.  The dataset is downloaded and masking images are visualized.

### Day 2:
The masked images are only 1004 while the total number of images is 2098. Since the dataset will be much smaller, I'll consider data augmentation. I still need to understand the dataset better to decide what to do with it. After all, this project is an exercise.

### Day 3:
The goal is still to develop an understanding of the dataset. The file [label_list.mat](https://github.com/bearpaw/clothing-co-parsing/blob/master/label_list.mat) maps label numbers to label names. 

I'm not sure I understand what is meant by 'image-level annotations'. It's probably treating the dataset as though it's for classification purposes rather than segmentation.

We can see from plotting the first image in every folder, the images are arranged in a different order for every folder. I need to find a way to map all the information in each folder/file to its corresponding image.

### Day 4:
It shouldn't have taken me this long to figure out how images are annotated. Half the images are segmented pixel-to-pixel, up t image number 1004. The other half is annotated by image, that is, the file for each image contains an array representing the classes in the image.

I was hoping to be able to build a model that can subtract whatever clothing item the user decides from the image. Not gonna happen. 1004 images are not enough to train a model to classify 59 different elements and what pixels they occupy. I think I can use them to subtract the person -including her/his clothes- from the background.

I'll keep searching for datasets for extracting clothes that are large enough.

The next step is to treat all 59 elements as one and the background, denoted as 'null', as the other. I can benefit from using grayscale now that the task is much less complex. Black and white it is.

Well, since this is a clothing dataset, maybe I'll feel better if instead of subtracting the model I subtract the clothes. Meaning I'll add hair and skin to the background as one class and all clothes as the other class. This will only affect preparing the data.

Reading raw images using `os.listdir(path)` returns an unordered list so I used `sorted()` to order them.