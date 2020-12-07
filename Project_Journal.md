For this project, the user will upload an image of a person and chose whether they want one-to-all clothing items or the whole person subtracted from the image.

### Day 1:
I chose the [clothing-co-parsing](https://github.com/bearpaw/clothing-co-parsing) dataset to build a background subtraction model.  The dataset is downloaded and masking images are visualized.

### Day 2:
The masked images are only 1004 while the total number of images is 2098. I was hoping to rely only on making images. Clearly, I can't.

### Day 3:
The goal is still to develop an understanding of the dataset. The file [label_list.mat](https://github.com/bearpaw/clothing-co-parsing/blob/master/label_list.mat) maps label numbers to label names. 

I'm not sure I understand what is meant by 'image-level annotations'. It's probably treating the dataset as though it's for classification purposes rather than segmentation.

We can see from plotting the first image in every folder, the images are arranged in a different order for every folder. I need to find a way to map all the information in each folder/file to its corresponding image.
