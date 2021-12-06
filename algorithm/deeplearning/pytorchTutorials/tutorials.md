## Pytorch Official Tutorials

#### Saving and loading a general checkpoint in pytorch



#### Create patches/windows of image dataset

https://discuss.pytorch.org/t/how-to-create-patches-windows-of-image-dataset/835/3

You need to write your own Dataset, that will extract all the possible patches positions and store those. Once you iterate on this dataset you will extract and return the patches.  It has the advantage of only storing the patches positions(use a numpy or tensor).





