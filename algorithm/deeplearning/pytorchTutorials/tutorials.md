## Pytorch Official Tutorials

#### Saving and loading a general checkpoint in pytorch



#### Create patches/windows of image dataset

https://discuss.pytorch.org/t/how-to-create-patches-windows-of-image-dataset/835/3

You need to write your own Dataset, that will extract all the possible patches positions and store those. Once you iterate on this dataset you will extract and return the patches.  It has the advantage of only storing the patches positions(use a numpy or tensor).

#### Create 3D Dataset/DataLoader with patches

https://discuss.pytorch.org/t/creating-3d-dataset-dataloader-with-patches/50861/26

There are different ways. Mostly `overlapping` and `non-overlapping` method.

```python
def generate_patch_32_3(MR, Mask, cor, sag, axi):
    """
    :param MR: 3D MR volume
    :param Mask: 3D Mask same shape MR volume
    :param cor:
    :param sag:
    :param axi:
    :return: MR patch and corresponding Mask patch with shape[32,32,32] and [16,16,16]
    """
    # cor = 16
    hCor = np.int(cor/4)
    # sag = 64
    hSag = np.int(sag/4)
    # axi = 64
    hAxi = np.int(axi/4)
    qShape = [96, 128, 128]
    c = [0, MR.shape[0] - qShape[0]]
    s = [0, MR.shape[1] - qShape[1]]
    a = [0, MR.shape[2] - qShape[2]]
    nQuad = len(c) * len(s) * len(a)
    nPatch = np.int(nQuad * (qShape[0] / cor) * (qShape[1] / sag) * (qShape[2] / axi))
    # print(nPatch)
    MR_patch = np.zeros([nPatch, cor, sag, axi]).astype(np.float32)
    Mask_patch = np.zeros([nPatch, np.int(cor/2), np.int(sag/2), np.int(axi/2)]).astype(np.int)
    # print
    patch_count = 0
    quad = 0
    for x in c:
        for y in s:
            for z in a:
                MR_quad = MR[x:x + 96, y:y + 128, z:z + 128]
                Mask_quad = Mask[x:x + 96, y:y + 128, z:z + 128]
                quad += 1           
                for k in range(0, MR_quad.shape[0], cor):  # stops when final slice
                    for i in np.arange(0, MR_quad.shape[1], sag):
                        for j in np.arange(0, MR_quad.shape[2], axi):
                            patch = MR_quad[k:k + cor, i:i + sag, j:j + axi]
                            # std_patch.append(np.max(patch))
                            # print(patch.shape, 'here')
                            MR_patch[patch_count, :, :, :] = patch
                            # std_batch.append(np.max(MR_patch))
                            patch = Mask_quad[k+hCor:k + cor-hCor, i+hSag:i + sag-hSag, j+hAxi:j + axi-hAxi]
                            # print('\t', patch.shape, 'here')
                            Mask_patch[patch_count, :, :, :] = patch
                            patch_count += 1
    return MR_patch, Mask_patch, nPatch
```

Try the function here. You can avoid the `Mask` variable if you don’t have one. The function works with loaded MR volume as `NIfTI(.nii)` data. You can use `nibabel` python library for that.

#### About large datasize, 3D data and patches

https://discuss.pytorch.org/t/about-large-datasize-3d-data-and-patches/112630

I am working on 3D data of 114 images each of dimensions `[180x256x256]`. Since such a large image can not be fed directly to the network, I am using overlapping patches of size `[64x64x64]`. Now there are around 22,000 patches in total for 114 images. which can not be loaded into the `Dataloader` as `cuda` memory runs out. Is there a way to iterate the loading of one image at a time and run patches from the image and go to the next image? N.B. for each image there is a target 3D mask with 9 different labels.

