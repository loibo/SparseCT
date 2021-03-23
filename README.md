# SparseCT

data.py contains functions to generate the synthetic dataset of ellipses. Note that those functions gives an output shape of (N, h, w, t) where the latter dimension "t" is the time, i.e. the slices.

Unfortunately, the functions that projects and backprojects the dataset needs the shape to be (N, t, h, w), so please run the command np.transpose(data, (0, 3, 1, 2)) before running the code, after you created the dataset.

To help visualization, it is possible to use functions in visualize.py, where the data is again supposed to be (N, h, w, t).

The data can be save in .npy, .npz, .tif, and the code for AstraToolbox now works only image-per-image.

---

An example on how to use the functions is in the file main.py.
