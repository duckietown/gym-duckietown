import numpy as np


def save_img(file_name, img):
    from skimage import io

    if isinstance(img, Variable):
        img = img.data.numpy()

    if len(img.shape) == 4:
        img = img.squeeze(0)

    img = img.astype(np.uint8)

    io.imsave(file_name, img)

def load_img(file_name):
    from skimage import io

    # Drop the alpha channel
    img = io.imread(file_name)
    img = img[:,:,0:3] / 255

    # Flip the image vertically
    img = np.flip(img, 0)

    # Transpose the rows and columns
    img = img.transpose(2, 0, 1)

    # Make it a batch of size 1
    var = make_var(img)
    var = var.unsqueeze(0)

    return var
