import torch as th
import numpy as np
import cv2
from math import ceil, floor

def resize(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Linearly resize an image to a new size.

    Args:
        image (np.ndarray): Image of shape (height, width)
        shape (tuple[int, int]): New image size as (new_height, new_width)

    Returns:
        np.ndarray: Resized image
    """
    return cv2.resize(
        image,
        shape[::-1],
        interpolation = cv2.INTER_LINEAR
    )

def get_bounding_box_from_mask(mask_batch: th.Tensor) -> th.Tensor:
    """Fit a bounding box around each mask in batch, and return the co-ordinates.

    Args:
        mask_batch (th.Tensor): Binary mask of shape (batch_size, height, width)

    Returns:
        th.Tensor: Bounding box coordinates (x1, y1, x2, y2) for each mask in the batch\
    Where x1, x2 are the row indices, and y1, y2 are the column indices
    """
    # get the indices of non-zero elements in the mask
    non_zero_indices = th.nonzero(mask_batch, as_tuple=True)

    # initialize the bounding box coordinates
    batch_size = mask_batch.shape[0]
    bounding_boxes = th.zeros((batch_size, 4), dtype=th.int32)

    # iterate over each mask in the batch
    for i in range(batch_size):
        # get the indices of non-zero elements in the current mask
        indices = non_zero_indices[1][non_zero_indices[0] == i], non_zero_indices[2][non_zero_indices[0] == i]

        # calculate the bounding box coordinates
        if len(indices[0]) > 0:
            x1, y1 = th.min(indices[0]), th.min(indices[1])
            x2, y2 = th.max(indices[0]), th.max(indices[1])
            bounding_boxes[i] = th.tensor([x1, y1, x2, y2])
        else:
            # if the mask is empty, set the bounding box coordinates to 0
            bounding_boxes[i] = th.tensor([0, 0, 0, 0])

    return bounding_boxes

def scale_bounding_box(x1: int, y1: int, x2: int, y2: int, shape_old: tuple[int, int], shape_new: tuple[int, int]) -> tuple[int, int, int, int]:
    """Scale a set of bounding box co-ordinates from an old to a new image shape

    Args:
        x1 (int): Start row index of the bounding box
        y1 (int): Start column index of the bounding box
        x2 (int): End row index of the bounding box
        y2 (int): End column index of the bounding box
        shape_old (tuple[int, int]): Shape of the original image
        shape_new (tuple[int, int]): Shape of the new image

    Returns:
        tuple[int, int, int, int]: Scaled bounding box co-ordinates
    """
    height_old, width_old = shape_old
    height_new, width_new = shape_new
    scale_y = width_new / width_old
    scale_x = height_new / height_old

    x1 = int(x1 * scale_x)
    y1 = int(y1 * scale_y)
    x2 = int(x2 * scale_x)
    y2 = int(y2 * scale_y)

    # Ensure the bounding box coordinates are within the new image bounds
    x1 = max(0, min(x1, height_new - 1))
    y1 = max(0, min(y1, width_new - 1))
    x2 = max(0, min(x2, height_new - 1))
    y2 = max(0, min(y2, width_new - 1))

    return x1, y1, x2, y2

def make_bounding_box_square(x1: int, y1: int, x2: int, y2: int, shape: tuple[int, int], allowance: float = 0.005) -> tuple[int, int, int, int]:
    """Make an existing bounding box square, and ensure it fits on its reference image.

    Args:
        x1 (int): Start row index of the bounding box
        y1 (int): Start column index of the bounding box
        x2 (int): End row index of the bounding box
        y2 (int): End column index of the bounding box
        shape (tuple[int, int]): Shape of the reference image
        allowance (float, optional): Growth of the major dimension in case of segmentation errors. Defaults to 0.005.

    Returns:
        tuple[int, int, int, int]: Squared bounding box co-ordinates
    """
    # subtract 1 to be in index scale
    img_height, img_width = shape
    img_height -= 1
    img_width -= 1

    height = x2 - x1
    width = y2 - y1
    
    # do nothing if already square
    if height == width:
        return x1, y1, x2, y2
    
    # set alias variables for the two cases
    elif height > width:
        w = img_height
        z = img_width
        w1, w2 = x1, x2
        z1, z2 = y1, y2
        ordering = True

    elif height < width:
        w = img_width
        z = img_height
        w1, w2 = y1, y2
        z1, z2 = x1, x2
        ordering = False

    # grow w to the allowance
    growth = allowance * (w2 - w1) / 2
    w1 = floor(w1 - growth)
    w2 = ceil(w2 + growth)

    # match z with w
    growth = ((w2 - w1) - (z2 - z1)) / 2
    z1 = floor(z1 - growth)
    z2 = ceil(z2 + growth)

    # shift bounding box until it fits in the image
    if z1 < 0 and z2 <= z:
        z2 = z2 - z1
        z1 = 0
    elif z1 >= 0 and z2 > z:
        z1 = z1 - (z2 - z)
        z2 = z

    if w1 < 0 and w2 <= w:
        w2 = w2 - w1
        w1 = 0
    elif w1 >= 0 and w2 > w:
        w1 = w1 - (w2 - w)
        w2 = w

    # if the bounding box dimensions are larger than the image then downscale the whole box
    if (w2 - w1) > w:
        diminish = ((w2 - w1) - w) / 2
        w1 = 0
        w2 = w
        z1 = ceil(z1 + diminish)
        z2 = floor(z2 - diminish)
    
    if (z2 - z1) > z:
        diminish = ((z2 - z1) - z) / 2
        z1 = 0
        z2 = z
        w1 = ceil(w1 + diminish)
        w2 = floor(w2 - diminish)

    # if from rounding there are some extra pixels
    extra_pixels = abs((w2 - w1) - (z2 - z1))
    # first make it so that there are an even number of extra pixels
    if extra_pixels % 2 == 1:
        if (w2 - w1) > (z2 - z1):
            w1 += 1
        else:
            z1 += 1

    # then remove the extra pixels on either side of the bounding box
    if extra_pixels % 2 == 0:
        extra_pixels_div_2 = int(extra_pixels / 2)
        if (w2 - w1) > (z2 - z1):
            w1 += extra_pixels_div_2
            w2 -= extra_pixels_div_2
        else:
            z1 += extra_pixels_div_2
            z2 -= extra_pixels_div_2

    if ordering:
        return w1, z1, w2, z2
    else:
        return z1, w1, z2, w2