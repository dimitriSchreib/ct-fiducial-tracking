import SimpleITK as sitk
import numpy as np
from ct_tracking_library import gui
import os

def print_info(selected_image):
    print('origin: ' + str(selected_image.GetOrigin()))
    print('size: ' + str(selected_image.GetSize()))
    print('spacing: ' + str(selected_image.GetSpacing()))
    print('direction: ' + str(selected_image.GetDirection()))
    print('pixel type: ' + str(selected_image.GetPixelIDTypeAsString()))
    print('number of pixel components: ' + str(selected_image.GetNumberOfComponentsPerPixel()))
    print()

def display_Dicom(file1,file2):
    assert isinstance(file1,str)
    assert isinstance(file1,str)
    if type(file1).__module__ == np.__name__:
        img1 = sitk.GetImageFromArray(file1)
    else:
        img1 = sitk.ReadImage(file1)
    if type(file2).__module__ == np.__name__:
        img2 = sitk.GetImageFromArray(file2)
    else:
        img2 = sitk.ReadImage(file2)
    print_info(img1)
    print_info(img2)
    # Obtain foreground masks for the two images using Otsu thresholding, we use these later on.
    msk1 = sitk.OtsuThreshold(img1,0,1)
    msk2 = sitk.OtsuThreshold(img2,0,1)

    gui.MultiImageDisplay(image_list = [img1, img2],
                          title_list = ['Original', 'Estimated'],
                          figure_size=(9,3))