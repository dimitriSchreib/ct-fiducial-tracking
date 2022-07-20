#general imports
import pickle
import numpy as np

#mesh processing
from skimage import measure
from stl import mesh

def convert_scan_to_mesh(scan_file, output_mesh_file = 'temp_mesh.stl', threshold_value = 1200):
    '''
    input:
        scan_file: pickled data containing 
            image stack (numpy array, 3xN, Hounsfield Units), 
            slice_spacing: z axis spacing of the CT scan data (float, millimeters),
            spacing_x: x axis pixel spacing of the CT scan data (float, millimeters),
            spacing_y: y axis pixel spacing of the CT scan data (float, millimeters),
        output_mesh_file: file name for storing the output thresholded mesh from marching cubes
        threshold_value: threshold value for marching cubes (float, Hounsfield Units)
    return:
    '''
    with open(scan_file, 'rb') as f:
        image_stack, slice_spacing, spacing_x, spacing_y, _, _, _ = pickle.load(f)
    
    spacing_array = np.array([slice_spacing, spacing_x, spacing_y])

    #run marching cubes
    ellip_double = image_stack
    verts, faces, normals, values = measure.marching_cubes(ellip_double, threshold_value)

    #respace mesh vertices based on DICOM spacing
    verts = verts * spacing_array

    #create mesh object and save to disk
    cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            cube.vectors[i][j] = verts[f[j],:]
    cube.save(output_mesh_file)

def convert_scan_to_mesh_mha(scan_file, output_mesh_file = 'temp_mesh.stl', threshold_value = 1200, odir=''):
    '''
    input:
        scan_file: DICOM data containing the sapcing deimesion
        output_mesh_file: file name for storing the output thresholded mesh from marching cubes
        threshold_value: threshold value for marching cubes (float, Hounsfield Units)
    return:
    '''
    OUTPUT_DIR = odir
    original_image = sitk.ReadImage(sitk.ImageSeriesReader_GetGDCMSeriesFileNames(scan_file))
    # Write the image.
    output_file_name_3D = os.path.join(OUTPUT_DIR, 'temp_mesh.mha')
    sitk.WriteImage(original_image, output_file_name_3D)
    
    # convert from SimpleITK to Numpy
    image_3D = sitk.ReadImage('temp_mesh.mha')
    spacing_array = np.array([image_3D.GetSpacing()[2],image_3D.GetSpacing()[1],image_3D.GetSpacing()[0]])
    print("spacing: ",spacing_array)

    #run marching cubes
    # convert mha into npdrarray with int type
    ellip_double = sitk.GetArrayFromImage(image_3D)
    ellip_double = ellip_double.astype(dtype='i2')
    print("image_stack shape: ",ellip_double.shape)
    verts, faces, normals, values = measure.marching_cubes(ellip_double, threshold_value)

    #respace mesh vertices based on DICOM spacing
    verts = verts * spacing_array

    #create mesh object and save to disk
    cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            cube.vectors[i][j] = verts[f[j],:]
    cube.save(output_mesh_file)
