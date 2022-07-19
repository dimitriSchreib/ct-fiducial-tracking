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
