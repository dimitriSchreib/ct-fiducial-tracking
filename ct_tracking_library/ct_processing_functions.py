#general imports
import pickle
import numpy as np
import copy
import os

#mesh processing
from skimage import measure
from stl import mesh
import open3d as o3d

#Dicom processing
import SimpleITK as sitk


def convert_scan_to_mesh(scan_file, output_mesh_file = 'temp_mesh.stl', threshold_value = 1200, debug=False):
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

    if debug:
        saved_mesh_o3d = o3d.io.read_triangle_mesh(output_mesh_file)
        saved_mesh_o3d.compute_vertex_normals()
        print("Displaying segmented mesh")
        o3d.visualization.draw_geometries([saved_mesh_o3d])

        
def convert_scan_to_mha(scan_file, output_mha_file = 'temp_mesh.mha', odir='', crop_z = None, debug=False):
    '''
    input:
        scan_file: DICOM data containing the sapcing deimesion
        output_mha_file: file name for storing the output mha of DICOM
    return:
    '''
    OUTPUT_DIR = odir
    original_image = sitk.ReadImage(scan_file)
    
    if crop_z:
        original_image=original_image[:,:,crop_z[0]:crop_z[1]]
        
    # Write the image.
    output_file_name_3D = os.path.join(OUTPUT_DIR, output_mha_file)
    sitk.WriteImage(original_image, output_file_name_3D)
        
def convert_mha_to_mesh(mha_file='temp_mesh.mha', output_mesh_file = 'temp_mesh.stl', threshold_value = 2000, odir='', debug=False):
    '''
    input:
        mha_file: DICOM data containing the sapcing deimesion
        output_mesh_file: file name for storing the output thresholded mesh from marching cubes
        threshold_value: threshold value for marching cubes (float, Hounsfield Units)
    return:
    '''
    # convert from SimpleITK to Numpy
    image_3D = sitk.ReadImage('temp_mesh.mha')

    spacing_array = np.array([image_3D.GetSpacing()[2],image_3D.GetSpacing()[1],image_3D.GetSpacing()[0]])
    print("spacing: ",spacing_array)

    #run marching cubes
    # convert mha into npdrarray with int type
    ellip_double = sitk.GetArrayFromImage(image_3D)
    ellip_double = ellip_double.astype(dtype='i2')
    #     if crop_z:
    #         ellip_double = ellip_double[crop_z[0]:crop_z[1],:,:]
    print("image_stack shape: ",ellip_double.shape)
    verts, faces, normals, values = measure.marching_cubes(ellip_double, threshold_value)

    #respace mesh vertices based on DICOM spacing
    verts = verts * spacing_array
    
    #offset verts due to origin
    verts += np.array([image_3D.GetOrigin()[2], image_3D.GetOrigin()[1], image_3D.GetOrigin()[0]])

    #create mesh object and save to disk
    cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            cube.vectors[i][j] = verts[f[j],:]
    cube.save(output_mesh_file)

    if debug:
        saved_mesh_o3d = o3d.io.read_triangle_mesh(output_mesh_file)
        saved_mesh_o3d.compute_vertex_normals()
        print("Displaying segmented mesh")
        o3d.visualization.draw_geometries([saved_mesh_o3d])

def load_and_remesh(input_mesh_path):
    """ Load a mesh from disk and remesh to fix connectivity issues.

    Args:
        input_mesh_path: Mesh file path on disk

    Returns:
        An Open3D mesh with fixed connectivit y

    """
    #load input mesh
    mesh = o3d.io.read_triangle_mesh(input_mesh_path)
    mesh.compute_vertex_normals()

    #convert mesh to pointcloud for remeshing
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.colors = mesh.vertex_colors
    pcd.normals = mesh.vertex_normals

    #downsample pointcloud
    downpcd = pcd.voxel_down_sample(voxel_size=0.5)

    #mesh full downsampled pointcloud
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        output_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            downpcd, depth=9)

    output_mesh.compute_vertex_normals()
    output_mesh.paint_uniform_color([1, 0.706, 0])

    return output_mesh

def get_largest_mesh_cluster(input_mesh):
    """ Finds largest cluster within the mesh and returns this cluster.

    Args:
        input_mesh: input mesh to be clustered and returned. mesh is not
        modified inside the function. mesh should be an open3d mesh object

    Returns:
        An Open3D mesh with only the largest cluster

    """
    
    #cluster connected triangles
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (
            input_mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)

    #keep largest cluster
    output_mesh = copy.deepcopy(input_mesh)
    largest_cluster_idx = cluster_n_triangles.argmax()
    triangles_to_remove = triangle_clusters != largest_cluster_idx
    output_mesh.remove_triangles_by_mask(triangles_to_remove)
    
    return output_mesh