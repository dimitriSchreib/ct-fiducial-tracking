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
import mcubes

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
        
def convert_mha_to_mesh(mha_file='temp_mesh.mha', output_mesh_file = 'temp_mesh.obj', threshold_value = 2000, odir='', debug=False):
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
    origin = np.array([image_3D.GetOrigin()[2], image_3D.GetOrigin()[1], image_3D.GetOrigin()[0]])
    #KEY POINT: http://simpleitk.org/SimpleITK-Notebooks/01_Image_Basics.html
    #(Z,Y,X) sapcing for numpy. Adding in -1 to fix Z inversion?
    #(X,Y,Z) spacing for simpleitk
    #spacing_array[0]*=-1
    #potential reason why to flip axis of numpy array: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6168008/pdf/nihms-987853.pdf
    #https://simpleitk.readthedocs.io/en/v1.2.4/Documentation/docs/source/fundamentalConcepts.html
    #function: https://simpleitk.readthedocs.io/en/v1.2.4/Documentation/docs/source/fundamentalConcepts.html
    #TransformIndexToPhysicalPoint
    #defining the above function: http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/21_Transforms_and_Resampling.html
    print('image shape in sitk (x,y,z): width {} then height {} then depth {}'.format(image_3D.GetWidth(), image_3D.GetHeight(), image_3D.GetDepth()))
    print(" origin: ", origin)
    # #run marching cubes
    # # convert mha into npdrarray with int type
    ellip_double = sitk.GetArrayFromImage(image_3D)
    print('numpy version shape: (z,y,x) {}'.format(ellip_double.shape))
    extreme_points = [image_3D.TransformIndexToPhysicalPoint((0,0,0)), 
                  #image_3D.TransformIndexToPhysicalPoint((image_3D.GetWidth(),0)),
                  image_3D.TransformIndexToPhysicalPoint((image_3D.GetWidth(),image_3D.GetHeight(),image_3D.GetDepth()))]
    extreme_pixels = [image_3D.GetPixel((256,256,0)), 
                  #image_3D.TransformIndexToPhysicalPoint((image_3D.GetWidth(),0)),
                  image_3D.GetPixel((256,256,image_3D.GetDepth()-1))]
                  #image_3D.TransformIndexToPhysicalPoint((0,image_3D.GetHeight(),0))]
    print('locations of different pixels in image {}'.format(extreme_points))
    print('there math https://discourse.itk.org/t/solved-transformindextophysicalpoint-manually/1031/10')
    print('perhpas mcubes mirrors?')
    print('extreme simple itk pixels: {}'.format(extreme_pixels))
    extreme_pixels = [ellip_double[(0,256,256)], 
                  #image_3D.TransformIndexToPhysicalPoint((image_3D.GetWidth(),0)),
                  ellip_double[(image_3D.GetDepth()-1,256,256)]]
        
    print('extreme simple itk pixels: {}'.format(extreme_pixels)) 
    ellip_double = ellip_double.astype(dtype='i2')
    # if crop_z:
    #     ellip_double = ellip_double[crop_z[0]:crop_z[1],:,:]
    #print("image_stack shape: ",ellip_double.shape)
    verts, triangles = mcubes.marching_cubes(ellip_double, threshold_value)
    # grid = pv.UniformGrid(
    #     dims=ellip_double.shape,
    #     spacing=(spacing_array[0],spacing_array[1],spacing_array[2]),
    #     origin=(origin[0],origin[1],origin[2]),
    #     )
    # mesh = grid.contour([1], ellip_double, method='marching_cubes')
    # mesh.plot(scalars=dist, smooth_shading=True, specular=5, cmap="plasma", show_scalar_bar=False)
    # verts, faces, normals, values = measure.marching_cubes(ellip_double, threshold_value)
    # #respace mesh vertices based on DICOM spacing
    verts = verts * spacing_array
    # #offset verts due to origin
    verts += np.array([image_3D.GetOrigin()[2], image_3D.GetOrigin()[1], image_3D.GetOrigin()[0]])
    #DIFFERENT IDEA! WE ARE IN Z Y X BUT NORMALLY THINGS ARE X Y Z. LETS SWAP.
    print('verts shape: {}'.format(verts.shape))
    print('verts type: {}'.format(type(verts)))
    verts_reordered = np.zeros(np.array(verts).shape)
    verts_reordered[:,0] = verts[:,2]
    verts_reordered[:,1] = verts[:,1]
    verts_reordered[:,2] = verts[:,0]
    verts = copy.deepcopy(verts_reordered)
    mcubes.export_obj(verts, triangles, output_mesh_file)
    # #create mesh object and save to disk
    # cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    # for i, f in enumerate(faces):
    #     for j in range(3):
    #         cube.vectors[i][j] = verts[f[j],:]
    # cube.save(output_mesh_file)
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