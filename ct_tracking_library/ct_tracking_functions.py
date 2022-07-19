#general imports
import copy
import os
import pickle
from tqdm import tqdm

#visualization
import matplotlib.pyplot as plt
from matplotlib import rcParams

#DICOM loading
import pydicom

#clustering and optimization
from scipy import linalg
from scipy.cluster.vq import kmeans, vq
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_gaussian_quantiles
from scipy.optimize import minimize
from skimage import measure
import itertools

#3D data processing
import numpy as np
import open3d as o3d
import transforms3d as t3d
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

def sphereFit(spX,spY,spZ):
    '''
    fit a least squares sphere to pointcloud data
    input:
    return:
    '''

    #   Assemble the A matrix
    spX = np.array(spX)
    spY = np.array(spY)
    spZ = np.array(spZ)
    A = np.zeros((len(spX),4))
    A[:,0] = spX*2
    A[:,1] = spY*2
    A[:,2] = spZ*2
    A[:,3] = 1

    #   Assemble the f matrix
    f = np.zeros((len(spX),1))
    f[:,0] = (spX*spX) + (spY*spY) + (spZ*spZ)
    C, residules, rank, singval = np.linalg.lstsq(A,f)

    #   solve for the radius
    t = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
    radius = np.sqrt(t)

    return radius, C[0], C[1], C[2], residules

def display_inlier_outlier(cloud, ind, display = False):
    '''
    demo code from open3d
    input: 
        cloud is open3d pointcloud 
        ind is indices downselect pointcloud
        display is flag to draw geometry or not
    return:
        inlier cloud after point downselect
    '''
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    if display:
        print("Showing outliers (red) and inliers (gray): ")
        outlier_cloud.paint_uniform_color([1, 0, 0])
        inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                        zoom=0.3412,
                                        front=[0.4257, -0.2125, -0.8795],
                                        lookat=[2.6172, 2.0475, 1.532],
                                        up=[-0.0694, -0.9768, 0.2024])
    return inlier_cloud

def rigid_transform_3D(A, B):
    '''
    finds least squares transform between point sets with correspondance
    input: expects 3xN np.array of points
    return: R,t
    R = 3x3 rotation matrix
    t = 3x1 column vector
    '''
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        #print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t

def find_centroid_clusters(centroids, good_inds, epsilon=35.0):
    '''
    move centroids into open3d for dbscan clustering to find groups that could be potential markers
    inputs: centroids, np.array of points
    return: 
    list of np.array for centroids clustered based on maximum cluster point-to-point connetectivity distance of epsilon
    o3d version of above as a pcd
    cluster indices based on prior code following this ordering of clusters
    '''

    pcd_centroids = o3d.geometry.PointCloud()
    pcd_centroids.points = o3d.utility.Vector3dVector(centroids)
    #o3d.visualization.draw_geometries([pcd_centroids])

    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            pcd_centroids.cluster_dbscan(eps=epsilon, min_points=1, print_progress=True))



    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd_centroids.colors = o3d.utility.Vector3dVector(colors[:, :3])
    #o3d.visualization.draw_geometries([pcd_centroids],
    #                                  zoom=0.455,
    #                                  front=[-0.4999, -0.1659, -0.8499],
    #                                  lookat=[2.1813, 2.0619, 2.0999],
    #                                  up=[0.1204, -0.9852, 0.1215])

    o3d_cluster_inds = []
    pcd_centroid_clusters = []
    for label in np.unique(labels).tolist():
        selected_indices = np.where(labels==label)
        print(selected_indices[0])
        pcd_selected_centroid = pcd_centroids.select_by_index(selected_indices[0])
        #if len(selected_indices) > 0:
        #    for i in range(len(selected_indices)-1):
        #        pcd_selected_centroid += pcd_centroids.select_by_index(selected_indices[i+1])
        pcd_centroid_clusters.append(copy.deepcopy(pcd_selected_centroid))

        o3d_selected_cluster_inds = [good_inds[i] for i in selected_indices[0].tolist()]
        o3d_cluster_inds.append(o3d_selected_cluster_inds)

    #print(pcd_centroid_clusters)

    centroid_clusters = []
    for pcd in pcd_centroid_clusters:
        centroid_clusters.append(np.array(pcd.points))

    #print(centroid_clusters)

    return centroid_clusters, pcd_centroid_clusters, o3d_cluster_inds