#general imports
import copy
import os
import pickle
from tqdm import tqdm

#clustering and optimization
from scipy import linalg
from skimage import measure
import itertools

#3D data processing
import numpy as np
import open3d as o3d
import transforms3d as t3d

#visualization
import matplotlib.pyplot as plt #for colors to color mesh

#our functions
from .ct_processing_functions import *

def visualize_tracked_marker(marker, final_R, final_t, permuted_centroids):
    ''' 
    create 3D mesh with coordinate frame from marker geometry coordinates using O3D both base and transformed
    input:
        marker
        R
        t
    returns:
        marker_3d_base
        marker_3d_transformed
        err
    '''
    
    R =  t3d.euler.euler2mat(np.pi/4+0.1, 0, -np.pi/6-0.2)@t3d.euler.euler2mat(0,0.6,0)# @ t3d.euler.euler2mat(0, np.pi/8, 0)
    marker = (R.T@marker.T).T

    marker_3d_base = []
    for i in range(marker.shape[0]):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=3.0, resolution=20).translate(marker[i,:]).paint_uniform_color([0., 0., 0.8])
        marker_3d_base.append(copy.deepcopy(sphere))

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0)#.rotate(R, center=(0, 0, 0))#mean_centroid_coordinates)
    marker_3d_base.append(coordinate_frame)

    #https://github.com/dangeo314/ct_imaging_library/blob/main/src/SVD_marker_class.ipynb
    transformed_marker = final_R @ marker.T + final_t #transforms the marker, second input to the permuted centroids from first input
    # print(permuted_centroids.T[:,:3])
    # print(transformed_marker.T)

    err = permuted_centroids.T[:,:3]-transformed_marker.T
    
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0)#.rotate(R, center=(0, 0, 0))#mean_centroid_coordinates)
    coordinate_frame_transformed = copy.deepcopy(coordinate_frame).rotate(final_R, center=(0, 0, 0)).translate(final_t)

    marker_3d_transformed = []
    for marker in marker_3d_base:
        marker_3d_transformed.append(copy.deepcopy(marker).rotate(final_R, center=(0, 0, 0)).paint_uniform_color([0.8, 0., 0.]).translate(final_t))

    return marker_3d_base, marker_3d_transformed, err

def find_best_transform_from_candidate_marker_clusters(marker, good_centroid_clusters):
    '''
    find best (lowest reconstruction error) marker set in each rigid body (if multiple)
    inputs: 
        marker
        good_centroid_clusters
    returns:
        final_R
        final_t
        permuted_centroids
        err
    '''
    num_markers = marker.shape[0]

    R_list = []
    t_list = []
    permuted_centroids_list = []
    error_list = []
    for segmented_marker in good_centroid_clusters:
        #segmented_marker = good_centroid_clusters[]

        point_combinations = list(itertools.combinations(list(range(segmented_marker.shape[0])), num_markers))
        #print(len(point_combinations))
        for i in range(len(point_combinations)):
            #print('point combinations: '.format(point_combinations[i]))
            sampled_marker = segmented_marker[point_combinations[i],:]
            #print('sampled marker: {}'.format(sampled_marker))
            final_R, final_t, permuted_centroids, error = calculate_transform(num_markers, sampled_marker, marker.T)
            R_list.append(final_R)
            t_list.append(final_t)
            permuted_centroids_list.append(permuted_centroids)
            error_list.append(error)

    error_list = np.array(error_list)
    print(error_list)
    #print(R_list)
    #print(t_list)
    #print(permuted_centroids_list)
    min_error = np.amin(error_list)
    min_error_index = np.argmin(error_list)
    max_error = np.amax(error_list)
    max_error_index = np.argmax(error_list)
    #rint(min_error_index)
    final_R = R_list[min_error_index]
    final_t = t_list[min_error_index]
    permuted_centroids = permuted_centroids_list[min_error_index]

    if min_error < 10:
        print("Everything looks good!")
        print("the final error is: ",min_error)
    else:
        print("Hmm something doesn't look right ...")

    return final_R, final_t, permuted_centroids, min_error

def calculate_transform(num_markers,centroids,marker_geometry, verbose=False):
    '''
    function to find the transfrom info of the input assuming correct centroids but without correspondance being solved
    input:
        num_markers: number of markers in the target rigid body (float)
        centroids: centroid locations of rigid body found in image (Nx3, numpy, millimeters)
        marker_geometry: centroid locations of rigid body manufacturered (Nx3, numpy, millimeters)
    return:
        final_R: solved rotation matrix
        final_t: solved translation vector
        permuted_centroids: new centroid locations permuted for minimum transform error
        min_error: rmse reconstruction error
    '''

    centroids = centroids.copy()
    marker_geometry = marker_geometry.copy()
    if verbose:
        print(centroids.shape)
    error = []
    R_list = []
    t_list = []
    permuted_centroids_list = []
    centroids_SE3 = np.vstack((centroids.T, np.ones(num_markers)))
    point_permutations = np.array(list(itertools.permutations(list(range(num_markers)),num_markers)))
    for i in (range(point_permutations.shape[0])):#keep this line
        permuted_centroids = centroids_SE3[:, point_permutations[i]]
        permuted_centroids_list.append(permuted_centroids.copy())
        ret_R, ret_t = rigid_transform_3D(marker_geometry, permuted_centroids[:3,:])
        B2 = (ret_R@marker_geometry) + ret_t
        err = B2 - permuted_centroids[:3,:]
        err = err * err
        err = np.sum(err)
        rmse = np.sqrt(err/len(marker_geometry[0]))
        R_list.append(ret_R)
        t_list.append(ret_t)
        error.append(rmse)
    error = np.array(error)
    min_error = np.amin(error)
    min_error_index = np.argmin(error)
    max_error = np.amax(error)
    max_error_index = np.argmax(error)
    #rint(min_error_index)
    final_R = R_list[min_error_index]
    final_t = t_list[min_error_index]
    permuted_centroids = permuted_centroids_list[min_error_index]

    if verbose:
        if min_error < 10:
            print("Everything looks good!")
            print("the final error is: ",min_error)
        else:
            print("Hmm something doesn't look right ...")
    return final_R,final_t, permuted_centroids, min_error

def find_candidate_centroids(target_marker, input_mesh_file = 'temp_mesh.stl', debug=False):
    '''
    finds candidate centroids and many other things...
    input:
        target_marker: name for target marker, brittle for how code and indexing is currently done
        input_mesh_file: mesh of environment including sepparated spheres used for fiducial trackcing of rigid bodies
    return:
    many things...
    '''
    marker = np.load('./test_data/'+target_marker+'.npy')
    R =  t3d.euler.euler2mat(np.pi/4+0.1, 0, -np.pi/6-0.2)@t3d.euler.euler2mat(0,0.6,0)# @ t3d.euler.euler2mat(0, np.pi/8, 0)
    marker = (R.T@marker.T).T

    mesh = o3d.io.read_triangle_mesh(input_mesh_file)
    mesh.compute_vertex_normals()
    if debug:
        print("Testing IO for meshes ...")
        o3d.visualization.draw_geometries([mesh])

    points = np.array(mesh.vertices)
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points)
    pcd2.estimate_normals()
    if debug:
        o3d.visualization.draw_geometries([pcd2])

    pcd2 = pcd2.voxel_down_sample(voxel_size = 0.8)
    
    #filter points with further than average distance to neighbors
    cl, ind = pcd2.remove_statistical_outlier(nb_neighbors=10,
                                                    std_ratio=0.8) 
    inlier_cloud = pcd2.select_by_index(ind)
    outlier_cloud = pcd2.select_by_index(ind, invert=True)
    if debug:
        o3d.visualization.draw_geometries([inlier_cloud])
    
    #cluster pointcloud
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
        inlier_cloud.cluster_dbscan(eps=2.5, min_points=10, print_progress=True))
    if debug:
        print(labels.max() )
    
    max_label = labels.max()  

    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    inlier_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
    inlier_cloud.estimate_normals()
    if debug:
        print(f"point cloud has {max_label + 1} clusters")
        o3d.visualization.draw_geometries([inlier_cloud])
    


    # find balls good fit quiality and are the right size
    good_inds = []
    centroids = []
    pointcloud_ball_clusters = []
    import tqdm

    for i in tqdm.tqdm(range(labels.max()+1)):
        selected_indices = np.where(labels==i)
        pcd_selected = inlier_cloud.select_by_index(selected_indices[0])
        points = np.array(pcd_selected.points)

        if points.shape[0] < 1000 and points.shape[0] > 20:
            correctX = points[:,0]
            correctY = points[:,1]
            correctZ = points[:,2]

            r, x0, y0, z0, residules = sphereFit(correctX,correctY,correctZ)
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x=np.cos(u)*np.sin(v)*r
            y=np.sin(u)*np.sin(v)*r
            z=np.cos(v)*r
            x = x + x0
            y = y + y0
            z = z + z0

            #See how good this sphere fit is
            #print(r, x0, y0, z0, residules)
            errors = []
            for j in range(len(correctX)):
                radius_point = ((correctX[j]-x0)**2+(correctY[j]-y0)**2+(correctZ[j]-z0)**2)**0.5
                error = np.abs(radius_point-r)
                errors.append(error)
            errors = np.array(errors)
            mean_error = np.mean(errors)

            #print('Mean Error: {}'.format(np.mean(errors)))
            #print('Radius: {}'.format(r))
            #good_inds.append(i)

            if target_marker == 'marker1':
                r_target = 2.8
            if target_marker == 'marker2':
                r_target = 3.4
                
            diameter_tolerance = 0.5
            sphere_fit_rmse_tolerance = 0.4
            if mean_error < sphere_fit_rmse_tolerance and r > r_target - diameter_tolerance and r < r_target + diameter_tolerance:
                if debug:
                    print('Mean Error: {}'.format(mean_error))
                    print('Radius: {}'.format(r))
                    print('index: {}'.format(i))
                centroids.append(np.array([x0,y0,z0]))
                good_inds.append(i)
                pointcloud_ball_clusters.append(pcd_selected)


        if debug:
            print(good_inds)
            print(target_marker)
            print('display pointcloud clusters that are a good fit to our target sphere diameter')
            o3d.visualization.draw_geometries(pointcloud_ball_clusters)

    
    centroid_clusters, pcd_centroid_clusters, o3d_selected_cluster_inds = find_centroid_clusters(centroids,good_inds)

    if debug:
        print('pointcloud centroid clusters: {}'.format(pcd_centroid_clusters))
        print('centroid clusters: {}'.format(centroid_clusters))
        print(o3d_selected_cluster_inds)



    
    #find clusters of balls that are close enough to eachother based on the marker geometry
    good_centroids = o3d_selected_cluster_inds[0]
    selected_indices = np.where(labels==good_centroids[0])
    pcd_selected = inlier_cloud.select_by_index(selected_indices[0])
    
    
    good_centroid_clusters = [centroid_cluster for centroid_cluster in centroid_clusters if len(centroid_cluster) >= marker.shape[0]]
    if debug:
        print('centroid clustrs: {}'.format(centroid_clusters))
        print('MEEE good centroid clusters: {}'.format(good_centroid_clusters))

    for good_centroids in o3d_selected_cluster_inds: #changed 6/22/2022
        if debug:
            print('Good centroids: {}'.format(good_centroids))
        if len(good_centroids) > 0:
            print('Multiple good clusters/centroids found')
            for i in range(len(good_centroids)): #removed -1 condition!!!!!!!!!! August 16th 2021, not tested with old code...
                selected_indices = np.where(labels==good_centroids[i])
                if debug:
                    print('adding cluster: {}'.format(i))
                pcd_selected += inlier_cloud.select_by_index(selected_indices[0]) #this adds the full cluster of points. index 0 is to get rid of the list wrapper

    pcd_selected.paint_uniform_color([0.8, 0.0, 0.8])

    if debug:
        o3d.visualization.draw_geometries([pcd_selected])


    if debug:
        print("Downsample the point cloud with a voxel of 0.05")
    marker_centroid_coordinates_list = []
    marker_centroid_coordinates = []

    for i in range(len(good_centroid_clusters)):
        local_marker_coordinates = good_centroid_clusters[0][i]# - base_centroid
        marker_centroid_coordinates.append(local_marker_coordinates)

    marker_centroid_coordinates = np.array(marker_centroid_coordinates).squeeze()
    if debug:
        print('centroid coordinates: {}'.format(marker_centroid_coordinates))
    mean_centroid_coordinates = np.mean(np.array(pcd_selected.points), axis=0)
    
    if debug:
        print('mean centroid coodinates: {}'.format(mean_centroid_coordinates))
        print('marker_centroid_coordinates: {}'.format(marker_centroid_coordinates))
        print('marker_centroid_coordinates_list: {}'.format(marker_centroid_coordinates_list))

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=marker.mean(axis=0))#mean_centroid_coordinates)

    if debug:
        print('marker: {}'.format(marker))
    
    return copy.deepcopy(marker), copy.deepcopy(marker_centroid_coordinates), copy.deepcopy(pcd_selected), copy.deepcopy(mesh), copy.deepcopy(coordinate_frame), np.array(good_centroid_clusters)#, np.array(marker_centroid_coordinates_list)



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