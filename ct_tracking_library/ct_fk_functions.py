import open3d as o3d
import transforms3d as t3d
import numpy as np
import math
import matplotlib.pyplot as plt
from spatialmath import *
from roboticstoolbox import ET as E
from ct_tracking_library.ct_tracking_functions import *
from ct_tracking_library.ct_processing_functions import *

def find_fk(m1,m2,robot,plot=False,debug=False):
    """
    This function finds the Farward transformation of the input Markers with respect to motor joint postion:

    Args:
        m1(Marker): the base marker object.
        m2(Marker): the end effector marker object.
        robot(Robot): the Robot oject saved with the markers.
        plot(bool): to show visualization for the plots aobout the markers.
        debug(bool): wheather or not the user want to see debug info.

    Returns:
        FK in scanner base
        rotional error matrix
        postion error matrix in meter
    """
    Tbase = SE3(m2.T)
    Tee = SE3(m1.T)
    e = E.tx(-0.04009572)*E.ty(0.02163274)*E.tz(-0.01404157)*E.tz()
    e = e*E.Rx(90, 'deg')*E.Ry(-90, 'deg')
    Tfk = SE3(e.eval([(robot.joint_postion-robot.zero_postion)/1000]))
    Tfinal = Tbase.inv()*Tee
    if debug:
        Tfinal.plot(frame='1',color='blue')
        Tfk.plot(frame='2',color='red')
        plt.legend(["tracked EE pose relative to base pose","FKee"])
        print("Postion Error vector(m): ", find_p_error(Tfk.t,Tfinal.t))
        print("Rotational Error: ", calc_needle_ori_error(Tfk.R, Tfinal.R))
    print("Postion Error norm(m): ", np.linalg.norm(find_p_error(Tfk.t,Tfinal.t)))
    print("Rotional Error norm(Euler Angle): ", calc_needle_ori_error(Tfk.R, Tfinal.R))
    return Tbase*Tfk,calc_needle_ori_error(Tfk.R, Tfinal.R),find_p_error(Tfk.t,Tfinal.t)

def find_p_error(p_target,p_current):
    """
    This function calucated the postion difference between target and where the robot is currently are.
    Args:
        p_target(np.array of 1x3 vector): the postion of the target desired
        p_current(np.array of 1x3 vector): the postion of the current robot
    Returns:
        error(np.array of 1x3 vector): the error in x,y,z deirection.
    """
    error = p_target-p_current
    return error

def calc_needle_ori_error(R_target, R_current):
    '''
    calculates axis-angle between target and current rotation matrices for robot EE control
    inputs:
        R_target: [3x3] SO3 rotation matrix numpy array for target rotation matrix
        R_current: [3x3] SO3 rotation matrix numpy array for current rotation matrix
    returns:
        [3x1] axis-angle orientation error where the magnitude is the rotation in radians around the axis
    '''
    target_vec = R_target[:,2]
    current_vec = R_current[:,2]
    normal = np.cross(target_vec, current_vec)
    normal /= np.linalg.norm(normal)
    projected = target_vec @ current_vec / (np.linalg.norm(target_vec)*np.linalg.norm(current_vec))
    angle = -np.arccos(projected).squeeze()
    axis = normal.squeeze()
    return axis*angle

def display_fk(fk,m1,m,debug=False):
    Tee = SE3(m1.T)
    marker = np.array([[-10,-5,0],[-10,5,0],[0,-5,0],[10,0,0]])
    R = np.eye(3)
    marker_3d_tracked = create_marker_visualization(marker, [0.2, 0.2, 0.8], Tee.R, Tee.t.reshape((3,1))*1000) #blue color
    marker_3d_transformed = create_marker_visualization(marker, [0.8, 0.8, 0.0], fk.R, fk.t.reshape((3,1))*1000) #yellow color
    marker_3d_transformed.append(create_coordinate_frame_visualization(fk.R, fk.t.reshape((3,1))*1000))
    #visualize tracked markers on top of mesh used for tracking
    visualization_list = marker_3d_tracked+marker_3d_transformed + [m]
    o3d.visualization.draw_geometries(visualization_list)

def display_error(data_load,FK_error=None):    
    Me_list = []
    Fe_list = []
    TT_list = []
    for x in data_load:
            Me_list.append(data_load[x][0].error)
            Fe_list.append(data_load[x][1].error)
            TT_list.append(data_load[x][0].time)
    plt.plot(TT_list,Me_list)
    plt.plot(TT_list,Fe_list)
    plt.ylabel('Error in mm')
    plt.xlabel('Time')
    plt.legend(["moving marker error","fixed marker error"])
    plt.title("Time vs. Marker Error")
    plt.show()
    if FK_error is not None:
        plt.figure()
        plt.plot(FK_error)
        plt.title("Error norm")
        plt.xlabel("scan #")
        plt.ylabel("error in m")
        print("Error STD value: ", np.std(FK_error))

def isR(R):
    Rt = np.transpose(R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - np.dot(Rt, R))
    return n < 1e-6

def R2E(R):
    assert(isR(R))
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])