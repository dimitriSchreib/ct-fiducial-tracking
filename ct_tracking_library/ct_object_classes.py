import numpy as np

class Marker():
    """
    name(string): the name of the marker using
    time(dataetime): the time marker object is created.
    key: the serlerial key for the dicom file.
    T(np.array): transform of the marker.
    error: rmse of the spere picked.
    """
    def __init__(self,time,name,d_key,geometry=None,T=None,error=None):
        self.time = time
        self.name = name
        self.key = d_key
        self.geometry = geometry
        self.T = T
        self.error = error
    
    def __repr__(self):
        return f"Marker({self.name}:{self.time})"

class Robot():
    """
    time(datetime): the time robot object is created.
    m(float):thea to joint_postion.
    theta(radians): the postion of the motor.
    joint_postion(float): motor postion in mm.
    STD_f(float): the standard deviation of the motor on front end. 
    STD_b(float): the standard deviation of the motor on back end. 
    """
    def __init__(self,time):
        self.time = time
        self.theta = None
        self.m = None
        self.joint_postion = None
        self.STD_f = None
        self.STD_b = None

    def __repr__(self):
        return f"Robot({self.time})"

###Helper functions

def append_value(dict_obj, key, value):
    # Check if key exist in dict or not
    if key in dict_obj:
        # Key exist in dict.
        # Check if type of value of key is list or not
        if not isinstance(dict_obj[key], list):
            # If type is not list then make it list
            dict_obj[key] = [dict_obj[key]]
        # Append the value in list
        dict_obj[key].append(value)
    else:
        # As key is not in dict,
        # so, add key-value pair
        dict_obj[key] = value

def Hinv(T):
    assert T.shape == (2, 2)
    r,d = T2t(T)
    T_1 = t2T(np.linalg.inv(r),-np.linalg.inv(r)@d)
    return T_1

def t2T(r,t):
    assert isinstance(r,np.ndarray)
    assert isinstance(t,np.ndarray)
    """
    This function that takes R and t to a homogeneous tansfamred matrix:

    Args: r(numpy.ndarray): rotation matrix
          t(numpy.ndarray): translation vector

    Returns: 4x4 matrix

    """
    return np.array([[r, t], [np.zeros((1,3)), 1]])

def T2t(T):
    assert T.shape == (2, 2)
    return T[0,0],T[0,1]