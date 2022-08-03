#import all the needed library
import asyncio
import math
import moteus
import seaborn as sns

async def motor_zero(c,stop_torque=.299,v=1,torque=0.3,test=False):
    """
    This function setup the motor to "zero":

    Args:
        stop_torque (float): the therhold torque that tells when the motor should stop.
        v (float): the verlocity for the motor to move in rad/s.
        torque (float): the maximum torque for the motor.

    Returns:
        float: the "zero" position in radians

    """
    print("begin zeroing")
    state = await c.set_position(position=math.nan, query=True) #get system state without moving motor
    if not test:
        print("state info: ",state)
        print("Position:", state.values[moteus.Register.POSITION])
        print("Torque:", state.values[moteus.Register.TORQUE])
        print()
    await asyncio.sleep(0.02) # wait for the motor to change state
    while True:
        state = await c.set_position(position=math.nan, velocity = -v, maximum_torque= torque,query=True)
        #print("Position:", state.values[moteus.Register.POSITION])
        #print("Torque:", state.values[moteus.Register.TORQUE])
        #print()
        await asyncio.sleep(0.001) # wait for spped command to reach the motor
        toq = state.values[moteus.Register.TORQUE]
        if abs(toq) >= stop_torque:
            #print("Position:", state.values[moteus.Register.POSITION])
            #print("Torque:", state.values[moteus.Register.TORQUE])
            #await c.set_stop()
            if not test:
                print("motor is ready")
            p = state.values[moteus.Register.POSITION]
            break
    if test:
        for x in range(700):
            await c.set_position(position=math.nan, velocity = v, maximum_torque= torque,query=True)
    print("Initial Starting Degree: ","{:.2f}".format(math.degrees(p)))
    return p

async def motor_home(c,stop_torque=.299,v=0.2,torque=0.3):
    """
    This function setup the motor to "home":

    Args:
        c: the moteus motor object
        stop_torque (float): the therhold torque that tells when the motor should stop.
        v (float): the verlocity for the motor to move in rad/s.
        torque (float): the maximum torque for the motor.

    Returns:
        float: the "home" position in radians

    """
    print("begin homing")
    #p1 = await motor_zero(c,v=-v)
    #p2 = await motor_zero(c,v=v,stop_torque=.31)
    while True:
        state = await c.set_position(position=math.nan, velocity = -v, maximum_torque= torque,query=True)
        await asyncio.sleep(0.001) # wait for spped command to reach the motor 
        if abs(state.values[moteus.Register.TORQUE]) > stop_torque:
            await c.set_stop()
            print("reach first limit")
            p1 = state.values[moteus.Register.POSITION]
            break
    while True:
        state = await c.set_position(position=math.nan, velocity = v, maximum_torque= torque,query=True)
        await asyncio.sleep(0.001) # wait for spped command to reach the motor 
        if state.values[moteus.Register.TORQUE] > stop_torque:
            await c.set_stop()
            print("reach secound limit")
            p2 = state.values[moteus.Register.POSITION]
            break
    m = (p1+p2)/2
    # np.sign*(m-p2) ?
    while True:
        state = await c.set_position(position=m, velocity = -v, maximum_torque=torque, query=True)
        await asyncio.sleep(0.001) # wait for spped command to reach the motor 
        p = state.values[moteus.Register.POSITION]
        if p <= m:
            print("homing")
            break
    #await c.set_stop()
    print("motor is ready")
    print("Initial Starting Degree: ","{:.2f}".format(math.degrees(p)))
    return p

async def one_axis_farward(c,intial_position,user_degree=0,stop_torque=.299,v=1,torque=0.3):
    """
    This function runs the motor farward:

    Args:
        c: the moteus motor object
        intial_position(float): zeroing postion from zero method
        user_degree(int): the desired angle user wants to moved
        stop_torque (float): the therhold torque that tells when the motor should stop.
        v (float): the verlocity for the motor to move in rad/s.
        torque (float): the maximum torque for the motor.
    Returns:
        p_list(list): a list of the postion that the motor moved.
    """
    p = intial_position
    p_list = []
    ud = math.radians(user_degree)/((2*math.pi))
    print("Begain Testing")
    while True:
        state = await c.set_position(position=math.nan+p+ud, velocity = v, maximum_torque=torque, query=True)
        await asyncio.sleep(0.001)
        if state.values[moteus.Register.TORQUE] > stop_torque:
            print("stoped")
            p_list.append(state.values[moteus.Register.POSITION]*(2*math.pi))
            break
        else:
            p_list.append(state.values[moteus.Register.POSITION]*(2*math.pi))
    await c.set_stop()
    print("Finish")
    return p_list

async def one_axis_backward(c,intial_position,user_degree=0,stop_torque=.299,v=1,torque=0.3):
    """
    This function runs the motor backward:

    Args:
        c: the moteus motor object
        intial_position(float): zeroing postion from zero method
        user_degree(int): the desired angle user wants to moved
        stop_torque (float): the therhold torque that tells when the motor should stop.
        v (float): the verlocity for the motor to move in rad/s.
        torque (float): the maximum torque for the motor.
    Returns:
        p_list(list): a list of the postion that the motor moved.
    """
    p = intial_position
    p_list = []
    ud = math.radians(user_degree)/((2*math.pi))
    print("car coming back")
    while True:
        state = await c.set_position(position=math.nan-(p+ud), velocity = -v, maximum_torque=torque, query=True)
        await asyncio.sleep(0.001)
        if state.values[moteus.Register.TORQUE] > stop_torque:
            print("stoped")
            break
        else:
            p_list.append(state.values[moteus.Register.POSITION]*(2*math.pi))
    await c.set_stop()
    print("Finish")
    return p_list

async def read_p(stop_torque=.299,v=0,torque=0):
    """
    This function stops the motor and returns the postion value of the motor:

    Args:

    Returns:
        p: the postion in revolution in the motor
    """
    print("Begain Testing")
    c = moteus.Controller()
    await c.set_stop()
    state = await c.set_position(position=math.nan, velocity = v, maximum_torque=torque, query=True)
    await asyncio.sleep(0.001)
    p = state.values[moteus.Register.POSITION]
    await c.set_stop()
    print("Position:", p)
    print("Position in Degree: ","{:.2f}".format(math.degrees(p)))
    return p
async def stopping_acc_test(c,n_loop):
    """
    This function stops the motor farward and backward for testing accuray of the stopping and shows a box plot:

    Args:
        c: motuoes motor object
        n_loop(int): the number of time for testing
    Returns:
        p: the postion in revolution in the motor
    """
    f, axes = plt.subplots(1, 2)
    position_setpoint_list_f = []
    position_setpoint_list_b = []
    save_time = []
    for i in range(n_loop):
        position_setpoint_list_f.append(math.degrees((await motor_zero(c,v=-1,test=True)*(2*math.pi))))
    for i in range(n_loop):
        position_setpoint_list_b.append(math.degrees((await motor_zero(c,stop_torque=.1,v=1,test=True)*(2*math.pi))))
    sns.boxplot(data=position_setpoint_list_f,  orient='v', ax=axes[0]).set(title='Forward')
    sns.boxplot(data=position_setpoint_list_b,  orient='v' , ax=axes[1]).set(title='Backward')
    f.suptitle('Stopping Accuracy Test Plot')
    f.text(0.5, 0.04, 'Group', ha='center')
    f.text(0.04, 0.5, 'degree', va='center', rotation='vertical')