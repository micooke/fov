###
import numpy as np
from pymap3d import ned
from scipy.spatial.transform import Rotation as R
import inspect
import math as m
import matplotlib.pyplot as plt
import shapely.plotting
from shapely.geometry import Polygon

np.set_printoptions(precision=3, suppress=True)

def LinePlaneCollision(rayPoint, rayDirection, planePoint = [0.,0.,0.], planeNormal = np.array([0.,0.,-1.]), epsilon=1e-6):
    ndotu = planeNormal.dot(rayDirection)
    if abs(ndotu) < epsilon:
        print("[ERROR] no intersection or line is within plane")

    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    Psi = w + si * rayDirection + planePoint

    return np.array(Psi)

def sanitise_angles(az, el, degrees:bool=True):
    if degrees == False:
        az = np.degrees(az)
        el = np.degrees(el)
    
    # fix radial wrapping
    if az > 360:
        az %= 360
    if el > 360:
        el %= 360
    
    # 0:360 => -180:180
    if az > 180:
        az -= 360
    if el > 180:
        el -= 360

    # return the result, as radians if specified
    if degrees == False:
        return np.radians((az,el))
    else:
        return np.array([az,el])

# Z = up (-down)
# Y = east
# X = south (-north)
# 
# rotation order: X->Y->Z
#
def rotate3d(ned, theta_deg, axis='z'):
    xyz = ned*[-1,1,-1]
    theta = np.radians(theta_deg)
    
    if axis in ['Z','z']:
        R = np.array([[m.cos(theta), -m.sin(theta), 0],
                      [m.sin(theta),  m.cos(theta), 0],
                      [           0,             0, 1]])
    elif axis in ['Y','y']:
        R = np.array([[ m.cos(theta), 0, m.sin(theta)],
                      [            0, 1,            0],
                      [-m.sin(theta), 0, m.cos(theta)]])
    else:
        R = np.array([[1,            0,            0],
                      [0, m.cos(theta), -m.sin(theta)],
                      [0, m.sin(theta),  m.cos(theta)]])
    
    xyz = R.dot(xyz)

    return xyz*[-1,1,-1]

# az_bw = azimuth, azimuth beamwidth
def create_fov(p0 = [0.,0.,-100.], az_bw_deg=[-45.,30.], el_bw_deg=[-45.,45.]):
    #print(inspect.stack()[0][3])
    
    C = np.array([1., 0., 0.])
    az_m = np.tan(np.radians(az_bw_deg[1]/2))
    el_m = np.tan(np.radians(el_bw_deg[1]/2))
    
    T = np.array([1., 0., -el_m])
    R = np.array([1., az_m,  0.])
    B = np.array([1., 0.,  el_m])
    L = np.array([1., -az_m, 0.])
    
    TL = T+(L-C)
    TR = T+(R-C)
    BR = B+(R-C)
    BL = B+(L-C)

    C = np.array(ned.aer2ned(az_bw_deg[0], el_bw_deg[0], 1.))
    v = {'TL':TL, 'TR':TR, 'BR':BR, 'BL':BL, 'C':C, 'T':T, 'R':R, 'B':B, 'L':L}
    
    for k in v.keys():
        if k == 'C':
            v[k] = LinePlaneCollision(p0, v[k])
        else:
            vec = rotate3d(v[k], el_bw_deg[0], axis='y')
            vec = rotate3d(vec, -az_bw_deg[0], axis='z')
            v[k] = LinePlaneCollision(p0, vec)
        
        #print(f"{k}: {vec_ned[k]} {v[k]}")
    
    outer = [tuple(v[k][:2]) for k in ['TL','TR','BR','BL']]
    inner = [tuple(v[k][:2]) for k in ['T','R','B','L']]
    centre = v['C']

    return (outer,inner,centre)

# az_bw = azimuth, azimuth beamwidth
def old_fov(p0 = [0.,0.,-100.], az_bw_deg=[-45.,30.], el_bw_deg=[-45.,45.]):
    C = np.array([az_bw_deg[0], el_bw_deg[0], 1.])
    
    TL = C + [-az_bw_deg[1]/2,  el_bw_deg[1]/2, 0.]
    TR = C + [ az_bw_deg[1]/2,  el_bw_deg[1]/2, 0.]
    BR = C + [ az_bw_deg[1]/2, -el_bw_deg[1]/2, 0.]
    BL = C + [-az_bw_deg[1]/2, -el_bw_deg[1]/2, 0.]

    v = {'TL':TL, 'TR':TR, 'BR':BR, 'BL':BL, 'C':C}
    C = np.array(ned.aer2ned(*C))
    
    for k in v.keys():
        v[k] = np.array(ned.aer2ned(*v[k]))
        v[k] = LinePlaneCollision(p0, v[k])
    
    outer = [tuple(v[k][:2]) for k in ['TL','TR','BR','BL']]
    centre = v['C']

    return (outer,centre)

def plot_fov(vertices = [(0, 5), (1, 1), (3, 0), (1.5, 3)], name=None, colour = 'tab:blue', ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    # switch n,e to e,n to match x,y
    vertices = [(y,x) for (x,y) in vertices]

    fov_polygon = Polygon(vertices)

    shapely.plotting.plot_polygon(fov_polygon, color=colour, ax=ax, label=name)
    for (idx,xy) in enumerate(vertices):
        ax.annotate(str(idx), xy=xy, ha='center', name=None)
    
    if ax is None:
        if name is not None:
            plt.title(name)
        plt.show()

###
p0=np.array([0., 0., -100.])
az_bw_deg=[  0., 40.]
el_bw_deg=[-45., 40.]
(outer,inner,centre) = create_fov(p0, az_bw_deg=az_bw_deg, el_bw_deg=el_bw_deg)
(old_,_) = old_fov(p0, az_bw_deg=az_bw_deg, el_bw_deg=el_bw_deg)

fig, ax = plt.subplots()
plot_fov(outer, name='outer', colour = 'tab:blue', ax=ax)
plot_fov(inner, name='inner', colour = 'tab:green', ax=ax)
plot_fov(old_, name='old method', colour = 'tab:orange', ax=ax)
ax.scatter(centre[1], centre[0], color='k', s=80, marker='+')
plt.title('FOV estimation')

plt.legend(loc="lower right")
#plt.savefig("FOV_plot.png")
plt.show()
