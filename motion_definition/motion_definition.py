#Prasanna code for motion generation

import numpy as np

way_points1=np.array([
    [100,195,1.5],
    [-20,110,1.5]
])

way_points2=np.array([
    [-20, 110, 1.5],
    [-28, 114, 1.5],
    [-36, 117, 1.5],
    [-44, 119, 1.5],
    [-52, 121, 1.5],
    [-60, 123, 1.5],
    [-68, 126, 1.5],
    [-76, 129, 1.5],
    [-84, 132, 1.5],
    [-92, 135, 1.5],
    [-100, 138, 1.5],
    [-108, 141, 1.5],
    [-116, 144, 1.5],
    [-124, 147, 1.5],
    [-132, 150, 1.5],
    [-138, 153, 1.5],
    [-142, 156, 1.5],
    [-144, 158, 1.5],
    [-146, 159, 1.5],
    [-148, 160, 1.5],
    [-148,151,1.5],
    [-155,150,1.5],
    [-160,152,1.5],
    [-172,158,1.5],
    [-185,166,1.5],
    [-200,170,1.5],
    [-213,176,1.5],
    [-220,175,1.6]
])

way_points3=np.array([
    [-220,175,1.6],
    [-262,135,1.6],
    [-262,130,1.6],
    [-264,125,1.6],
    [-273,118,1.6]
])

way_points4=np.array([
    [-273,118,1.6],
    [-360,118,1.6]
])

way_points5=np.array([
    [-360,118,1.6],
    [-360,25,1.6]
])

way_points6=np.array([
    [-360,25,1.6],
    [-280,25,1.6]
])

way_points7=np.array([
    [-280, 25, 1.6],
    [-260, 20, 1.55],
    [-240, 15, 1.5],
    [-220, 9, 1.5],
    [-200, 4, 1.5],
    [-180, -10, 1.5],
    [-180,-25,1.6],
    [-180,-40,1.6],
    [-180,-50,1.6],
    [-180,-60,1.6],
    [-190,-75,1.6],
    [-198,-90,1.6],
    [-205,-115,1.6]
])

way_points8= np.array([
    [-205,-115,1.6],
    [-185,-125,1.6],
    [-165,-130,1.6]
])



def interpolate_path(waypoints, step_size=1.0):
    interpolated = []
    for i in range(len(waypoints) - 1):
        p1 = waypoints[i]
        p2 = waypoints[i + 1]
        vec = p2 - p1
        dist = np.linalg.norm(vec)
        steps = int(np.floor(dist / step_size))
        for j in range(steps):
            point = p1 + vec * (j / steps)
            interpolated.append(point)
    interpolated.append(waypoints[-1])  # include final point
    return np.array(interpolated)


path1=interpolate_path(way_points1,step_size=10)
path2=interpolate_path(way_points2,step_size=3.0)
path3=interpolate_path(way_points3,step_size=5.0)
path4=interpolate_path(way_points4,step_size=10.0)
path5=interpolate_path(way_points5,step_size=10.0)
path6=interpolate_path(way_points6,step_size=10.0)
path7=interpolate_path(way_points7,step_size=3.0)
path8=interpolate_path(way_points8,step_size=5.0)


vehicle_path=np.vstack([path1,path2,path3,path4,path5,path6,path7,path8])