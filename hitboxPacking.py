import Classes.hitboxes as hbx
test = True
'''Imported Hitboxes'''
corner1sImp = "-2728.486;-133.531;-1941.562|-3335.724;-129.954;-5702.128|-3942.962;-126.376;-9462.694|-4550.2;-122.798;-13223.26|-291.991;-484.962;-11775.301"
corner2sImp = "-870.257;169.396;-1583.458|-1477.495;172.974;-5344.024|-2084.733;176.551;-9104.59|-2691.97;180.129;-12865.156|1.51;527.263;-10762.903"

corner1s = hbx.uncodePoints(corner1sImp)
corner2s = hbx.uncodePoints(corner2sImp)

if test:
    import matplotlib.pyplot as plt
    xs = [pt[0] for pt in corner1s]+[pt[0] for pt in corner2s]
    zs = [pt[2] for pt in corner1s]+[pt[2] for pt in corner2s]
    print(xs)
    plt.scatter(xs, zs)
    plt.show()