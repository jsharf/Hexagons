from math import sin, cos, sqrt, floor
import matplotlib.pyplot as plt
import numpy.random as rnd
import numpy as np
import png

PI = 3.1415926535

Xdim = 1000
Ydim = 1000

BGR = 20
BGG = 75
BGB = 150

bitmap = [[[BGR, BGG, BGB] for _ in range(Ydim)] for _ in range(Xdim)]

def GaussianArray(radius, stddev):
    arr = [0 for _ in range(radius)]
    for i in range(radius):
        arr[i] = np.exp(-(float(i)**2)/(2.0*(stddev**2)))
    return arr

def blur(radius, stddev):
    global bitmap
    newmap = [[[c for c in pix] for pix in row] for row in bitmap]
    arr = GaussianArray(radius*2, stddev)
    for x in range(Xdim):
        for y in range(Ydim):
            left = max(x - radius, 0)
            right = min(x + radius, Xdim)
            top = max(y - radius, 0)
            bottom = min(y + radius, Ydim)
            total = [0, 0, 0]
            for px in range(left, right):
                for py in range(top, bottom):
                    val = bitmap[px][py] 
                    # calculate x and y parameters for gaussian
                    gradius = sqrt((px - x)**2 + (y - py)**2)
                    total = [total[i] + val[i]*arr[int(abs(gradius))] for i in
                            range(3)]
            newmap[x][y] = [min(int(c), 255) for c in total]
        print("Row {0} done out of {1}\n".format(x, Xdim))
    bitmap = newmap

class Coordinate:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z
    def rotateZ(self, deg):
        rads = deg*PI/180.0
        newZ = self.z
        newY = self.x*sin(rads) + self.y*cos(rads)
        newX = self.x*cos(rads) - self.y*sin(rads)
        return Coordinate(newX, newY, newZ)
    def dot(self, other):
        return self.x*other.x + self.y*other.y + self.z*other.z
    def mag(self):
        return sqrt(self.x**2 + self.y**2 + self.z**2)
    def mult(self, c):
        return Coordinate(self.x*c, self.y*c, self.z*c)
    def div(self, c):
        return self.mult(1.0/c)
    def norm(self):
        magnitude = self.mag()
        if (magnitude == 0):
            return Coordinate(self.x, self.y, self.z)
        return self.div(magnitude)
    def __add__(self, other):
        newx = self.x + other.x
        newy = self.y + other.y
        newz = self.z + other.z
        return Coordinate(newx, newy, newz)
    def proj(self, other):
        dp = self.dot(other.norm())
        return other.norm().mult(dp)
    def __repr__(self):
        return "(" + str(self.x) + ", " + str(self.y) + ")"

center = Coordinate(Xdim/2.0, Ydim/2.0, 0)

def switchToOctantZeroFrom(octant, x, y):
    return [(x, y), (y, x), (y, -x), 
            (-x, y), (-x, -y), (-y, -x), (-y, x),
            (x, -y)][octant]

def switchToOctantFromZero(octant, x, y):
    return [(x, y), (y, x), (-y, x), 
            (-x, y), (-x, -y), (-y, -x), (y, -x),
            (x, -y)][octant]

def sign(x):
    if (x >= 0):
        return 1
    else:
        return -1

def octant(x, y):
    qd = {
            (-1, -1) : 4 if (y > x) else 5,
            (-1, 1)  : 2 if (y > abs(x)) else 3,
            (1, -1)  : 6 if (abs(y) > x) else 7,
            (1, 1)   : 1 if (y > x) else 0,
        }
    ret = qd[(sign(x), sign(y))]
    return ret


def zeroLine(zero, xp, yp):
    # convert to octant zero for Bresenham's line algorithm
    (xo, yo) = (xp - zero[0], yp - zero[1])
    o = octant(xo, yo)
    (x, y) = switchToOctantZeroFrom(o, xo, yo)
    dx = float(x)
    dy = float(y)
    error = float(0)
    derr = abs(float(dy/dx))
    y = int(0)
    for xi in range(0, int(x)):
        (px, py) = switchToOctantFromZero(o, xi, y)
        bitmap[int(zero[0] + px)][int(zero[1] + py)] = [255, 255, 255]
        error += derr
        while (error >= 0.5):
            (px, py) = switchToOctantFromZero(o, xi, y)
            bitmap[int(zero[0] + px)][int(zero[1] + py)] = [255, 255, 255]
            y += sign(dy)
            error -= 1.0

def ipart(x):
    return int(x)

def round(x):
    return ipart(x + 0.5)

def fpart(x):
    if x < 0:
        return 1 - (x - floor(x))
    return x - floor(x)

def rfpart(x):
    return 1 - fpart(x)

def plot(x, y, c):
    current = bitmap[x][y]
    remaining = [(255 - v) for v in current]
    increments = [int(c*float(remain)) for remain in remaining]
    bitmap[x][y] = [cur + incr for (cur, incr) in zip(current, increments)]

def constrain(x, minimum, maximum):
    if (x < minimum):
        return minimum
    if (x > maximum):
        return maximum
    return x

def zeroPlotInOct(zero, octant, x, y, c):
    (xo, yo) = switchToOctantFromZero(octant, x, y)
    (xb, yb) = (constrain(xo + zero[0], 0, Xdim-1), constrain(yo + zero[1], 0,
        Ydim-1))
    plot(int(xb), int(yb), c)

def xiaolinZeroLine(zero, xp, yp):
    (xo, yo) = (xp - zero[0], yp - zero[1])
    o = octant(xo, yo)
    (x, y) = switchToOctantZeroFrom(o, xo, yo)
    dy = int(y)
    dx = int(x)
    gradient = float(dy)/float(dx)
    # first endpoint
    xend  = 0
    yend  = 0
    xgap  = rfpart(0.5)
    xpx11 = xend
    ypx11 = ipart(yend)
    zeroPlotInOct(zero, o, xpx11, ypx11, rfpart(yend) * xgap)
    zeroPlotInOct(zero, o, xpx11, ypx11+1, fpart(yend) * xgap)
    intery = yend + gradient
    # second endpoint
    xend  = round(x)
    yend  = y + gradient * (xend - x)
    xgap  = fpart(x + 0.5)
    xpx12 = xend
    ypx12 = ipart(yend)
    zeroPlotInOct(zero, o, xpx12, ypx12, rfpart(yend) * xgap)
    zeroPlotInOct(zero, o, xpx12, ypx12 + 1, fpart(yend) * xgap)
    for x in range(xpx11 + 1, xpx12 - 1):
        zeroPlotInOct(zero, o, x, ipart(intery), rfpart(intery)) 
        zeroPlotInOct(zero, o, x, ipart(intery) + 1, fpart(intery)) 
        intery += gradient

def line(x0, y0, x1, y1):
    xiaolinZeroLine((y0, x0), y1, x1)

def draw(p1, p2):
    line(p1.x + center.x, p1.y + center.y, p2.x + center.x, p2.y + center.y)

def line_center(x0, y0, x1, y1):
    line(x0 + center.x, y0 + center.y, x1 + center.x, y1 + center.y)

def render():
    resized = [[0 for _ in range(3*Ydim)] for _ in range(Xdim)]
    for i in range(Xdim):
        for j in range(Ydim):
            resized[i][j*3 + 0] = bitmap[i][j][0]
            resized[i][j*3 + 1] = bitmap[i][j][1]
            resized[i][j*3 + 2] = bitmap[i][j][2]
    pic = png.from_array(resized, mode='RGB;8')
    return pic

class Cube:
    Directions = {
        'RU' : Coordinate(-1, 0, 1),
        'LU' : Coordinate(0, -1, 1),
        'R': Coordinate(-1, 1, 0),
        'L': Coordinate(1, -1, 0),
        'RD' : Coordinate(0, 1, -1),
        'LD' : Coordinate(1, 0, -1),
    }
    def __init__(self, coord, grid=1):
        self.loc = Coordinate(coord.x, coord.y, coord.z)
        self.loc.z = -(self.loc.x + self.loc.y)
        self.gridSize = grid
    def getNeighbor(self, direction):
        center = self.loc + Cube.Directions[direction]
        return Cube(center, self.gridSize)
    def hexagon(self):
        height = 2.0*self.gridSize
        basisr = Coordinate(sqrt(3.0)/2.0, 0, 0).mult(height)
        basisq = Coordinate(sqrt(3.0)/4.0, -0.75, 0).mult(height)
        center = basisr.mult(self.loc.y) + basisq.mult(self.loc.z)
        edge = self.gridSize
        return Hexagon(center.x, center.y, edge)

class Hexagon:
    def __init__(self, x=0, y=0, edge=1):
        self.center = Coordinate(x, y, 0)
        self.edge = edge
    # corner #0 is at the top, counts up CW
    def getCorner(self, i):
        d = Coordinate(0, 1, 0).norm().mult(self.edge)
        d = d.rotateZ(60*i)
        return d + self.center
    def draw(self):
        for i in range(6):
            p1 = self.getCorner(i)
            p2 = self.getCorner(i + 1)
            draw(p1, p2)    
    def cube(self):
        self.center.x
        y = (-self.center.x * sqrt(3.0) / 3 + self.center.y / 3.0)/self.edge
        z = (-self.center.y * 2.0/3)/self.edge
        x = -(y + z)
        return Cube(Coordinate(x, y), self.edge)
        

def Column(x, edge):
    nHexagons = int(Ydim/(1.5*edge)) - 1
    Top = Hexagon(x, -Ydim/2 + edge, edge)
    Top.draw()
    chain = Top.cube().getNeighbor('RD')
    for _ in range(nHexagons/2):
        chain.hexagon().draw()
        chain = chain.getNeighbor('LD')
        chain.hexagon().draw()
        chain = chain.getNeighbor('RD')

A = Hexagon(0, 0, 100)
B = A.cube()
E = Cube(Coordinate(B.loc.x - 0.353, B.loc.y - 0.353, B.loc.z), 100)
C = Cube(Coordinate(E.loc.x, E.loc.y + 1, E.loc.z), 100)
D = Cube(Coordinate(E.loc.x + 1, E.loc.y, E.loc.z), 100)
C.hexagon().draw()
D.hexagon().draw()
E.hexagon().draw()
A.draw()


pic = render()
pic.save('output.png')

