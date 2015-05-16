from math import sin, cos, sqrt
import matplotlib.pyplot as plt
import numpy.random as rnd
import numpy as np
import png

PI = 3.1415926535

Xdim = 1000
Ydim = 1000

bitmap = [[0 for _ in range(Ydim)] for _ in range(Xdim)]

def GaussianMatrix(radius, stddev):
    mat = [[0 for _ in range(radius*2 + 1)] for _ in range(radius*2 + 1)]
    for i in range(radius*2 + 1):
        for j in range(radius*2 + 1):
            rad = sqrt((i - radius)**2 + (j - radius)**2)
            mat[i][j] = np.exp(-(rad**2)/(2*(stddev**2)))
    return mat

def blur(radius, stddev):
    global bitmap
    newmap = [[x for x in row] for row in bitmap]
    mat = GaussianMatrix(radius, stddev)
    for x in range(Xdim):
        for y in range(Ydim):
            left = max(x - radius, 0)
            right = min(x + radius, Xdim)
            top = max(y - radius, 0)
            bottom = min(y + radius, Ydim)
            total = 0
            for px in range(left, right):
                for py in range(top, bottom):
                    val = bitmap[px][py] 
                    # calculate x and y parameters for gaussian
                    gradius = sqrt(x**2 + y**2)
                    total += val*mat[px - left][py - top]
            newmap[x][y] = min(int(total), 255)
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
    #print(ret)
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
    print(o)
    print((xo, yo))
    print((int(x), int(y)))
    y = int(0)
    for xi in range(0, int(x)):
        (px, py) = switchToOctantFromZero(o, xi, y)
        bitmap[int(zero[0] + px)][int(zero[1] + py)] = 255
        #print((int(zero[0] + px), int(zero[1] + py)))
        error += derr
        while (error >= 0.5):
            (px, py) = switchToOctantFromZero(o, xi, y)
            bitmap[int(zero[0] + px)][int(zero[1] + py)] = 255
            #print((int(zero[0] + px), int(zero[1] + py)))
            y += sign(dy)
            error -= 1.0

def line(x0, y0, x1, y1):
    zeroLine((y0, x0), y1, x1)

def draw(p1, p2):
    line(p1.x + center.x, p1.y + center.y, p2.x + center.x, p2.y + center.y)

def line_center(x0, y0, x1, y1):
    line(x0 + center.x, y0 + center.y, x1 + center.x, y1 + center.y)

def render():
    pic = png.from_array(bitmap, mode='L')
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
        basisq = Coordinate(sqrt(3.0)/4.0, 0.75, 0).mult(height)
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
            print((p1.x, p1.y, p2.x, p2.y))
            draw(p1, p2)    
    def Cube(self):
        height = self.edge*2
        basisr = Coordinate(sqrt(3.0)/2.0, 0, 0).mult(height)
        basisq = Coordinate(sqrt(3.0)/4.0, -0.75, 0).mult(height)
        y = self.center.proj(basisq)
        remaining = self.center + y.mult(-1)
        x = remaining.proj(basisr)
        return Cube(Coordinate(x.mag(), y.mag()), self.edge) 

A = Cube(Coordinate(0, 0), 50)
A.hexagon().draw()

#blur(5, 0.8)

pic = render()
pic.save('output.png')
