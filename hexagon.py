from math import sin, cos, sqrt
PI = 3.1415926535
import matplotlib.pyplot as plt
import png

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
        return self.mult(1/c)
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


Xdim = 1000
Ydim = 1000
bitmap = Xdim*[Ydim*[0]]

center = Coordinate(Xdim/2, Ydim/2, 0)

def draw(p1, p2):
    startc = center + p1
    vec = (p2 + p1.mult(-1)).norm().div(100.0)
    for i in range(int(100*(p2 + p1.mult(-1)).mag())):
        point = startc + vec.mult(i)
        x = int(point.x)
        y = int(point.y)
        bitmap[x][y] = 1

def render():
    pic = png.from_array(bitmap, mode='L;1')
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
        print(self.loc)
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
    def Cube(self):
        height = self.edge*2
        basisr = Coordinate(sqrt(3.0)/2.0, 0, 0).mult(height)
        basisq = Coordinate(sqrt(3.0)/4.0, -0.75, 0).mult(height)
        y = self.center.proj(basisq)
        remaining = self.center + y.mult(-1)
        x = remaining.proj(basisr)
        return Cube(Coordinate(x.mag(), y.mag()), self.edge) 

A = Cube(Coordinate(0, 0), 1)
B = A.getNeighbor('LD')
C = A.getNeighbor('RD')
B.hexagon().draw()
C.hexagon().draw()
A.hexagon().draw()

pic = render()
pic.save('output.png')

