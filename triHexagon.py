A = Hexagon(0, 0, 100)
B = A.cube().getNeighbor('LU').hexagon().draw()
C = A.cube().getNeighbor('RU').hexagon().draw()
A.draw()
