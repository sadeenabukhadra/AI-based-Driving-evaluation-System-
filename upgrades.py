import math
from collections import deque

history = deque(maxlen=15)

def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1+x2)//2, (y1+y2)//2)

def dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def pixel_to_cm(px):
    return px * 0.5

def smooth(val):
    history.append(val)
    return sum(history)/len(history)

def calculate_distance(box1, box2):
    c1 = get_center(box1)
    c2 = get_center(box2)
    return smooth(pixel_to_cm(dist(c1, c2)))

def get_closest_cones(car_center, cones, k=4):
    cones = sorted(cones, key=lambda c: dist(car_center, get_center(c["box"])))
    return cones[:k]

def is_centered(car_center, cones):
    xs = [get_center(c["box"])[0] for c in cones]
    mid = (min(xs)+max(xs))/2
    return abs(car_center[0] - mid) < 20

def inside_parking(car_center, cones):
    xs = [get_center(c["box"])[0] for c in cones]
    ys = [get_center(c["box"])[1] for c in cones]

    return (min(xs) < car_center[0] < max(xs)) and (min(ys) < car_center[1] < max(ys))

def check_collision(dist):
    return dist < 20