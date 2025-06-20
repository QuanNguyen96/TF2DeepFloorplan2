from skimage.morphology import skeletonize
import cv2
import numpy as np
import networkx as nx
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    return morph

def extract_skeleton(binary):
    skel = skeletonize(binary // 255)
    return skel.astype(np.uint8)

def skeleton_to_graph(skel):
    G = nx.Graph()
    h, w = skel.shape
    for y in range(1, h-1):
        for x in range(1, w-1):
            if skel[y, x]:
                neighbors = [(y+dy, x+dx) for dy in [-1,0,1] for dx in [-1,0,1] if (dy,dx)!=(0,0)]
                for ny, nx_ in neighbors:
                    if skel[ny, nx_]:
                        G.add_edge((x, y), (nx_, ny))
    return G

def find_rooms(graph, min_area=500):
    cycles = nx.cycle_basis(graph)
    rooms = []
    for cycle in cycles:
        poly = Polygon(cycle)
        if poly.is_valid and poly.area > min_area:
            rooms.append(poly)
    return rooms

def draw_result(image_path, graph, rooms):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 12))
    plt.imshow(img_rgb)
    ax = plt.gca()

    for u, v in graph.edges:
        ax.plot([u[0], v[0]], [u[1], v[1]], color='red', linewidth=1)

    for poly in rooms:
        x, y = poly.exterior.xy
        ax.plot(x, y, color='blue', linewidth=2, linestyle='--')

    plt.title("Walls (red) and Rooms (blue)")
    plt.axis('off')
    plt.show()
    
img_path = "./resources/123.jpg"

binary = preprocess_image(img_path)
skeleton = extract_skeleton(binary)
G = skeleton_to_graph(skeleton)
rooms = find_rooms(G, min_area=800)
draw_result(img_path, G, rooms)