# import numpy as np
# from math import sqrt

# class UnionFind:
#     def __init__(self, n):
#         self.parent = list(range(n))
#     def find(self, x):
#         while self.parent[x] != x:
#             self.parent[x] = self.parent[self.parent[x]]
#             x = self.parent[x]
#         return x
#     def union(self, x, y):
#         px, py = self.find(x), self.find(y)
#         if px != py:
#             self.parent[py] = px

# def euclidean_dist(p1, p2):
#     return sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# def group_points(points, tolerance):
#     n = len(points)
#     uf = UnionFind(n)
#     for i in range(n):
#         for j in range(i+1, n):
#             if euclidean_dist(points[i], points[j]) <= tolerance:
#                 uf.union(i, j)
#     groups = {}
#     for i, p in enumerate(points):
#         root = uf.find(i)
#         groups.setdefault(root, []).append(p)
#     return list(groups.values())

# def fill_and_clean(arr, tolerance, label_fill, min_area=None):
#     points_label = list(zip(*np.where(arr == label_fill)))
#     if not points_label:
#         return np.zeros_like(arr)
    
#     groups = group_points(points_label, tolerance)
#     result = np.zeros_like(arr)
    
#     # Nếu không truyền min_area thì mặc định dùng max(tolerance * 2, 4)
#     if min_area is None:
#         min_area = max(tolerance * 2, 4)

#     for g in groups:
#         rows = [p[0] for p in g]
#         cols = [p[1] for p in g]
#         r_min, r_max = min(rows), max(rows)
#         c_min, c_max = min(cols), max(cols)
#         area = (r_max - r_min + 1) * (c_max - c_min + 1)
#         if area >= min_area:
#             result[r_min:r_max+1, c_min:c_max+1] = label_fill
    
#     return result

# # Test ví dụ bạn đưa
# arr = np.array([
#     [3,0,4,0,5,0,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#     [0,4,4,0,0,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#     [0,1,1,1,1,0,0,2,2,2,2,0,1,1,1,0,0,0,0,2,0],
#     [0,1,1,1,1,2,2,2,2,2,0,0,1,1,1,0,0,0,2,0,0],
#     [0,1,1,1,1,0,2,2,0,2,0,0,0,2,0,0,0,2,0,0,0],
#     [0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,2,0,2,0,0,0,0,0,0,0,0,0,0,0]
# ])

# tolerance = 2
# label_fill = 4
# result = fill_and_clean(arr, tolerance,label_fill,min_area=4)
# print(result)



import numpy as np
from math import sqrt

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px != py:
            self.parent[py] = px

def euclidean_dist(p1, p2):
    return sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def group_points(points, tolerance):
    n = len(points)
    uf = UnionFind(n)
    for i in range(n):
        for j in range(i+1, n):
            if euclidean_dist(points[i], points[j]) <= tolerance:
                uf.union(i, j)
    groups = {}
    for i, p in enumerate(points):
        root = uf.find(i)
        groups.setdefault(root, []).append(p)
    return list(groups.values())

def group_points_axis_aligned(points, tolerance):
    from collections import defaultdict

    point_set = set(points)
    point_idx = {p: i for i, p in enumerate(points)}
    uf = UnionFind(len(points))

    for idx, (i, j) in enumerate(points):
        for d in range(1, tolerance + 1):
            for ni, nj in [(i + d, j), (i - d, j), (i, j + d), (i, j - d)]:
                if (ni, nj) in point_set:
                    uf.union(idx, point_idx[(ni, nj)])

    groups = defaultdict(list)
    for i, p in enumerate(points):
        root = uf.find(i)
        groups[root].append(p)
    return list(groups.values())

def fill_and_clean(arr, tolerance, label_fill, min_area=None, aspect_ratio_threshold=1.0):
    points_label = list(zip(*np.where(arr == label_fill)))
    if not points_label:
        return np.zeros_like(arr)
    
    groups = group_points_axis_aligned(points_label, tolerance)
    result = np.zeros_like(arr)
    
    if min_area is None:
        min_area = max(tolerance * 2, 4)

    for g in groups:
        rows = [p[0] for p in g]
        cols = [p[1] for p in g]
        r_min, r_max = min(rows), max(rows)
        c_min, c_max = min(cols), max(cols)
        height = r_max - r_min + 1
        width = c_max - c_min + 1
        area = height * width

        # Kiểm tra điều kiện diện tích và tỉ lệ chiều dài/rộng
        longer = max(height, width)
        shorter = min(height, width)

        if area >= min_area and longer / shorter >= aspect_ratio_threshold:
            result[r_min:r_max+1, c_min:c_max+1] = label_fill

    return result


# Test ví dụ bạn đưa
arr = np.array([
    [3,0,4,0,5,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,4,4,0,0,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,1,1,1,0,0,2,2,2,2,0,1,1,1,0,0,0,0,2,0],
    [0,1,1,1,1,0,0,2,2,2,0,0,1,1,1,0,0,0,2,0,0],
    [0,1,1,1,1,0,0,2,0,2,0,0,0,2,0,0,0,2,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,2,0,2,0,0,0,0,0,0,0,0,0,0,0]
])

tolerance = 2
label_fill = 2
min_area = 4
aspect_ratio = 1  # Ví dụ: chiều dài phải gấp 2.5 lần chiều rộng


result = fill_and_clean(arr, tolerance, label_fill, min_area=min_area, aspect_ratio_threshold=aspect_ratio)
print(result)