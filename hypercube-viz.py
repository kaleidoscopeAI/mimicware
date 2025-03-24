import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Set
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

@dataclass
class Vector4D:
    x: float
    y: float
    z: float
    w: float

@dataclass
class Supercluster:
    position: Vector4D
    intensity: float
    connections: Set[Tuple[int, int]]

class HypercubeStringNetwork:
    def __init__(self, dimension: int = 4, resolution: int = 10):
        self.dimension = dimension
        self.resolution = resolution
        self.vertices = self._generate_vertices()
        self.strings = self._generate_strings()
        self.superclusters = self._find_intersections()
        
    def _generate_vertices(self) -> List[Vector4D]:
        vertices = []
        for x in [-1, 1]:
            for y in [-1, 1]:
                for z in [-1, 1]:
                    for w in [-1, 1]:
                        vertices.append(Vector4D(x, y, z, w))
        return vertices
    
    def _generate_strings(self) -> List[Tuple[Vector4D, Vector4D]]:
        strings = []
        steps = np.linspace(-1, 1, self.resolution)
        
        # Generate strings for each face pair in 4D
        for dim1 in range(4):
            for dim2 in range(dim1 + 1, 4):
                for i in steps:
                    for j in steps:
                        start = [0] * 4
                        end = [0] * 4
                        start[dim1] = i
                        start[dim2] = j
                        start[-1] = -1
                        end[dim1] = i
                        end[dim2] = j
                        end[-1] = 1
                        strings.append((
                            Vector4D(*start),
                            Vector4D(*end)
                        ))
        return strings
    
    def _find_intersections(self) -> List[Supercluster]:
        superclusters = []
        threshold = 0.1  # Distance threshold for intersection detection
        
        # O(n^2) intersection check with spatial optimization
        string_segments = np.array([[
            [s[0].x, s[0].y, s[0].z, s[0].w],
            [s[1].x, s[1].y, s[1].z, s[1].w]
        ] for s in self.strings])
        
        for i in range(len(self.strings)):
            for j in range(i + 1, len(self.strings)):
                intersection = self._compute_intersection(
                    string_segments[i],
                    string_segments[j]
                )
                if intersection is not None:
                    superclusters.append(Supercluster(
                        position=Vector4D(*intersection),
                        intensity=1.0,
                        connections={(i, j)}
                    ))
        
        return self._merge_nearby_clusters(superclusters, threshold)
    
    def _compute_intersection(self, seg1: np.ndarray, seg2: np.ndarray) -> np.ndarray:
        # Compute closest point between two 4D line segments using linear algebra
        d1 = seg1[1] - seg1[0]
        d2 = seg2[1] - seg2[0]
        n1 = np.linalg.norm(d1)
        n2 = np.linalg.norm(d2)
        
        if n1 < 1e-10 or n2 < 1e-10:
            return None
            
        d1 /= n1
        d2 /= n2
        
        # Simplified 4D check using the first 3 components for cross product
        normal = np.cross(d1[:3], d2[:3])
        if np.linalg.norm(normal) < 1e-10:
            return None
            
        # Solve system of equations for intersection parameters
        A = np.vstack((d1, -d2)).T
        b = seg2[0] - seg1[0]
        
        try:
            t, s = np.linalg.lstsq(A, b, rcond=None)[0]
            if 0 <= t <= n1 and 0 <= s <= n2:
                return seg1[0] + t * d1
            return None
        except np.linalg.LinAlgError:
            return None
    
    def _merge_nearby_clusters(
        self,
        clusters: List[Supercluster],
        threshold: float
    ) -> List[Supercluster]:
        if not clusters:
            return []
            
        merged = []
        used = set()
        
        for i, c1 in enumerate(clusters):
            if i in used:
                continue
                
            current = c1
            used.add(i)
            
            for j, c2 in enumerate(clusters[i+1:], i+1):
                if j in used:
                    continue
                    
                dist = np.sqrt(
                    (c1.position.x - c2.position.x) ** 2 +
                    (c1.position.y - c2.position.y) ** 2 +
                    (c1.position.z - c2.position.z) ** 2 +
                    (c1.position.w - c2.position.w) ** 2
                )
                
                if dist < threshold:
                    current.intensity += c2.intensity
                    current.connections.update(c2.connections)
                    used.add(j)
            
            merged.append(current)
            
        return merged
    
    def project_to_3d(self, w_slice: float = 0) -> Tuple[np.ndarray, np.ndarray]:
        # Project 4D strings and clusters to 3D for visualization
        string_points_3d = []
        for start, end in self.strings:
            if abs(start.w - w_slice) < 0.1 or abs(end.w - w_slice) < 0.1:
                string_points_3d.append([
                    [start.x, start.y, start.z],
                    [end.x, end.y, end.z]
                ])
                
        cluster_points_3d = []
        intensities = []
        for cluster in self.superclusters:
            if abs(cluster.position.w - w_slice) < 0.1:
                cluster_points_3d.append([
                    cluster.position.x,
                    cluster.position.y,
                    cluster.position.z
                ])
                intensities.append(cluster.intensity)
                
        return np.array(string_points_3d), np.array(cluster_points_3d), intensities

    def visualize(self, w_slices: List[float] = [-0.5, 0, 0.5]):
        fig = plt.figure(figsize=(15, 5))
        
        for i, w in enumerate(w_slices, 1):
            ax = fig.add_subplot(1, len(w_slices), i, projection='3d')
            strings_3d, clusters_3d, intensities = self.project_to_3d(w)
            
            # Plot strings
            for string in strings_3d:
                ax.plot3D(
                    string[:, 0],
                    string[:, 1],
                    string[:, 2],
                    'b-',
                    alpha=0.1
                )
            
            # Plot superclusters
            if len(clusters_3d) > 0:
                clusters_3d = np.array(clusters_3d)
                intensities = np.array(intensities)
                ax.scatter(
                    clusters_3d[:, 0],
                    clusters_3d[:, 1],
                    clusters_3d[:, 2],
                    c=intensities,
                    cmap='viridis',
                    s=100
                )
            
            ax.set_title(f'w = {w}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    network = HypercubeStringNetwork(dimension=4, resolution=10)
    network.visualize()
