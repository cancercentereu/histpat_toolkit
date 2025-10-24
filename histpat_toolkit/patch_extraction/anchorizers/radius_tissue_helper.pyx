# Import necessary modules
import numpy as np
cimport numpy as np

# Define the `next_mod` function in Cython
cdef int next_mod(int x, int mod):
    return x + 1 if x < mod - 1 else 0

# Define the `bfs` function in Cython
cdef void bfs(int start_x, int start_y, int mod, int vis_value, int radius, np.ndarray[np.uint8_t, ndim=2] tissue_mask, np.ndarray[np.int32_t, ndim=2] vis, np.ndarray[np.int32_t, ndim=2] queue):
    cdef int queue_start = 0
    cdef int queue_end = 0

    vis[start_x, start_y] = vis_value
    queue[queue_end, 0] = start_x
    queue[queue_end, 1] = start_y
    queue[queue_end, 2] = radius
    queue_end = next_mod(queue_end, mod)

    cdef int x, y, radius_left
    cdef int nx, ny
    while queue_end != queue_start:
        x, y, radius_left = queue[queue_start, 0], queue[queue_start, 1], queue[queue_start, 2]
        queue_start = next_mod(queue_start, mod)

        if radius_left == 0:
            continue
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= tissue_mask.shape[0] or ny < 0 or ny >= tissue_mask.shape[1]:
                    continue
                if vis[nx, ny] == vis_value:
                    continue
                if tissue_mask[nx, ny] == 0:
                    continue

                vis[nx, ny] = vis_value
                queue[queue_end, 0] = nx
                queue[queue_end, 1] = ny
                queue[queue_end, 2] = radius_left - 1
                queue_end = next_mod(queue_end, mod)

# Define the `anchorize_tissue` function in Cython
cpdef np.ndarray[np.uint8_t, ndim=2] anchorize_tissue(np.ndarray[np.uint8_t, ndim=2] tissue_mask, int radius):
    cdef np.ndarray[np.int32_t, ndim=2] vis = np.zeros((tissue_mask.shape[0], tissue_mask.shape[1]), dtype=np.int32)
    cdef np.ndarray[np.uint8_t, ndim=2] anchors_mask = np.zeros((tissue_mask.shape[0], tissue_mask.shape[1]), dtype=np.uint8)

    cdef int cnt = 0
    where_x, where_y = np.where(tissue_mask)
    cdef int mod = where_x.size
    cdef np.ndarray[np.int32_t, ndim=2] queue = np.zeros((mod, 3), dtype=np.int32)
    for x, y in zip(where_x, where_y):
        if vis[x, y] == 0:
            anchors_mask[x, y] = 255
            cnt += 1
            bfs(x, y, mod, cnt, radius, tissue_mask, vis, queue)

    return anchors_mask