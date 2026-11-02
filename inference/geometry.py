import numpy as np


def padded_size(height, width, multiple, minimum):
    return tuple(max(minimum, (v+multiple-1)//multiple*multiple) for v in (height,width))


def pet_pixels(points_yx, height, width):
    points = np.asarray(points_yx, dtype=np.float32).reshape(-1,2)
    return points[:, ::-1].copy() * np.array([width,height], dtype=np.float32)


def grid_pixels(rows, columns, input_height, input_width, output_height, output_width):
    sy, sx = input_height/output_height, input_width/output_width
    return np.column_stack([np.asarray(columns)*sx+(sx-1)/2,
                            np.asarray(rows)*sy+(sy-1)/2]).astype(np.float32)


def valid_predictions(points, scores, width, height, threshold):
    points = np.asarray(points, dtype=np.float32).reshape(-1,2)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(points) != len(scores):
        raise ValueError('Point/score length mismatch')
    keep = (np.isfinite(points).all(1) & np.isfinite(scores) & (scores > threshold)
            & (points[:,0]>=0) & (points[:,0]<width) & (points[:,1]>=0) & (points[:,1]<height))
    return points[keep], scores[keep]


def tile_axis(length, size, overlap):
    if length < 1 or size < 1 or not 0 <= overlap < size:
        raise ValueError('Invalid tile dimensions')
    if length <= size:
        return [(0,length,0.,float(length))]
    starts = list(range(0,length-size+1,size-overlap))
    if starts[-1] != length-size:
        starts.append(length-size)
    bounds = [0.] + [(left+size+right)/2 for left,right in zip(starts,starts[1:])] + [float(length)]
    return [(s,s+size,bounds[i],bounds[i+1]) for i,s in enumerate(starts)]


def owned_predictions(points, scores, left, top, bounds):
    points = np.asarray(points,dtype=np.float32).reshape(-1,2)+np.array([left,top],dtype=np.float32)
    x0,y0,x1,y1 = bounds
    keep = ((points[:,0]>=x0)&(points[:,0]<x1)&(points[:,1]>=y0)&(points[:,1]<y1))
    return points[keep], np.asarray(scores)[keep]
