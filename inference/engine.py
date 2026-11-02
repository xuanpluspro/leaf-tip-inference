import numpy as np
from .geometry import tile_axis, owned_predictions


def predict_image(adapter,image,mode='full',tile_size=512,overlap=128,amp=False):
    if mode == 'full':
        return adapter.predict(image,amp)
    if mode != 'tile':
        raise ValueError('Expected full or tile')
    points,scores = [],[]
    for top,bottom,own_top,own_bottom in tile_axis(image.height,tile_size,overlap):
        for left,right,own_left,own_right in tile_axis(image.width,tile_size,overlap):
            crop = image.crop((left,top,right,bottom))
            p,s = adapter.predict(crop,amp)
            p,s = owned_predictions(p,s,left,top,(own_left,own_top,own_right,own_bottom))
            points.append(p)
            scores.append(s)
    return np.concatenate(points),np.concatenate(scores)
