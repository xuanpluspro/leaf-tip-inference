import os
from pathlib import Path
import uuid

import numpy as np
from PIL import Image, ImageDraw, ImageFont

EXTENSIONS={'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}


def normalized_image(image):
    x = np.array(image.convert('RGB'),dtype=np.float32).transpose(2,0,1)/255.
    mean=np.array([.485,.456,.406],dtype=np.float32)[:,None,None]
    std=np.array([.229,.224,.225],dtype=np.float32)[:,None,None]
    return np.ascontiguousarray((x-mean)/std)


def plan_images(source,output):
    source,output=Path(source).resolve(),Path(output).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_dir() and (source==output or source in output.parents):
        raise ValueError('Output must be outside input directory')
    files=sorted(p for p in source.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS) if source.is_dir() else [source]
    if not files or any(p.suffix.lower() not in EXTENSIONS for p in files):
        raise ValueError('No supported images')
    plan,seen=[],set()
    for path in files:
        relative=path.relative_to(source) if source.is_dir() else Path(path.name)
        overlay=output/'overlays'/relative.parent/(path.stem+'_overlay.jpg')
        key=str(overlay).casefold()
        if key in seen:
            raise ValueError(f'Same-stem inputs would collide; rename first: {path}')
        seen.add(key)
        plan.append((path,relative,overlay))
    return plan


def draw_points(image,points,radius=None):
    # At 3072px: radius 15, white outline 3px, count font 66px.
    scale=min(image.size)/1024.
    radius=max(4,round(5*scale)) if radius is None else radius
    outline=max(1,round(radius/5))
    result=image.copy()
    draw=ImageDraw.Draw(result)
    for x,y in points:
        draw.ellipse((float(x)-radius,float(y)-radius,float(x)+radius,float(y)+radius),
                     fill=(255,0,0),outline=(255,255,255),width=outline)
    label=f'Count: {len(points)}'
    font_size=max(16,round(22*scale))
    try:
        font=ImageFont.truetype('DejaVuSans.ttf',font_size)
    except OSError:
        try:
            font=ImageFont.load_default(size=font_size)
        except TypeError:
            font=ImageFont.load_default()
    padding=max(6,round(8*scale))
    box=draw.textbbox((0,0),label,font=font)
    draw.rectangle((0,0,box[2]-box[0]+2*padding,box[3]-box[1]+2*padding),fill=(0,0,0))
    draw.text((padding-box[0],padding-box[1]),label,font=font,fill=(255,255,0))
    return result


def save_jpeg(image,path,quality=90):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('wb') as stream:
            image.save(stream,format='JPEG',quality=quality,optimize=True)
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)

