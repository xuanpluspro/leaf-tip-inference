"""Portable editable point annotations; no torch, GUI or model imports."""
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import tempfile

from PIL import Image

from .images import plan_images
from .records import atomic_json, atomic_text


def relative_path(value):
    if not isinstance(value,str) or not value or '\\' in value:
        raise ValueError(f'Expected portable relative path: {value!r}')
    path=PurePosixPath(value)
    if path.is_absolute() or PureWindowsPath(value).drive or '..' in path.parts:
        raise ValueError(f'Unsafe path: {value}')
    return Path(*path.parts)


def finite_number(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError(f'Expected finite numeric coordinate/score: {value!r}')
    return float(value)


def point_shape(x,y,label,score=None):
    shape=dict(label=label,points=[[x,y]],group_id=None,description='',shape_type='point',flags={})
    if score is not None: shape['score']=score
    return shape


def annotation_payload(image_name,width,height,points,label='leaf_tip'):
    if not label.strip(): raise ValueError('Label cannot be empty')
    shapes=[]
    for row in points:
        x,y=finite_number(row['x']),finite_number(row['y'])
        score=finite_number(row['score'])
        if not 0<=x<width or not 0<=y<height or not 0<=score<=1:
            raise ValueError('Prediction coordinates/score out of bounds')
        shapes.append(point_shape(x,y,label,score))
    return dict(version='5.0.1',flags={},checked=False,shapes=shapes,imagePath=image_name,
                imageData=None,imageHeight=height,imageWidth=width,
                description='Model-generated leaf-tip points; human review required.')


def validate_image(path,width,height):
    with Image.open(path) as image:
        if image.size!=(width,height): raise ValueError(f'Image dimensions differ from predictions: {path}')
        if image.getexif().get(274,1)!=1:
            raise ValueError(f'Nontrivial EXIF orientation: {path}. Normalize orientation before inference and rerun; coordinates are not silently rotated.')
        if getattr(image,'n_frames',1)!=1: raise ValueError(f'Multi-frame image not supported for annotation export: {path}')


def new_destination(output,protected):
    output=Path(output).resolve()
    for path in protected:
        path=Path(path).resolve()
        if output==path or path in output.parents:
            raise ValueError('Export destination must be outside source images/annotations/prediction directory')
    if output.exists(): raise FileExistsError(f'Use a NEW output directory; existing manual labels will never be overwritten: {output}')
    return output


def export_annotations(source,run,output,label='leaf_tip',min_score=None):
    source,run=Path(source).resolve(),Path(run).resolve()
    output=new_destination(output,[source,run])
    if min_score is not None and not 0<=min_score<=1: raise ValueError('min-score must be in [0,1]')
    state=json.loads((run/'meta/run.json').read_text(encoding='utf-8-sig'))
    completed=state.get('completed')
    if not isinstance(completed,dict) or not completed: raise ValueError('No completed predictions in meta/run.json')
    image_map={relative.as_posix():path for path,relative,_ in plan_images(source,output)}
    plans=[]; names=set()
    for key,row in sorted(completed.items()):
        relative=relative_path(key)
        if key not in image_map: raise FileNotFoundError(f'Original image is missing: {key}')
        if not row.get('points'): raise ValueError('Prediction coordinates not saved; rerun inference with --save-points')
        record_path=run/relative_path(row['points'])
        if not record_path.resolve().is_relative_to(run): raise ValueError('Prediction path escapes run directory')
        record=json.loads(record_path.read_text(encoding='utf-8-sig'))
        if record.get('image')!=key or record.get('coordinate_order')!='xy' or record.get('coordinate_unit')!='original-image pixels':
            raise ValueError(f'Unexpected coordinate metadata: {record_path}')
        width,height=record['width'],record['height']
        if (width,height)!=(row['width'],row['height']): raise ValueError('Run manifest and point-record dimensions disagree')
        if len(record['points'])!=record['count'] or record['count']!=row['count']: raise ValueError('Prediction count mismatch')
        payload=annotation_payload(relative.name,width,height,record['points'],label)
        if min_score is not None: payload['shapes']=[s for s in payload['shapes'] if s['score']>=min_score]
        if row.get('input',{}).get('size',image_map[key].stat().st_size)!=image_map[key].stat().st_size:
            raise ValueError(f'Original image file size changed since inference: {key}')
        validate_image(image_map[key],width,height)
        annotation=relative.with_suffix('.json')
        for target in (relative,annotation):
            if target.as_posix().casefold() in names: raise ValueError('Duplicate export filename')
            names.add(target.as_posix().casefold())
        plans.append((image_map[key],relative,annotation,payload))
    # Stage the complete portable package first. No original image/annotation is edited.
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix='.'+output.name+'.',dir=output.parent))
    try:
        for image_path,relative,annotation,payload in plans:
            target=temporary/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(image_path,target) # exact source bytes, no resizing/JPEG re-encoding
            atomic_json(temporary/annotation,payload)
        if output.exists(): raise FileExistsError(output)
        os.rename(temporary,output)
    finally:
        if temporary.exists(): shutil.rmtree(temporary)
    return dict(images=len(plans),points=sum(len(p[3]['shapes']) for p in plans),output=str(output))


def corrected_points(payload,label='leaf_tip'):
    width,height=payload['imageWidth'],payload['imageHeight']
    if not isinstance(width,int) or not isinstance(height,int) or min(width,height)<1: raise ValueError('Invalid image dimensions')
    if not isinstance(payload.get('shapes'),list): raise ValueError('Missing shapes list; use GUI annotation JSON, not inference coordinate JSON')
    result=[];seen=set()
    for shape in payload['shapes']:
        if shape.get('label')!=label: raise ValueError(f'Unexpected label {shape.get("label")!r}; expected {label!r}')
        if shape.get('shape_type')!='point' or len(shape.get('points',[]))!=1 or len(shape['points'][0])!=2:
            raise ValueError('Every shape must be a single point, not a polygon/rectangle')
        x,y=map(finite_number,shape['points'][0])
        if not 0<=x<width or not 0<=y<height: raise ValueError('Corrected point outside image')
        if (x,y) in seen: raise ValueError('Duplicate corrected point; remove the duplicate in the annotation software')
        seen.add((x,y));result.append((x,y))
    return result


def annotations_to_points(source,output,label='leaf_tip'):
    source=Path(source).resolve();output=new_destination(output,[source])
    paths=sorted(source.rglob('*.json'))
    if not paths: raise ValueError('No annotation JSON files')
    plans=[];seen=set()
    for path in paths:
        payload=json.loads(path.read_text(encoding='utf-8-sig'))
        points=corrected_points(payload,label)
        image_relative=relative_path(payload['imagePath'])
        image_path=(path.parent/image_relative).resolve()
        if not image_path.is_relative_to(source): raise ValueError('Referenced image must be inside the annotation package')
        validate_image(image_path,payload['imageWidth'],payload['imageHeight'])
        relative=path.relative_to(source).with_suffix('.txt')
        if relative.as_posix().casefold() in seen: raise ValueError('Output filename collision')
        seen.add(relative.as_posix().casefold())
        # repr preserves float precision; fixed 6 decimals could merge nearby points.
        plans.append((relative,''.join(f'{x!r} {y!r}\n' for x,y in points),len(points)))
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix='.'+output.name+'.',dir=output.parent))
    try:
        for relative,content,_ in plans: atomic_text(temporary/relative,content)
        if output.exists(): raise FileExistsError(output)
        os.rename(temporary,output)
    finally:
        if temporary.exists(): shutil.rmtree(temporary)
    return dict(images=len(plans),points=sum(p[2] for p in plans),output=str(output))
