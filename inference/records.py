import csv
import hashlib
import json
import os
from pathlib import Path
import uuid


def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path,content):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        temporary.write_text(content,encoding='utf-8')
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path,payload):
    atomic_text(path,json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def open_run(output,config,resume=False,overwrite=False):
    output=Path(output)
    manifest=output/'meta/run.json'
    if resume and overwrite:
        raise ValueError('resume and overwrite are mutually exclusive')
    if manifest.exists() and resume:
        state=json.loads(manifest.read_text(encoding='utf-8'))
        if state['config']!=config:
            raise ValueError('Resume requires identical model, weights, input and inference settings')
    else:
        if output.exists() and any(output.iterdir()) and not overwrite:
            raise FileExistsError('Output nonempty: use --resume, --overwrite or a new directory')
        state=dict(config=config,completed={})
        atomic_json(manifest,state)
        (output/'counts.csv').unlink(missing_ok=True)
    return state,manifest


def export_counts(output,state,keys):
    import io
    stream=io.StringIO(newline='')
    fields=['image','count','width','height','inference_seconds','save_seconds','peak_allocated_gib','overlay','points']
    writer=csv.DictWriter(stream,fieldnames=fields)
    writer.writeheader()
    for key in keys:
        row=state['completed'].get(key)
        if row:
            writer.writerow(dict(image=key,**{field:row[field] for field in fields if field!='image'}))
    atomic_text(Path(output)/'counts.csv','\ufeff'+stream.getvalue())
