"""Unified PET/P2RLoss/APGCC/SAH leaf-tip CUDA inference."""
import argparse
import json
import logging
from pathlib import Path
import sys
import time

from PIL import Image

from inference.engine import predict_image
from inference.annotations import annotation_payload,validate_image,save_annotation_pair
from inference.images import plan_images,draw_points,save_jpeg
from inference.records import sha256_file,open_run,atomic_json,atomic_text,export_counts


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,choices=['pet','p2rloss','apgcc','sah'])
    p.add_argument('--weights','--checkpoint',dest='weights',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True,help='Use a separate output directory per model/checkpoint')
    p.add_argument('--mode',choices=['full','tile'],default='full')
    p.add_argument('--tile-size',type=int,default=512)
    p.add_argument('--overlap',type=int,default=128)
    p.add_argument('--threshold',type=float,default=.5,help='Foreground probability; P2RLoss .5 equals output logit > 0')
    p.add_argument('--p2r-weights',choices=['student','teacher'],default='student')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--amp',action=argparse.BooleanOptionalAction,default=False,help='Optional FP16 autocast; default FP32 matches validation')
    p.add_argument('--radius',type=int,default=None,help='Point radius in original pixels; default auto: 15px at 3072 square')
    p.add_argument('--save-annotations',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--annotation-label',default='leaf_tip')
    p.add_argument('--jpg-quality',type=int,default=90)
    p.add_argument('--save-points',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--max-images',type=int,default=0,help='0=all; e.g. 1 for initial GPU smoke check')
    p.add_argument('--trust-checkpoint',action='store_true',help='Allow pickle for your own trusted legacy training checkpoint only')
    group=p.add_mutually_exclusive_group()
    group.add_argument('--resume',action='store_true')
    group.add_argument('--overwrite',action='store_true')
    return p


def main(args):
    if not 0<args.threshold<1 or not 1<=args.jpg_quality<=100 or (args.radius is not None and args.radius<1) or args.max_images<0:
        raise ValueError('Invalid threshold, JPG quality, radius or max-images')
    if not args.annotation_label.strip():
        raise ValueError('annotation-label cannot be empty')
    if args.tile_size<256 or args.tile_size%256 or not 0<=args.overlap<args.tile_size:
        raise ValueError('tile-size must be >=256 and a multiple of 256; 0 <= overlap < tile-size')
    if args.model!='p2rloss' and args.p2r_weights!='student':
        raise ValueError('--p2r-weights teacher applies only to P2RLoss')
    import torch
    from inference.adapters import Adapter
    device=torch.device(args.device)
    if device.type!='cuda' or not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required; no CPU fallback or CPU neural-network execution')
    torch.cuda.set_device(device)
    source,output=args.input.resolve(),args.output.resolve()
    plan=plan_images(source,output)
    if args.save_annotations:
        for path,_,_ in plan:
            with Image.open(path) as image:
                validate_image(path,image.width,image.height)
    if args.max_images:
        plan=plan[:args.max_images]
    config=dict(model=args.model,weights_sha256=sha256_file(args.weights),source=str(source),
                mode=args.mode,tile_size=args.tile_size,overlap=args.overlap,
                threshold=args.threshold,p2r_weights=args.p2r_weights,amp=args.amp,
                radius=args.radius,jpg_quality=args.jpg_quality,save_points=args.save_points,
                save_annotations=args.save_annotations,annotation_label=args.annotation_label,
                visualization_version=2,
                upstream=json.loads((Path(__file__).parent/'UPSTREAM.json').read_text(encoding='utf-8')))
    # Load and validate the model before resetting an existing run's metadata.
    adapter=Adapter(args.model,args.weights,device,args.threshold,args.p2r_weights,args.trust_checkpoint)
    state,manifest=open_run(output,config,args.resume,args.overwrite)
    logger=logging.getLogger('leaf-tip')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    for handler in (logging.StreamHandler(sys.stdout),logging.FileHandler(output/'meta/console.log',
                    mode='w' if args.overwrite else 'a',encoding='utf-8')):
        handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
        logger.addHandler(handler)
    keys=[relative.as_posix() for _,relative,_ in plan]
    try:
        logger.info('model=%s | Params=%.3fM | checkpoint epoch index=%s | GPU=%s | %s | mode=%s | images=%d',
                    args.model,adapter.params/1e6,adapter.epoch,torch.cuda.get_device_name(device),
                    'AMP' if args.amp else 'FP32',args.mode,len(plan))
        for index,(path,relative,overlay) in enumerate(plan,1):
            key=relative.as_posix()
            stat=path.stat()
            signature=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns)
            json_path=output/'points'/relative.parent/(relative.stem+'.json')
            txt_path=json_path.with_suffix('.txt')
            required=[overlay]+([json_path,txt_path] if args.save_points else [])
            annotation_image=output/'annotations'/relative
            annotation_json=annotation_image.with_suffix('.json')
            if args.save_annotations:
                required += [annotation_image,annotation_json]
            previous=state['completed'].get(key)
            if previous and previous['input']==signature and all(p.is_file() for p in required):
                logger.info('[%d/%d] skip completed %s',index,len(plan),key)
                continue
            if not previous and not(args.overwrite or args.resume) and any(p.exists() for p in required):
                raise FileExistsError(overlay)
            with Image.open(path) as stream:
                image=stream.convert('RGB')
            logger.info('[%d/%d] %s | %dx%d',index,len(plan),key,image.width,image.height)
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            begin=time.perf_counter()
            try:
                points,scores=predict_image(adapter,image,args.mode,args.tile_size,args.overlap,args.amp)
            except torch.OutOfMemoryError:
                logger.error('CUDA OOM. No automatic resizing/tiling/CPU fallback. '
                             'Choose --mode tile, or explicitly try --amp, with a new --output.')
                raise
            torch.cuda.synchronize(device)
            seconds=time.perf_counter()-begin
            peak=torch.cuda.max_memory_allocated(device)/2**30
            save_begin=time.perf_counter()
            point_rows=[dict(x=float(x),y=float(y),score=float(s)) for (x,y),s in zip(points,scores)]
            if args.save_annotations:
                save_annotation_pair(path,annotation_image,annotation_payload(relative.name,image.width,image.height,point_rows,args.annotation_label))
            if args.save_points:
                atomic_json(json_path,dict(image=key,model=args.model,count=len(points),width=image.width,
                    height=image.height,coordinate_order='xy',coordinate_unit='original-image pixels',
                    score_kind='foreground probability',threshold=args.threshold,
                    points=point_rows))
                atomic_text(txt_path,''.join(f'{x:.6f} {y:.6f}\n' for x,y in points))
            save_jpeg(draw_points(image,points,args.radius),overlay,args.jpg_quality)
            state['completed'][key]=dict(input=signature,count=len(points),width=image.width,height=image.height,
                inference_seconds=seconds,save_seconds=time.perf_counter()-save_begin,peak_allocated_gib=peak,
                annotation=str(annotation_json.relative_to(output)) if args.save_annotations else '',
                overlay=str(overlay.relative_to(output)),points=str(json_path.relative_to(output)) if args.save_points else '')
            atomic_json(manifest,state)
            export_counts(output,state,keys)
            logger.info('count=%d | inference=%.2fs | peak=%.2f GiB | saved=%s',len(points),seconds,peak,overlay.name)
        export_counts(output,state,keys)
        logger.info('Finished: %s',output)
    except Exception:
        logger.exception('Stopped; completed outputs retained. Continue with identical settings and --resume.')
        raise


if __name__=='__main__':
    main(parser().parse_args())

