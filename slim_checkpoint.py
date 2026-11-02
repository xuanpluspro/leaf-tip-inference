"""Extract one inference model on CPU; no model construction, training or dtype conversion."""
import argparse
from pathlib import Path
import uuid

from inference.checkpoints import read_checkpoint,select_state,compact_checkpoint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,choices=['pet','p2rloss','apgcc','sah'])
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--p2r-weights',choices=['student','teacher'],default='student')
    p.add_argument('--trust-checkpoint',action='store_true')
    args=p.parse_args()
    if args.weights.resolve()==args.output.resolve() or args.output.exists():
        raise FileExistsError('Use a new output file; original is retained')
    import torch
    state=read_checkpoint(args.weights,args.trust_checkpoint)
    result=compact_checkpoint(state,args.model,args.p2r_weights)
    original=select_state(state,args.model,args.p2r_weights)
    result['model']={k:v.detach().cpu().clone() for k,v in original.items()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temp=args.output.with_name(args.output.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temp.open('wb') as stream:
            torch.save(result,stream)
        saved=read_checkpoint(temp)
        if saved['model'].keys()!=original.keys():
            raise ValueError('Model key mismatch')
        for name,tensor in original.items():
            actual=saved['model'][name]
            if tensor.shape!=actual.shape or tensor.dtype!=actual.dtype:
                raise ValueError(f'Tensor metadata mismatch: {name}')
            torch.testing.assert_close(actual,tensor,rtol=0,atol=0,equal_nan=True)
        temp.replace(args.output)
    finally:
        temp.unlink(missing_ok=True)
    print(f'{args.weights.stat().st_size/2**20:.2f} -> {args.output.stat().st_size/2**20:.2f} MiB')
    print(f'Weights verified, source retained, no precision conversion: {args.output}')


if __name__=='__main__':
    main()
