"""Only load unsafe legacy training checkpoints with explicit user opt-in."""
import argparse
from pathlib import Path
from types import SimpleNamespace


def read_checkpoint(path, trust=False):
    import torch
    try:
        with Path(path).open('rb') as stream:
            state = torch.load(stream,map_location='cpu',weights_only=not trust)
    except Exception as exc:
        if not trust:
            raise RuntimeError('Safe checkpoint load failed. If this is your own trusted training file, '
                               'retry with --trust-checkpoint; otherwise do not enable unsafe pickle loading.') from exc
        raise
    if not isinstance(state,dict):
        raise ValueError('Expected a checkpoint dictionary')
    return state


def select_state(checkpoint, model, weights='student'):
    compact = checkpoint.get('leaf_tip_inference')
    if compact:
        if compact['model'] != model:
            raise ValueError('Checkpoint model type differs from --model')
        if model == 'p2rloss' and compact.get('weights','student') != weights:
            raise ValueError('Compact checkpoint contains a different student/teacher selection')
    key = weights if model == 'p2rloss' and not compact else 'model'
    if key not in checkpoint or not isinstance(checkpoint[key],dict):
        raise ValueError(f'Checkpoint requires {key!r}; plain state_dict is unsupported')
    return checkpoint[key]


def pet_config(checkpoint):
    defaults = dict(backbone='vgg16_bn',position_embedding='sine',dec_layers=2,
                    dim_feedforward=512,hidden_dim=256,dropout=0.,nheads=8)
    saved = checkpoint.get('args',{})
    if isinstance(saved,(argparse.Namespace,SimpleNamespace)):
        saved = vars(saved)
    if not isinstance(saved,dict):
        raise ValueError('Unsupported PET args metadata')
    defaults.update({k:saved[k] for k in defaults if k in saved})
    if defaults['backbone'] != 'vgg16_bn':
        raise ValueError('This PET adapter supports VGG16-BN only')
    return defaults


def compact_checkpoint(checkpoint, model, weights='student'):
    state = select_state(checkpoint,model,weights)
    result = dict(model=state,epoch=checkpoint.get('epoch'),
                  leaf_tip_inference=dict(version=1,model=model,weights=weights if model=='p2rloss' else 'model'))
    if model in ('sah','apgcc'):
        if 'model_config' not in checkpoint:
            raise ValueError('Checkpoint missing model_config')
        result['model_config'] = checkpoint['model_config']
    if model == 'pet':
        result['args'] = pet_config(checkpoint)
    return result
