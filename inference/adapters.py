from types import SimpleNamespace

import numpy as np

from .checkpoints import read_checkpoint, select_state, pet_config
from .geometry import padded_size, pet_pixels, grid_pixels, valid_predictions


class Adapter:
    def __init__(self, name, checkpoint, device, threshold=.5, weights='student', trust=False):
        import torch
        self.name, self.device, self.threshold = name, device, threshold
        state = read_checkpoint(checkpoint,trust)
        tensors = select_state(state,name,weights)
        if name == 'sah':
            from vendor.sah.models import SAHCC, ModelConfig
            self.model = SAHCC(ModelConfig(**state['model_config']))
        elif name == 'apgcc':
            from vendor.apgcc.models import APGCC, ModelConfig
            self.model = APGCC(ModelConfig(**state['model_config']))
        elif name == 'p2rloss':
            from vendor.p2rloss.models.vgg16bn import VGG16_BN
            self.model = VGG16_BN(None)
        elif name == 'pet':
            from vendor.pet.models.pet import PET
            from vendor.pet.models.backbones import build_backbone_vgg
            args = SimpleNamespace(**pet_config(state),device=str(device))
            self.model = PET(build_backbone_vgg(args),num_classes=1,args=args)
        else:
            raise ValueError('Unknown model')
        self.model.load_state_dict(tensors,strict=True)
        self.model.to(device).eval()
        self.params = sum(p.numel() for p in self.model.parameters())
        self.epoch = state.get('epoch')

    def predict(self,image,amp=False):
        import torch
        import torch.nn.functional as F
        from .images import normalized_image
        x = torch.from_numpy(normalized_image(image)).to(self.device)
        height,width = image.height,image.width
        with torch.inference_mode(), torch.autocast('cuda',dtype=torch.float16,enabled=amp):
            if self.name == 'pet':
                from vendor.pet.util.misc import nested_tensor_from_tensor_list
                samples = nested_tensor_from_tensor_list([x])
                output = self.model(samples,test=True,threshold=self.threshold)
                h,w = samples.tensors.shape[-2:]
                points = pet_pixels(output['pred_points'][0].float().cpu().numpy(),h,w)
                scores = output['pred_logits'][0].float().softmax(-1)[:,1].cpu().numpy()
            else:
                multiple,minimum = (32,32) if self.name == 'p2rloss' else (16,32)
                h,w = padded_size(height,width,multiple,minimum)
                x = F.pad(x,(0,w-width,0,h-height)).unsqueeze(0)
                output = self.model(x)
                if self.name == 'p2rloss':
                    probability = output[0,0].float().sigmoid()
                    rows,columns = torch.nonzero(probability>self.threshold,as_tuple=True)
                    oh,ow = probability.shape
                    points = grid_pixels(rows.cpu().numpy(),columns.cpu().numpy(),h,w,oh,ow)
                    scores = probability[rows,columns].cpu().numpy()
                else:
                    points = output['pred_points'][0].float().cpu().numpy()
                    logits = output['pred_logits'][0].float()
                    scores = (logits.sigmoid().squeeze(-1) if self.name == 'sah'
                              else logits.softmax(-1)[:,1]).cpu().numpy()
        return valid_predictions(points,scores,width,height,self.threshold)
