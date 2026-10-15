"""Frozen official kl-f4 VAE, without Lightning/training dependencies.

Encoder/Decoder and tensor weights are taken unchanged from ACPV-Net's ldm.
The wrapper only implements the official posterior sampling and scale factor.
"""
import hashlib
from pathlib import Path
import sys
import types
import torch
from torch import nn
from torch.nn import functional as F

from project_paths import ROOT, SOURCE, DATA, RAW
sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor'))
from ldm.modules.diffusionmodules.model import Encoder, Decoder, AttnBlock

CKPT_SHA='76e617c8b69d692f0a2b8bdfe2ee15ba2460a74645fd8ce965de722cc137ee7a'
SCALE=0.086164


class CheckpointMetadata:
    """Unused Lightning callback metadata; no Lightning code is executed."""
    pass


def sdpa_forward(self,x):
    b,c,h,w=x.shape
    normalized=self.norm(x)
    q,k,v=[layer(normalized).flatten(2).transpose(1,2).unsqueeze(1) for layer in [self.q,self.k,self.v]]
    out=F.scaled_dot_product_attention(q,k,v,dropout_p=0.0,is_causal=False)
    out=out.squeeze(1).transpose(1,2).reshape(b,c,h,w)
    return x+self.proj_out(out)


class FrozenVAE(nn.Module):
    def __init__(self):
        super().__init__()
        config=dict(double_z=True,z_channels=3,resolution=256,in_channels=3,out_ch=3,
                    ch=128,ch_mult=[1,2,4],num_res_blocks=2,attn_resolutions=[],dropout=0.0)
        self.encoder=Encoder(**config)
        self.decoder=Decoder(**config)
        self.quant_conv=nn.Conv2d(6,6,1)
        self.post_quant_conv=nn.Conv2d(3,3,1)
        path=ROOT/'sources/kl-f4/model.ckpt'
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
        if digest.hexdigest()!=CKPT_SHA:raise ValueError('Official VAE checkpoint checksum mismatch')
        # The official checkpoint names a Lightning callback in training metadata.
        # Load tensors with the restricted loader and a inert metadata placeholder.
        callback='pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint'
        unsafe=torch.serialization.get_unsafe_globals_in_checkpoint(path)
        if set(unsafe)-{callback}:raise ValueError(f'Unexpected checkpoint globals: {unsafe}')
        with torch.serialization.safe_globals([(CheckpointMetadata,callback)]):
            document=torch.load(path,map_location='cpu',weights_only=True)
        source=document.get('state_dict',document)
        expected=self.state_dict()
        selected={key:source[key] for key in expected}
        self.load_state_dict(selected,strict=True)
        self.requires_grad_(False);self.eval()
        self.source_sha256=CKPT_SHA

    def use_sdpa(self):
        for module in self.modules():
            if isinstance(module,AttnBlock):module.forward=types.MethodType(sdpa_forward,module)

    def posterior(self,image):
        moments=self.quant_conv(self.encoder(image))
        mean,logvar=moments.chunk(2,dim=1)
        return mean,torch.exp(0.5*logvar.clamp(-30.0,20.0))

    def sample(self,image,seeds):
        mean,std=self.posterior(image)
        # CPU RNG makes each sample independent of GPU/batch ordering and resumption.
        noise=torch.stack([torch.randn(mean.shape[1:],generator=torch.Generator().manual_seed(seed)) for seed in seeds]).to(mean.device)
        return (mean.float()+std.float()*noise)*SCALE

    def decode(self,latent):
        return self.decoder(self.post_quant_conv(latent/SCALE))
