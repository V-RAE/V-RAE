# Third-party runtime and weights

This branch vendors only the runtime subset needed by the V-JEPA 2.1 adapter:
`third_party/vjepa2/`. Its upstream licenses are kept next to the source.

## V-JEPA 2.1 checkpoint

```bash
mkdir -p ckpts/pretrained/encoders/vjepa2_1
curl -fL \
  https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt \
  -o ckpts/pretrained/encoders/vjepa2_1/model.pt
```

## Reconstruction loss weights

The template expects local VGG16 and LPIPS calibration files:

```bash
mkdir -p ckpts/pretrained/perceptual
curl -fL https://download.pytorch.org/models/vgg16-397923af.pth \
  -o ckpts/pretrained/perceptual/vgg16.pt
curl -fL \
  https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/weights-vgg16-lpips.pth \
  -o ckpts/pretrained/perceptual/lpips_vgg.pt
```

## VideoMAE discriminator weights

The reconstruction GAN uses the legacy InternVideo VideoMAE-B checkpoint
`vit_b_hybrid_pt_800e.pth`. The original Hugging Face repository is gated and
requires accepting its access conditions first. The same file is also listed
in the official OpenGVLab ModelScope mirror, which can be downloaded directly:

```bash
mkdir -p ckpts/pretrained/discriminators/videomae
curl -fL \
  https://modelscope.cn/models/OpenGVLab/InternVideo1.0/resolve/master/\
internvideomae_classification/vit_b_hybrid_pt_800e.pth \
  -o ckpts/pretrained/discriminators/videomae/vit_b_hybrid_pt_800e.pth
```

Alternatively, after `hf auth login` and accepting the gated model terms:

```bash
hf download OpenGVLab/InternVideo1.0 \
  internvideomae_classification/vit_b_hybrid_pt_800e.pth \
  --local-dir ckpts/pretrained/discriminators/videomae
mv ckpts/pretrained/discriminators/videomae/\
internvideomae_classification/vit_b_hybrid_pt_800e.pth \
ckpts/pretrained/discriminators/videomae/vit_b_hybrid_pt_800e.pth
```

The final path must match `gan.discriminator_checkpoint` in the reconstruction
configuration.
