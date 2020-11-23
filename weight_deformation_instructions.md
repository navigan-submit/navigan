## Calculating hessian's eigenvectors
```
python hessian_power_iteration.py \
    --out result \
    --gan_weights stylegan2-car-config-f.pt \
    --resolution 512 \
    --gan_conv_layer_index 3 \
    --num_samples 512 \
    --batch_size 32 \
    --num_eigenvectors 64 \
    --samples_for_videos cars_samples_for_videos.pt
```
## Training
### Rectification over SVD
```
python run_train.py \
    --out results \
    --gan_type StyleGAN2 \
    --gan_weights stylegan2-car-config-f.pt \
    --resolution 512 \
    --shift_predictor_size 256 \
    --deformator_target weight_svd \
    --deformator_conv_layer_index 3 \
    --directions_count 64 \
    --shift_scale 3500 \
    --shift_weight 0.0025 \
    --min_shift 300 \
    --batch_size 16 \
    --n_steps 100001 \
    --make_videos True \
    --samples_for_videos cars_samples_for_videos.pt
```
Some explanations:
* **out** -- folder to store the results;
* **shift_predictor_size** -- images are interpolated to this size before being fed into the shift predictor. This argument can be used to save computational time & GPU-memory consumption.
* **deformator_target** -- weight_svd or weight_fixedbasis (there is an example in the following section).
* **deformator_conv_layer_index** -- number of the convolutional layer to be explored. Zero-based numbering.
* **shift_weight** -- contribution of the shift reconstructor loss to the total loss. This parameter can be used to align shift and classification loss in case of high **shift_scale**.
* If you set **make_videos** to True, then you can also set parameter **video_interpolate**. This parameter controls the output videos resolution. For example, in case of FFHQ 1024 you may want to set **video_interpolate** to 512 in order to save computational time & GPU-memory consumption.
### Rectification over hessian's eigenvectors
```
python run_train.py \
    --out results \
    --gan_type StyleGAN2 \
    --gan_weights stylegan2-car-config-f.pt \
    --resolution 512 \
    --shift_predictor_size 256 \
    --deformator_target weight_fixedbasis \
    --basis_vectors_path eigenvectors_layer3_stylegan2-car-config-f.pt \
    --deformator_conv_layer_index 3 \
    --directions_count 64 \
    --shift_scale 80 \
    --min_shift 15 \
    --batch_size 16 \
    --n_steps 100001 \
    --make_videos True \
    --samples_for_videos /home/a-cherepkov/cars_samples_for_videos.pt
```
Some explanations:
* **basis_vectors_path** -- path to the hessian's eigenvectors.

## Inference
Firstly, import the required modules and load the generator:
```
from inference import GeneratorWithWeightDeformator
from loading import load_generator

G = load_generator(
    args={'resolution': 512, 'gan_type': 'StyleGAN2'},
    G_weights='stylegan2-car-config-f.pt',
    shift_in_w=False
)
```

Secondly, patch the GAN using one of the methods below. 
### SVD
```
G = GeneratorWithWeightDeformator(
    generator=G,
    deformator_type='svd',
    layer_ix=3,
)
```
### Rectification over SVD
```
G = GeneratorWithWeightDeformator(
    generator=G,
    deformator_type='svd_rectification',
    layer_ix=3,
    checkpoint_path='results/checkpoint.pt',
)
```
### Hessian's eigenvectors
```
G = GeneratorWithWeightDeformator(
    generator=G,
    deformator_type='hessian',
    layer_ix=3,
    eigenvectors_path='eigenvectors_layer3_stylegan2-car-config-f.pt'
)
```
### Rectification over hessian's eigenvectors
```
G = GeneratorWithWeightDeformator(
    generator=G,
    deformator_type='hessian_rectification',
    layer_ix=3,
    checkpoint_path='results/checkpoint.pt',
    eigenvectors_path='eigenvectors_layer3_stylegan2-car-config-f.pt'
)
```
Now you can enable deformations for every element in the batch in the following manner:
```
# Generate some samples
zs = torch.randn((4, 512)).cuda()

# Specify deformation index and shift for every sample in the batch
batch_directions = torch.LongTensor([2, 2, 5, 1]).cuda()
batch_shifts = torch.FloatTensor([100, 100, 90, -50]).cuda()
G.deformate(batch_directions, batch_shifts)

# Simply call the generator
imgs_deformated = G(zs)
```

## Saving one deformation into file
You can save the deformation parameters (including layer_ix and data) into one file.
In order to do this:
1. Patch the GAN in the manner described in the paragraph Inference;
2. Call G.save_deformation(path, direction_ix).

## Loading one deformation from file
Firstly, import the required modules and load the generator:
```
from inference import GeneratorWithFixedWeightDeformation
from loading import load_generator

G = load_generator(
    args={'resolution': 512, 'gan_type': 'StyleGAN2'},
    G_weights='stylegan2-car-config-f.pt',
    shift_in_w=False
)
```

Secondly, patch the GAN:
```
G = GeneratorWithFixedWeightDeformation(generator=G, deformation_path='deform.pt')
```

Now you can enable deformations for every element in the batch in the following manner:
```
# Generate some samples
zs = torch.randn((4, 512)).cuda()

# Specify deformation index and shift for every sample in the batch
batch_shifts = torch.FloatTensor([100, 100, 90, -50]).cuda()
G.deformate(batch_shifts)

# Simply call the generator
imgs_deformated = G(zs)
```
