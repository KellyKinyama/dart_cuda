# Loaders

The loaders depend on **Pillow** for decoding (`pip install dart_cuda[images]`).
All produce float arrays normalized to `[0, 1]` in channels-last RGB layout
`(y, x, c)`.

## `images.image_to_tensor`

```python
from dart_cuda.loaders.images import image_to_tensor

t = image_to_tensor("photo.jpg", target_size=32)   # [1, 32*32*3]
```

## `ImageFolderLoader` — classification + triplets

Expects:

```
<root>/
  <class_0>/  *.jpg|*.png
  <class_1>/  *.jpg|*.png
  ...
```

```python
from dart_cuda.loaders.image_folder_loader import ImageFolderLoader

loader = ImageFolderLoader(
    "Original Images",
    image_size=32,
    patch_size=8,
    val_split=0.2,
    seed=7,
)

print(loader.num_classes, loader.num_train, loader.num_val)
print(loader.num_patches, loader.patch_pixels)

# Classification batch: list of (patchified_flat, label)
batch = loader.sample_train_batch(batch_size=16)
patches, label = batch[0]
# Send to GPU:
import dart_cuda as dc
x = dc.Tensor.from_list([loader.num_patches, loader.patch_pixels], list(patches))

# Validation iterator
for patches, label in loader.val_samples():
    ...

# Triplet sampling for metric learning
trip = loader.sample_triplet()
trip.anchor, trip.positive, trip.negative   # array('f')
```

## `triplet_loader.TripletLoader` — disk-backed

Re-decodes images on every batch (low memory). Returns `Tensor` objects
already on the GPU.

```python
from dart_cuda.loaders.triplet_loader import TripletLoader

loader = TripletLoader("Original Images", image_size=32)
batch = loader.next_batch(batch_size=8)
batch["anchor"], batch["positive"], batch["negative"]   # dc.Tensor
```

## `triplet_loader2.TripletLoader` — RAM-cached

Decodes every image up-front and caches as `array('f')`. Returns flat float
arrays (host-side) — useful when you want to assemble custom batches before
pushing to the GPU.

```python
from dart_cuda.loaders.triplet_loader2 import TripletLoader

loader = TripletLoader("Original Images", image_size=32, num_of_files=200)
batch = loader.next_batch(batch_size=8)
# batch["anchor"] is array('f') of length 8 * 32 * 32 * 3
```

## End-to-end triplet face-embedding example

```python
import dart_cuda as dc
from dart_cuda.loaders.image_folder_loader import ImageFolderLoader
from dart_cuda.core.transformers.vision.vit_face_embedding import ViTFaceEmbeddingGPU
from dart_cuda.core.utils.triplet_loss import TripletLossGPU
from dart_cuda.core.optimizers.adam import Adam

loader = ImageFolderLoader(
    "Original Images", image_size=32, patch_size=8, val_split=0.1,
)
model = ViTFaceEmbeddingGPU(
    imageSize=32, patchSize=8, embedSize=64, outputDim=64, numLayers=2,
)
loss_fn = TripletLossGPU(margin=0.2)
opt = Adam(model.parameters(), lr=1e-3)

shape = [loader.num_patches, loader.patch_pixels]

for step in range(100):
    tracker: list[dc.Tensor] = []
    trip = loader.sample_triplet()

    a = dc.Tensor.from_list(shape, list(trip.anchor))
    p = dc.Tensor.from_list(shape, list(trip.positive))
    n = dc.Tensor.from_list(shape, list(trip.negative))
    tracker += [a, p, n]

    ea = model.get_face_embedding(a, tracker)
    ep = model.get_face_embedding(p, tracker)
    en = model.get_face_embedding(n, tracker)

    loss = loss_fn.forward(ea, ep, en, tracker)
    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 10 == 0:
        print(f"step {step:3d}  loss={loss.fetch_data()[0]:.4f}")
    for t in tracker:
        t.dispose()

opt.dispose()
```
