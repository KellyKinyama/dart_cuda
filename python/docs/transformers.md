# Transformers

The `core/transformers/` tree contains three families:

- `aft/` — pure transformer encoder & decoder blocks (text-style).
- `vision/` — Vision Transformer backbone and task heads.
- `modalities/` — text / audio / video encoders and multimodal fusion.

## Encoder / decoder

```python
from dart_cuda.core.transformers.aft.transformer_encoder import TransformerEncoder
from dart_cuda.core.transformers.aft.transformer_decoder import TransformerDecoder

encoder = TransformerEncoder(
    vocabSize=4096,
    embedSize=128,
    blockSize=64,
    numLayers=4,
    numHeads=4,
)

decoder = TransformerDecoder(
    vocabSize=4096,
    embedSize=128,
    blockSize=64,
    numLayers=4,
    numHeads=4,
    encoderEmbedSize=128,
)

tracker: list[dc.Tensor] = []
enc_out = encoder.forward(src_ids, tracker)             # src_ids: list[int]
logits  = decoder.forward(tgt_ids, enc_out, tracker)    # [T, vocabSize]
```

`forward_embeddings(x, tracker)` lets you feed a `[T, embedSize]` tensor
directly (used by ViT, audio/video encoders, fusion stacks).

## Vision Transformer

```python
from dart_cuda.core.transformers.vision.vit_backbone import ViTBackbone
from dart_cuda.core.transformers.vision.vit_face_embedding import ViTFaceEmbeddingGPU
from dart_cuda.core.transformers.vision.vit_object_detector import ViTObjectDetector

# Patchified input: [numPatches, patchSize*patchSize*3] (see ImageFolderLoader).
backbone = ViTBackbone(
    imageSize=32, patchSize=8, embedSize=128, numLayers=4,
)
encoded = backbone.forward(patches, tracker)            # [numPatches+1, 128]

face = ViTFaceEmbeddingGPU(
    imageSize=32, patchSize=8, embedSize=128, outputDim=128,
)
emb = face.get_face_embedding(patches, tracker)         # L2-normalized

detector = ViTObjectDetector(
    imageSize=32, patchSize=8, numChannels=3,
    embedSize=128, numLayers=4, numHeads=4,
    numClasses=80, numQueries=50,
)
out = detector.forward(patches, tracker)
out["logits"]   # [numQueries, numClasses + 1]
out["boxes"]    # [numQueries, 4]  (sigmoid-bounded)
```

## Modality encoders

```python
from dart_cuda.core.transformers.modalities.text_transformer  import TextTransformer
from dart_cuda.core.transformers.modalities.audio_transformer import AudioTransformer
from dart_cuda.core.transformers.modalities.video_transformer import VideoTransformer
from dart_cuda.core.transformers.modalities.text_decoder      import TextDecoder

text  = TextTransformer(vocabSize=4096, maxSeqLen=64, embedSize=128)
audio = AudioTransformer(
    featureDim=80, maxSequenceLength=128, embedSize=128, numClasses=10,
)
video = VideoTransformer(
    frameEmbedDim=512, embedSize=128,
    maxVideoSequenceLength=32, numClasses=10,
)

# Decoder for caption-style outputs:
dec = TextDecoder(vocabSize=4096, maxSeqLen=64, embedSize=128)
logits = dec.forward(prompt_ids, encoder_output, tracker)
```

## Multimodal fusion

```python
from dart_cuda.core.transformers.modalities.multi_modal_transformer  import MultimodalTransformer
from dart_cuda.core.transformers.modalities.multi_modal_transformer2 import MultimodalTransformer2
from dart_cuda.core.transformers.modalities.multi_modal_transformer_encoder import MultimodalTransformerEncoder

# audio + video -> classes
mm = MultimodalTransformer(audio_model=audio, video_model=video, numClasses=10)
logits = mm.forward(audio_in, video_in, tracker)

# audio + video + text -> classes
mm2 = MultimodalTransformer2(
    audio_model=audio, video_model=video, text_model=text, numClasses=10,
)
logits = mm2.forward(audio_in, video_in, text_tokens, tracker)

# Joint multimodal stack feeding a downstream decoder
enc = MultimodalTransformerEncoder(
    audio_encoder=audio, video_encoder=video, text_encoder=text,
    jointEmbedSize=128, fusionLayers=2, fusionHeads=4, maxTotalSeqLen=300,
)
fused = enc.forward(audio_in, video_in, prompt_ids, tracker)
caption_logits = dec.forward(target_ids, fused, tracker)
```
