"""Standard transformer modules (port of `lib/transformer/` from dart-torch).

Provides regular softmax-attention encoders, decoders and a full GPT-style
decoder-only transformer — complementing the AFT-based modules under
`dart_cuda.core.transformers.aft`.

Re-exports the most commonly used classes for convenience.
"""

from .transformer_block import TransformerBlock
from .transformer_encoder_block import TransformerEncoderBlock
from .transformer_decoder_block import TransformerDecoderBlock
from .transformer import Transformer
from .transformer_encoder import TransformerEncoder
from .transformer_decoder import TransformerDecoder
from .encoder_decoder_transformer import EncoderDecoderTransformer
from .mtp_module import MTPModule

__all__ = [
    "TransformerBlock",
    "TransformerEncoderBlock",
    "TransformerDecoderBlock",
    "Transformer",
    "TransformerEncoder",
    "TransformerDecoder",
    "EncoderDecoderTransformer",
    "MTPModule",
]
