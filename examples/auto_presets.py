"""
Automatic Optimizer Presets and Parameter Groups

Choose an update algorithm, let the builder identify parameter roles, and
inspect or customize the resulting PyTorch parameter-group dictionaries.

Grouping roadmap:
    auto: Start here. Uses G1 role separation with preset-specific quantization.
    G0:   Minimal grouping: all, or apollo + fallback for APOLLO presets.
    G1:   Separate one_d, embedding, lm_head and other; add apollo when selected.
          The first three groups default to no quantization or weight decay.
    G2:   Separate one_d, embedding, lm_head, matrix and nd; add apollo when selected.
          Embeddings use full-rank, momentum-free updates. Set a lower LR explicitly.

Parameter roles:
    one_d:    Biases, normalization parameters and vectors.
    embedding: Token embeddings; tied token-embedding/LM-head weights appear once.
    lm_head:  Recognized untied language-model output-head weights.
    apollo:   Eligible 2-D weights matching the selected name substrings.
    matrix:   Other 2-D weights, including recognized positional embeddings.
    nd:       Higher-dimensional weights such as convolutions.

Only non-empty groups are returned. G0/G1 combine some roles as described above.
The auto profile leaves one_d states unquantized; token-embedding/output-head
quantization follows the chosen preset. Quantization thresholds still apply.
"""
from torch import nn

from adafactor8bit import Adafactor8Bit, build_param_groups


def main():
    # A small model for inspecting all G2 routes; no training loop is needed.
    model = nn.Module()
    model.token_embed = nn.Embedding(32, 256)
    model.norm = nn.LayerNorm(256)
    model.attn = nn.Linear(256, 256, bias=False)
    model.mlp = nn.Linear(256, 512, bias=False)
    model.output_projection = nn.Linear(512, 256, bias=False)
    model.lm_head = nn.Linear(256, 32, bias=False)
    model.conv = nn.Conv2d(3, 16, kernel_size=3, bias=False)

    groups = build_param_groups(
        model,
        preset="apollo",
        grouping="G2",                 # Keep matrix and N-D routes separate.
        apollo_rank=128,
        apollo_targets=("attn", "mlp"), # Name substrings for eligible 2-D weights.
        embedding_lr=1e-4,              # Set an embedding-specific learning rate.
        embedding_quantize=False,
    )
    optimizer = Adafactor8Bit(groups, lr=1e-3, weight_decay=1e-2, use_cuda_kernel=False)

    # one_d: norms/biases; embedding: token embeddings; apollo: selected matrices;
    # matrix: other 2-D weights; nd: higher-dimensional tensors, when present.
    # Tied embedding/LM-head weights appear once, on the embedding route.
    print(f"{'Group':<12} {'Elements':>10} {'LR':>10} {'Quantize':>10} {'Rank':>6}")
    for group in optimizer.param_groups:
        count = sum(param.numel() for param in group["params"])
        print(f"{group['group_name']:<12} {count:>10,} {group['lr']:>10.1e} "
              f"{str(group['quantize']):>10} {group['apollo_rank']:>6}")

    # Other choices: sgd, adamw, adafactor, rmsprop, came.
    # Plain SGD uses beta1=None; beta1=0.9 enables momentum.
    # Adam: adamw with weight_decay=0. APOLLO-Mini: apollo_rank=1.
    # Continual Adafactor: beta2=0.999, beta2_decay=None.
    # These remain ordinary editable PyTorch parameter groups.
    # Optimizer-state creation still applies min_8bit_size to each parameter.


if __name__ == "__main__":
    main()
