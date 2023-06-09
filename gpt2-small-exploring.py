# %%
from typing import List

import torch as t
import matplotlib.pyplot as plt
from tqdm import tqdm
from einops import einsum, rearrange, repeat
from transformer_lens.HookedTransformer import HookedTransformer
from typing import Callable

from algebraic_value_editing import completion_utils, utils, hook_utils 
import overwriting_hook_utils
from algebraic_value_editing.prompt_utils import (
    ActivationAddition,
    get_x_vector,
)

utils.enable_ipython_reload()

# %%
model: HookedTransformer = HookedTransformer.from_pretrained(
    model_name="gpt2-xl",
    device="cpu",
)
_ = model.to("cuda")

# %%
#collecting EOT token activations in a forward pass
vocab_size = 50257
n_layers = 48
_, cache = model.run_with_cache("<|endoftext|>")
eot_cache = []
for activation_name, value in cache.items():
    if "hook_resid_mid" in activation_name:
        eot_cache.append(value[0,-2])
eot_cache = t.stack(eot_cache, dim=0).detach().cpu()
eot_cache.shape #should be (n_layers, resid_dim=1024)
# %%
plt.plot(t.norm(eot_cache, dim=-1).cpu().numpy())
plt.yscale('log')
plt.title("EOT token activation norms by layer")
plt.show()
# %%
input_weights = []
output_weights = []
for param_name, param in model.named_parameters():
    if 'mlp.W_in' in param_name: 
        input_weights.append(param)
    if 'mlp.W_out' in param_name:
        output_weights.append(param)
input_weights = t.stack(input_weights, dim=0).detach().cpu()
output_weights = t.stack(output_weights, dim=0).detach().cpu()
input_weights.shape, output_weights.shape
# %%
layer = 5
k = 5
layer_quantiles = []
for layer in range(n_layers):
    eot_embedding = eot_cache[layer]
    input_weight = input_weights[layer]
    output_weight = output_weights[layer]
    cosine_similarities = t.cosine_similarity(input_weight, eot_embedding.unsqueeze(-1), dim=0)
    #compute the indices of the k largest cosine similarities
    top_k_indices = cosine_similarities.topk(k=k, dim=0).indices
    norms = t.norm(output_weight, dim=-1)
    avg_norm_of_topk = norms[top_k_indices].mean()
    #compute the quantile of this average norm amongst all norms
    quantile = (norms < avg_norm_of_topk).float().mean()
    layer_quantiles.append(quantile/(1-quantile))
# %%
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
eot_norms = t.norm(eot_cache, dim=-1)
eot_norm_changes = eot_norms[1:]/eot_norms[:-1]
plt.plot(range(1,n_layers-2), eot_norm_changes[1:-1])
plt.title("Relative change in EOT token\nactivation norms by layer")
plt.subplot(1,2,2)
plt.plot(range(n_layers), layer_quantiles)
plt.xlabel("Layer")
plt.title("Relative strength of the top 5 write neurons\nby cosine similarity to EOT token")
plt.show()
# %%
layer = 16
k = 5

eot_embedding = eot_cache[layer]
input_weight = input_weights[layer]
output_weight = output_weights[layer]
cosine_similarities = t.cosine_similarity(input_weight, eot_embedding.unsqueeze(-1), dim=0)
#compute the indices of the k largest cosine similarities
top_k_indices = cosine_similarities.topk(k=k, dim=0).indices
norms = t.norm(output_weight, dim=-1)
avg_norm_of_topk = norms[top_k_indices].mean()
#compute the quantile of this average norm amongst all norms
quantile = (norms < avg_norm_of_topk).float().mean()
plt.scatter(cosine_similarities.cpu().numpy(), norms.cpu().numpy(),s=1)
plt.title(f"Layer {layer} cosine similarities vs output weight norms")
plt.show()
# %%
