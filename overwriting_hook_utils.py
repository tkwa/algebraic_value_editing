""" Utilities for hooking into a model and modifying activations. """

from typing import List, Callable, Optional, Dict, Tuple, Union, Any
from collections import defaultdict
from jaxtyping import Float, Int
import torch
from einops import reduce
import matplotlib.pyplot as plt

from transformer_lens import ActivationCache
from transformer_lens.utils import get_act_name
from transformer_lens.HookedTransformer import HookedTransformer, Loss
from transformer_lens.hook_points import HookPoint, LensHandle
from algebraic_value_editing.prompt_utils import (
    ActivationAddition,
    pad_tokens_to_match_activation_additions,
    get_block_name,
)

#Extra function to do manual overwrites
def activation_overwriting_hook_fn(
    activations_to_overwrite
):
    def prompt_hook(normal_activations,hook):
        assert normal_activations.shape[1] >= activations_to_overwrite.shape[1]
        max_pos = activations_to_overwrite.shape[1]
        assert activations_to_overwrite.shape[0] == normal_activations.shape[0]
        assert activations_to_overwrite.shape[2:] == normal_activations.shape[2:]
        normal_activations[:,:max_pos] = activations_to_overwrite[:,:max_pos]
        return normal_activations
    return prompt_hook
    

def generate_overwriting_hook_fns(model: HookedTransformer, 
                                  baseline_prompt: str,
                                  act_name: Union[str,Callable]) -> Dict[str, Callable]:
    if type(act_name)==str:
        act_name = (lambda s: s==act_name)
    _, cache = model.run_with_cache(baseline_prompt)
    target_keys = [k for k in cache.keys() if act_name(k)]
    assert len(target_keys) > 0, "No activations matched the provided act_name"
    output = {}
    for key in target_keys:
        output[key] = activation_overwriting_hook_fn(cache[key])
    return output

def neuron_ablating_at_token_hook_fn(
        neuron_indices_to_ablate: List[int],
        token_index: int,
):
    def prompt_hook(normal_activations,hook):
        normal_activations[:, token_index, neuron_indices_to_ablate] = 0
        return normal_activations
    return prompt_hook

def eot_scaling_neurons(model: HookedTransformer,
                        cosine_quantile: float = 0.99,
                        norm_quantile: float = 0.99,
                        layers: Optional[List[int]] = None,
                        n_layers: int = 48,
                        ):
    """
    Returns a dictionary from integers (layer indices) to a list of neuron indices
    which have cosine_quantile and norm_quantile above the specified thresholds. 

    Currently this function eats like 4GB of memory and doesn't give it back.
    TODO: fix this
    """
    if layers is None:
        layers = range(n_layers)
    _, cache = model.run_with_cache("")
    eot_cache = []
    for activation_name, value in cache.items():
        if "hook_resid_mid" in activation_name:
            eot_cache.append(value[0,0])
    eot_cache = torch.stack(eot_cache, dim=0).detach()
    input_weights = model.W_in
    output_weights = model.W_out
    layer_to_neuron_indices = {}
    for layer in layers:
        eot_embedding = eot_cache[layer]
        input_weight = input_weights[layer]
        output_weight = output_weights[layer]
        cosine_similarities = torch.cosine_similarity(eot_embedding.unsqueeze(-1), input_weight, dim=0)
        #print(eot_embedding.shape, input_weight.shape, cosine_similarities.shape)
        norms = output_weight.norm(dim=-1)
        cosine_threshold = cosine_similarities.quantile(cosine_quantile)
        norm_threshold = norms.quantile(norm_quantile)
        layer_to_neuron_indices[layer] = torch.where((cosine_similarities > cosine_threshold) & (norms > norm_threshold))[0]
    del eot_cache
    del input_weights
    del output_weights
    return layer_to_neuron_indices


def generate_neuron_ablation_hook_fns(model, 
                                    layer_to_neuron_indices: Dict[int, torch.tensor],
                                    token_index: int,
                                    ):
    """
    Generates a dictionary of hook functions that ablate the given neurons.
    """
    output = {}
    #collect all activation names with post-activation gelu values
    for layer,indices in layer_to_neuron_indices.items():
        act_name = get_act_name("post",layer)
        output[act_name] = neuron_ablating_at_token_hook_fn(layer_to_neuron_indices[layer], token_index)
    return output
