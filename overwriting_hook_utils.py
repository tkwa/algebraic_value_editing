""" Utilities for hooking into a model and modifying activations. """

from typing import List, Callable, Optional, Dict, Tuple, Union, Any
from collections import defaultdict
from jaxtyping import Float, Int
import torch
from einops import reduce

from transformer_lens import ActivationCache
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
