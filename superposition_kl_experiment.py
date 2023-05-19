""" This script demonstrates how to use the algebraic_value_editing library to generate comparisons
between two prompts. """

# %%
from typing import List

import torch as t
import matplotlib.pyplot as plt
from tqdm import tqdm
from einops import einsum, rearrange, repeat
from transformer_lens.HookedTransformer import HookedTransformer

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
# #boilerplate example code from the original file
# activation_additions: List[ActivationAddition] = [
#     *get_x_vector(
#         prompt1="I",
#         prompt2="Hate",
#         coeff=0.36,
#         act_name=6,
#         model=model,
#         pad_method="tokens_right",
#     ),
# ]

# completion_utils.print_n_comparisons(
#     prompt="I am the",
#     num_comparisons=5,
#     model=model,
#     activation_additions=activation_additions,
#     seed=0,
#     temperature=1,
#     freq_penalty=1,
#     top_p=0.3,
# )
# %%
#returns a tensor of shape (50257,) with pre-softmax logits 
#from applying (prompt1-prompt2) at a strength of x to baseline_prompt at the given layer
def logits_from_shifted_prompt(baseline_prompt, prompt1, prompt2,x,layer,
                               model=model,overwriting = 'none',token_index=-1):
    extra_hooks = {}
    if overwriting == 'attention':
        extra_hooks = overwriting_hook_utils.generate_overwriting_hook_fns(
            model = model,
            baseline_prompt = baseline_prompt,
            act_name = (lambda layer_name: 'hook_attn_out' in layer_name)
        )
    elif overwriting == 'mlp':
        extra_hooks = overwriting_hook_utils.generate_overwriting_hook_fns(
            model = model,
            baseline_prompt = baseline_prompt,
            act_name = (lambda layer_name: 'hook_mlp_out' in layer_name)
        )
    activation_additions: List[ActivationAddition] = [
        *get_x_vector(
            prompt1=prompt1,
            prompt2=prompt2,
            coeff=x,
            act_name=layer,
            model=model,
            pad_method="tokens_right",
        ),
    ]

    mod_df = completion_utils.gen_using_activation_additions(
        prompt_batch=[baseline_prompt],
        tokens_to_generate = 0, #just looking at the next token
        model=model,
        activation_additions=activation_additions,
        extra_hooks = extra_hooks,
        addition_location="front",
        res_stream_slice=slice(None),
        seed=0,
        temperature=1,
        freq_penalty=1,
        top_p=0.3,
        include_logits = True,
    )
    mod_logits = mod_df['logits'].iloc[0][token_index]
    return t.tensor(mod_logits)
#same as above, but returns a tensor of shape (49,num_tokens,1600) with each res)idual stream
#(49 because we have pre-layer-0 and post-layer-47)
def modified_cache_tensor(baseline_prompt, prompt1, prompt2,x,layer,
                               model=model,token_index=-1):
    activation_additions: List[ActivationAddition] = [
        *get_x_vector(
            prompt1=prompt1,
            prompt2=prompt2,
            coeff=x,
            act_name=layer,
            model=model,
            pad_method="tokens_right",
        ),
    ]

    mod_df = completion_utils.gen_using_activation_additions(
        prompt_batch=[baseline_prompt],
        tokens_to_generate = 0, #just looking at the next token
        model=model,
        activation_additions=activation_additions,
        addition_location="front",
        res_stream_slice=slice(None),
        seed=0,
        temperature=1,
        freq_penalty=1,
        top_p=0.3,
        include_full_cache = True,
    )
    return mod_df['full_cache'].iloc[0]

def kl_divergence(new_logits,baseline,sum_up = True):
    #new_logits is a tensor of shape (n,50257)
    #baseline_logits is of shape (50257,)
    log_scores = t.nn.functional.kl_div(
        t.nn.functional.log_softmax(new_logits,dim=-1),
        t.nn.functional.softmax(baseline,dim=-1),
        reduction='none'
    )
    if sum_up:
        return log_scores.sum(dim=-1)
    else:
        return log_scores

def track_differences(baseline_prompt,prompt1,prompt2,spike_center, spike_width,
                      plotshape=None,difference_mode='tangent',clean_tokens=True):
    """
    For each nonzero xval, plot the differences from the residual streams of the baseline 
    to those of the modified prompt by token index and layer.
    """
    xvals = [spike_center - spike_width, spike_center, spike_center + spike_width]
    caches = {}
    n = len(xvals)
    if plotshape is None:
        plotshape = (1,n)
    for x in [0]+list(xvals):
        caches[x] = modified_cache_tensor(baseline_prompt,prompt1,prompt2,x=x,layer=6)

    #this is of shape (49,1600)
    #we want to imshow it with an aspect ratio that makes a square output
    baseline_tokens = model.tokenizer.tokenize('<|endoftext|>'+baseline_prompt)
    if clean_tokens:
        baseline_tokens = [s.replace('Ġ','_') for s in baseline_tokens]
    plt.figure(figsize=(12,7))
    for i in range(n):
        plt.subplot(1,n,i+1)
        if difference_mode == 'tangent':
            distances = t.norm(caches[0]-caches[xvals[i]],dim=-1)/baseline_norms
        elif difference_mode == 'cosine':
            distances = t.cosine_similarity(caches[0],caches[xvals[i]],dim=-1)
        plt.imshow(distances,origin='lower',aspect='auto')
        plt.xticks(range(len(baseline_tokens)),baseline_tokens,rotation=45)
        plt.colorbar()
        plt.title(f"Baseline to x={xvals[i]:.2f}")
        plt.suptitle("Residual stream differences between baseline and steered prompt, by token position and layer\n"
                     f"{prompt1} - {prompt2} at layer 6")
    plt.show()


def get_logits_by_intervention_layer(model,baseline_prompt,prompt1,prompt2, xrange,
                                     use_tqdm = True, n_layers = 48,token_index=-1):
    """
    Outputs a (n_layers, len(xrange), 50257) tensor with logits for each intervention by layer and lambda.
    """
    layer_iterator = (tqdm(range(n_layers)) if use_tqdm else range(n_layers))

    output = t.zeros((n_layers,len(xrange),50257))
    for layer in layer_iterator:
        for i,x in enumerate(xrange):
            output[layer,i] = logits_from_shifted_prompt(baseline_prompt,prompt1,prompt2,x,layer,
                                                         model=model,token_index=token_index)
    return output

# %%
def entropy(logits):
    post_softmax = t.nn.functional.softmax(logits,dim=-1)
    return -(post_softmax * t.log(post_softmax)).sum(dim=-1)
# %%
baseline_prompt = "I hate you because you are the"
variant_prompt = "I love you because you are the"
prompt1 = "I"
prompt2 = "Hate"
intended_prompt_length = 2
# assert len(model.tokenizer.tokenize(prompt1)) == intended_prompt_length,model.tokenizer.tokenize(prompt1)
# assert len(model.tokenizer.tokenize(prompt2)) == intended_prompt_length,model.tokenizer.tokenize(prompt2)


# %%
"""
Assorted things one can explore with a prompt pair
"""
baseline_norms = t.norm(modified_cache_tensor(baseline_prompt,prompt1,prompt2,x=0,layer=6),dim=-1)
track_differences(baseline_prompt,prompt1,prompt2,spike_center = 0.18, spike_width = 0.05)
# %%
"""Measuring the angle between small differences and large ones
"""
all_caches = []
for x in [0,0.13,0.1508,0.17,0.1]:
    mct = modified_cache_tensor(baseline_prompt,prompt1,prompt2,x=x,layer=6)[:,target_token_index]
    all_caches.append(mct)
all_caches = t.stack(all_caches,dim=0)
all_caches.shape
# %%
before_diffs = all_caches[1] - all_caches[0]
spike_diffs = all_caches[2] - all_caches[0]
after_diffs = all_caches[3] - all_caches[0]
cosines = t.cosine_similarity(before_diffs,after_diffs,dim=-1)
plt.plot(cosines,label="before vs after")
plt.plot(t.cosine_similarity(before_diffs,spike_diffs,dim=-1),label="before vs spike")
plt.plot(t.cosine_similarity(spike_diffs,after_diffs,dim=-1),label="spike vs after")
plt.legend()
plt.show()
# %%
#store the distances from baseline for a single token position across x and layer
xrange = t.arange(0.13,0.25,0.003)
target_token_index = 1
layer = 6
baseline_at_target = modified_cache_tensor(baseline_prompt,prompt1,prompt2,x=0,layer=6)[:,target_token_index]
baseline_at_target_norms = t.norm(baseline_at_target,dim=-1)
all_distances = []
for x in tqdm(xrange):
    mct = modified_cache_tensor(baseline_prompt,prompt1,prompt2,x=x,layer=layer)[:,target_token_index]
    distances = t.norm(mct-baseline_at_target,dim=-1)/baseline_at_target_norms
    all_distances.append(distances)
all_distances = t.stack(all_distances,dim=0)
all_distances.shape
# %%
for layer in [10,12,13,14,15,16,17,18,19,20]:
    plt.plot(xrange,all_distances[:,layer],label=f'layer {layer}')
plt.legend()
plt.yscale('log')
plt.title(f"Distance from baseline for token {target_token_index} at different layers")
plt.xlabel("x")
plt.ylabel("distance")
plt.show()
# %%
plt.imshow(t.log(all_distances[:,6:].T),origin='lower',aspect='auto')
plt.xlabel("x")
plt.ylabel("layer")
#x axis spans xrange, y axis goes from 6 to 48
#add relevant ticks (not everywhere, just at reasonable spacing)
plt.xticks(range(0,len(xrange),5),[f'{x.item():.3f}' for x in xrange[::5]],rotation=45)
plt.yticks(range(0,43,5),range(6,48,5))

plt.colorbar()
plt.title(f"Log distance from baseline for token {target_token_index} at different layers")
plt.show() 
# %%

def detect_spike(lambda_range, quantity, use='entropy'):
    """
    Returns (spike location, spike slope) by finding the largest diff between two
    consecutive values of quantity.
    """
    if use != 'entropy':
        raise NotImplementedError
    x_res = lambda_range[1] - lambda_range[0]
    diffs = t.diff(quantity, prepend=t.tensor([quantity[0]])) * (1 / (lambda_range[1] - lambda_range[0]))
    # Take differences between the list, and find the first place where slope > slope_threshold
    argmax_slope = t.abs(diffs).argmax()
    return lambda_range[argmax_slope].item() - x_res / 2, diffs[argmax_slope].item()

detect_spike(lambda_range,entropy(all_logits))


# %%
baseline_prompt = "I"
baseline_logits =t.tensor(completion_utils.gen_using_model(
    model = model,
    prompt_batch=[baseline_prompt],
    tokens_to_generate=0,
    seed = 0, #this doesn't matter since we look at logits
    include_logits=True,
)['logits'].iloc[0][-1]) #taking just the last token
# variant_prompt = "I love you because you are the"
prompt1 = "<|endoftext|>"
prompt2 = "I"
intended_prompt_length = 2
actual_steerprompt_length = tuple(len(model.tokenizer.tokenize(p)) for p in [prompt1,prompt2])
print(f"Using prompts of length {actual_steerprompt_length}")
# assert len(model.tokenizer.tokenize(prompt1)) == intended_prompt_length,model.tokenizer.tokenize(prompt1)
# assert len(model.tokenizer.tokenize(prompt2)) == intended_prompt_length,model.tokenizer.tokenize(prompt2)

logit_list = []
lambda_range = t.arange(0.,1,0.02) #vector scaling coefficient
for x in tqdm(lambda_range):
    logit_list.append(logits_from_shifted_prompt(
        baseline_prompt, prompt1, prompt2,x=x,layer=layer))
all_logits = t.stack(logit_list,dim=0)
all_logits.shape #should be (50,50257)

# %%


def logit_diff_plot(lambda_range:t.Tensor,
                    all_logits:t.Tensor, # (lambda_range, vocab_size)
                    tokens_of_interest:list[str], use_logprobs=False) -> None:
    """
    Plots logit difference for 4 tokens, entropy, and KL divergence
    as a function of the steering coefficient lambda.
    Requires all_logits to already be created with range lambda_range.
    """
    if use_logprobs:
        all_logits = t.nn.functional.log_softmax(all_logits,dim=-1)
        logit_or_logprob = "logprob"
    else:
        logit_or_logprob = "logit"
    plt.figure(figsize=(12,8))
    for i,tok in enumerate(tokens_of_interest):
        plt.subplot(2,3,i+1)
        tok_idx = model.tokenizer.encode(tok)[0]
        plt.plot(lambda_range,all_logits[:,tok_idx])
        line_x = 0.36
        min_val = all_logits[:,tok_idx].min()
        max_val = all_logits[:,tok_idx].max()
        #plt.plot([line_x,line_x],[min_val,max_val],color='red')
        plt.title(f"{logit_or_logprob} of |{tok}|")
        if i==3:
            plt.xlabel("lambda")
        #plt.ylabel("logit")
    # now plot entropy and KL
    plt.subplot(2,3,5)
    plt.plot(lambda_range,entropy(all_logits))
    plt.title("Entropy")
    plt.xlabel("lambda")
    line_x, spike_slope = detect_spike(lambda_range,entropy(all_logits))
    min_val, max_val = plt.ylim()
    plt.plot([line_x,line_x],[min_val,max_val],color='red')
    plt.text(line_x, (max_val + min_val)/2, f"spike at {line_x:.5f}\n with slope {spike_slope:.2f}")
    plt.subplot(2,3,6)
    plt.plot(lambda_range,kl_divergence(all_logits,baseline_logits))
    plt.title("KL Divergence")
    plt.xlabel("lambda")
    #add an overall title
    plt.suptitle(f"""{logit_or_logprob}s for baseline prompt: "{baseline_prompt}"\nIntervening with lambda*(|{prompt1}| - |{prompt2}|). (length={actual_steerprompt_length})""")
    plt.show()

tokens_of_interest = [' worst',' best',' only',' apple']
logit_diff_plot(lambda_range,all_logits,tokens_of_interest)

# %%
logit_diff_plot(lambda_range,all_logits,tokens_of_interest,use_logprobs=True)

# %%
#Look at which tokens get upweighted by the spike
spike_measurement = 0.36
non_spike_measurement = 0.42
token_index = -2
spike_logits = logits_from_shifted_prompt(baseline_prompt,prompt1,prompt2,x=spike_measurement,
                                          layer=6,token_index = token_index)
non_spike_logits = logits_from_shifted_prompt(baseline_prompt,prompt1,prompt2,x=non_spike_measurement,
                                              layer=6, token_index = token_index)
divergences = kl_divergence(spike_logits,baseline_logits,sum_up=False) - kl_divergence(non_spike_logits,baseline_logits,sum_up=False)
#divergences is of shape (50257,)
print(divergences.sum())
for index in divergences.topk(5).indices:
    print(model.tokenizer.decode([index]),divergences[index].item())
# %%
# How do token logits change when the steering vector intervention layer varies?
intervention_logits = get_logits_by_intervention_layer(model,baseline_prompt,prompt1,prompt2,
                                                       xrange = t.arange(0,1,0.1))
tokens_of_interest = [' wedding']
for tok in tokens_of_interest:
    #3D plot of the logit of each token as a function of lambda, by layer
    plot_values = intervention_logits[:,:,model.tokenizer.encode(tok)[0]]
    #plot_values is of shape (48,100)
    plt.figure(figsize=(12,8))
    plt.imshow(plot_values.T,origin='lower',aspect='auto')
    plt.colorbar()
    plt.title(f"Logit of |{tok}|")
    plt.xlabel("layer")
    plt.ylabel("lambda")
    plt.show()


# %%
token_position = 1
token_logits = t.stack([logits_from_shifted_prompt(baseline_prompt,prompt1,prompt2,x,
                                          layer=6,token_index = token_position)
                                          for x in tqdm(lambda_range)],dim=0)
token_baseline_logits = baseline_logits[token_position]
plt.plot(lambda_range,kl_divergence(token_logits,baseline = variant_logits),label="variant")
plt.plot(lambda_range,kl_divergence(token_logits,baseline = baseline_logits),label="baseline")
plt.legend()
plt.title(f"KL divergence between baseline and variant (token_index = {token_position})")
plt.xlabel("lambda")
plt.ylabel("KL divergence")
plt.show()

# %%
# KL divergence between before-spike and after-spike logits
reshaped_logits_A = intervention_logits[6].unsqueeze(0)
reshaped_logits_B = intervention_logits[6].unsqueeze(1)
all_kl_divergences = kl_divergence(reshaped_logits_A,baseline = reshaped_logits_B)
plt.imshow(all_kl_divergences,origin='lower',aspect='auto')
plt.colorbar()
plt.show()
# %%
"""
Things Drake is curious to explore from here:
* We think that if we intervene at layer 0 with x=1 (and a vector from 
  the current token to a new token)it should be just like running a 
  clean prompt. Is that true?
     - looks like yes
* We think that prompt1,prompt2 = (_ love, _ hate) should be the same as 
  (? love, ? hate) or any other first token. Is that true?
   - oh wait no, actually that's false, nvm (embedding on the second token should be different)
* How do these plots vary based on which token we do the intervention on?
* Which layer?
* What if we look at projections of the final layernorm vector onto the space spanned by
  the embeddings of " best" and " worst" - how does it move through that space?
* In this example, which tokens spike the most at 0.4? Why? 
  Is the spike definitely in the same place for each token that behaves weirdly there?
  Why does the change stick around for | worst| but not other tokens?
* What if we take x out to like 10? Down to -5? 
"""
# %%
#testing interv???
activation_additions: List[ActivationAddition] = [
    *get_x_vector(
        prompt1=prompt1,
        prompt2=prompt2,
        coeff=1,
        act_name=0,
        model=model,
        pad_method="tokens_right",
    ),
]

mod_df = completion_utils.gen_using_activation_additions(
    prompt_batch=[baseline_prompt],
    tokens_to_generate = 0, #just looking at the next token
    model=model,
    activation_additions=activation_additions,
    addition_location="front",
    res_stream_slice=slice(None),
    seed=0,
    temperature=1,
    freq_penalty=1,
    top_p=0.3,
    include_logits = True,
)
mod_logits = mod_df['logits'].iloc[0][-1]
# %%

