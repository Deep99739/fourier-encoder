import torch
import torch.nn as nn
import types
import weakref

from models.adapter import ParallelAdapter

import inspect
from transformers import PreTrainedModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import AutoModel, GPT2Model, GPT2LMHeadModel, LlamaForCausalLM, BertModel
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions


def add_parallel_adapters(
        model, 
        adapter_type="linear", 
        rank=8, 
        scaling=1.0
    ):
    """
    Add parallel adapters to any transformer-based model, using an external embedding layer.
    
    Args:
        model: The pre-trained transformer model
        adapter_type: Type of adapter ("linear" or "affine" or "low_rank")
        rank: Rank size for low-rank adapters
        scaling: Scaling factor for adapter outputs
    """
    # Get model configuration
    if hasattr(model, "config"):
        hidden_size = model.config.hidden_size
    else:
        # Try to infer hidden size
        for module in model.modules():
            if isinstance(module, nn.Linear) and module.in_features == module.out_features:
                hidden_size = module.in_features
                break
        else:
            raise ValueError("Could not determine hidden size from model.")
    
    # Find transformer blocks
    transformer_blocks = []
    
    # Common architectures
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        transformer_blocks = model.encoder.layer  # BERT style
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        transformer_blocks = model.transformer.h  # GPT style
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        transformer_blocks = model.model.layers  # LLaMA, OPT style
    elif hasattr(model, "layers"):
        transformer_blocks = model.layers  # Some decoder-only models
    
    # Try to detect blocks using heuristics if not found
    if not transformer_blocks:
        for name, module in model.named_children():
            if any(x in name.lower() for x in ["block", "layer", "transformer"]):
                if isinstance(module, nn.ModuleList) or (hasattr(module, "__len__") and len(module) > 1):
                    transformer_blocks = module
                    break
    
    if not transformer_blocks:
        raise ValueError("Could not detect transformer blocks in the model.")
    
    # Add adapters to each transformer block
    for block in transformer_blocks:
        # Find attention and feed-forward modules
        attention_module = None
        ffn_module = None
        
        # Common names for these modules
        attention_names = ["attn", "attention", "self_attn", "self_attention"]
        ffn_names = ["mlp", "ffn", "feed_forward", "ff"]
        
        for name, module in block.named_children():
            if any(attn_name in name.lower() for attn_name in attention_names):
                attention_module = module
            elif any(ffn_name in name.lower() for ffn_name in ffn_names):
                ffn_module = module
        
        # Add adapters if components were found
        if attention_module:
            block.attn_adapter = ParallelAdapter(
                hidden_size,
                adapter_type=adapter_type,
                rank=rank,
                scaling=scaling
            )
        
        if ffn_module:
            block.mlp_adapter = ParallelAdapter(
                hidden_size,
                adapter_type=adapter_type,
                rank=rank,
                scaling=scaling
            )
    
    # Modify the model's forward method to compute adapter embeddings
    if not hasattr(model, "original_forward"):
        model.original_forward = model.forward
        
        def new_model_forward(self, *args, **kwargs):
            # Get input_ids
            input_ids = kwargs.get("input_ids", None)
            # Get fourier_embeddings
            fourier_embeddings = kwargs.get("fourier_embeddings", None)

            if input_ids is None and len(args) > 0:
                input_ids = args[0]
            
            # Check if fourier_embeddings exists and is a tensor
            if fourier_embeddings is not None and isinstance(fourier_embeddings, torch.Tensor):
                has_fourier_embeddings = (fourier_embeddings != 0).any()
                if has_fourier_embeddings:
                    self._adapter_embeddings = fourier_embeddings
                else:
                    self._adapter_embeddings = None
            else:
                self._adapter_embeddings = None
            
            # Call the original forward pass
            outputs = self.original_forward(*args, **kwargs)
            
            return outputs
        
        model.forward = types.MethodType(new_model_forward, model)
    
    # Modify each transformer block's forward method
    for block in transformer_blocks:
        if not hasattr(block, "original_forward"):
            block.original_forward = block.forward
            
            def new_block_forward(self, hidden_states, *args, **kwargs):
                # Call original forward method
                try:
                    if len(inspect.signature(self.original_forward).parameters) == 1:
                        outputs = self.original_forward(hidden_states)
                    else:
                        outputs = self.original_forward(hidden_states, *args, **kwargs)
                except Exception:
                    outputs = self.original_forward(hidden_states, *args, **kwargs)
                
                # Extract hidden states from outputs
                if isinstance(outputs, tuple):
                    out = outputs[0]
                else:
                    out = outputs
                
                # Apply adapters using external embeddings
                if hasattr(self, '_model_ref') and self._model_ref() is not None:
                    model = self._model_ref()
                    if hasattr(model, '_adapter_embeddings') and model._adapter_embeddings is not None:
                        if hasattr(self, "attn_adapter"):
                            # Ensure embeddings are on the same device as the adapter
                            adapter_embeddings = model._adapter_embeddings.to(self.attn_adapter.adapter.weight.device if hasattr(self.attn_adapter, "adapter") else self.attn_adapter.down_proj.weight.device)
                            attn_adapter_output = self.attn_adapter(adapter_embeddings)
                            out = out + attn_adapter_output
                        
                        if hasattr(self, "mlp_adapter"):
                            # Ensure embeddings are on the same device as the adapter
                            adapter_embeddings = model._adapter_embeddings.to(self.mlp_adapter.adapter.weight.device if hasattr(self.mlp_adapter, "adapter") else self.mlp_adapter.down_proj.weight.device)
                            mlp_adapter_output = self.mlp_adapter(adapter_embeddings)
                            out = out + mlp_adapter_output
                
                # Update outputs
                if isinstance(outputs, tuple):
                    outputs = (out,) + outputs[1:]
                else:
                    outputs = out
                
                return outputs
            
            # Create a closure to preserve the block reference
            def make_forward(block):
                return types.MethodType(new_block_forward, block)
            
            block.forward = make_forward(block)
            block._model_ref = weakref.ref(model)  # Use weak reference instead of direct reference
    
    # Freeze all parameters except adapters and adapter embedding
    for name, param in model.named_parameters():
        if 'adapter' not in name and 'adapter_embedding' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    return model


def add_parallel_adapters_to_gpt2(
        model, 
        adapter_type="low_rank", 
        rank=8, 
        scaling=1.0
    ):
    """
    Add parallel adapters with external embeddings to a GPT-2 model.
    """
    # Get the GPT-2 configuration
    hidden_size = model.config.hidden_size
    
    # Determine which component contains the transformer blocks
    if hasattr(model, "transformer"):
        gpt2 = model.transformer
    else:
        gpt2 = model
    
    # Add adapters to each transformer block
    for block in gpt2.h:
        # Add adapters
        block.attn_adapter = ParallelAdapter(
            hidden_size,
            adapter_type=adapter_type,
            rank=rank,
            scaling=scaling
        )
        
        block.mlp_adapter = ParallelAdapter(
            hidden_size,
            adapter_type=adapter_type,
            rank=rank,
            scaling=scaling
        )
    
    # Override the model's forward method to compute adapter embeddings
    if not hasattr(model, "original_forward"):
        model.original_forward = model.forward
        
        def new_gpt2_forward(self, input_ids=None, fourier_embeddings=None, past_key_values=None, attention_mask=None, 
                           token_type_ids=None, position_ids=None, head_mask=None, 
                           inputs_embeds=None, use_cache=None, output_attentions=None, 
                           output_hidden_states=None, return_dict=None, labels=None, **kwargs):

            # Check if fourier_embeddings exists and is a tensor
            if fourier_embeddings is not None and isinstance(fourier_embeddings, torch.Tensor):
                has_fourier_embeddings = (fourier_embeddings != 0).any()
                if has_fourier_embeddings:
                    self._adapter_embeddings = fourier_embeddings
                else:
                    self._adapter_embeddings = None
            else:
                self._adapter_embeddings = None
            
            # Call original forward method
            return self.original_forward(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
                **kwargs
            )
        
        model.forward = types.MethodType(new_gpt2_forward, model)
    
    # Override each transformer block's forward method
    for block in gpt2.h:
        if not hasattr(block, "original_forward"):
            block.original_forward = block.forward
            
            def new_block_forward(self, hidden_states, layer_past=None, attention_mask=None,
                                head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None,
                                use_cache=False, output_attentions=False):
                residual = hidden_states
                hidden_states = self.ln_1(hidden_states)
                
                # Original attention calculation
                attn_outputs = self.attn(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions
                )
                attn_output = attn_outputs[0]
                
                # Parallel adapter for attention using external embeddings
                if hasattr(self, '_model_ref') and self._model_ref() is not None:
                    model = self._model_ref()
                    if hasattr(model, "_adapter_embeddings") and model._adapter_embeddings is not None:
                        if hasattr(self, "attn_adapter"):
                            # Ensure embeddings are on the same device as the adapter
                            adapter_embeddings = model._adapter_embeddings.to(self.attn_adapter.adapter.weight.device if hasattr(self.attn_adapter, "adapter") else self.attn_adapter.down_proj.weight.device)
                            attn_adapter_output = self.attn_adapter(adapter_embeddings)
                            attn_output = attn_output + attn_adapter_output
                
                # Add residual connection
                hidden_states = attn_output + residual
                
                # MLP part
                residual = hidden_states
                hidden_states = self.ln_2(hidden_states)
                
                # Original MLP calculation
                mlp_output = self.mlp(hidden_states)
                
                # Parallel adapter for MLP using external embeddings
                if hasattr(self, '_model_ref') and self._model_ref() is not None:
                    model = self._model_ref()
                    if hasattr(model, "_adapter_embeddings") and model._adapter_embeddings is not None:
                        if hasattr(self, "mlp_adapter"):
                            # Ensure embeddings are on the same device as the adapter
                            adapter_embeddings = model._adapter_embeddings.to(self.mlp_adapter.adapter.weight.device if hasattr(self.mlp_adapter, "adapter") else self.mlp_adapter.down_proj.weight.device)
                            mlp_adapter_output = self.mlp_adapter(adapter_embeddings)
                            mlp_output = mlp_output + mlp_adapter_output
                
                # Add residual connection
                hidden_states = mlp_output + residual
                
                # Create outputs tuple
                outputs = (hidden_states,)
                if use_cache:
                    outputs = outputs + (attn_outputs[1],)
                if output_attentions:
                    outputs = outputs + (attn_outputs[2],)
                    
                return outputs
            
            # Create closure for binding
            def make_forward(block):
                return types.MethodType(new_block_forward, block)
            
            block.forward = make_forward(block)
            block._model_ref = weakref.ref(model)  # Use weak reference instead of direct reference
    
    # Freeze all parameters except adapters and adapter embedding
    for name, param in model.named_parameters():
        if 'adapter' not in name and 'adapter_embedding' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    return model


def add_parallel_adapters_to_llama(
        model, 
        adapter_type="linear", 
        rank=8, 
        scaling=1.0
    ):
    """
    Add parallel adapters to LLaMA model, using external embeddings.
    
    Args:
        model: The pre-trained LLaMA model
        adapter_type: Type of adapter ("linear", "affine", or "low_rank")
        rank: Rank size for low-rank adapters
        scaling: Scaling factor for adapter outputs
    """
    # Get model configuration
    hidden_size = model.config.hidden_size
    
    # Get LLaMA's transformer blocks
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        transformer_blocks = model.model.layers
    else:
        raise ValueError("Could not find LLaMA layers in the model structure.")
    
    # Add adapters to each transformer block
    for block in transformer_blocks:
        # Add attention adapter
        if hasattr(block, "self_attn"):
            block.attn_adapter = ParallelAdapter(
                hidden_size,
                adapter_type=adapter_type,
                rank=rank,
                scaling=scaling
            )
        
        # Add MLP adapter
        if hasattr(block, "mlp"):
            block.mlp_adapter = ParallelAdapter(
                hidden_size,
                adapter_type=adapter_type,
                rank=rank,
                scaling=scaling
            )
    
    # Modify the model's forward method to handle Fourier embeddings
    if not hasattr(model, "original_forward"):
        model.original_forward = model.forward
        
        def new_llama_forward(self, input_ids=None, attention_mask=None, position_ids=None, 
                           past_key_values=None, inputs_embeds=None, use_cache=None,
                           output_attentions=None, output_hidden_states=None, return_dict=None,
                           fourier_embeddings=None, labels=None, **kwargs):
            
            # Check if fourier_embeddings exists and is a tensor
            if fourier_embeddings is not None and isinstance(fourier_embeddings, torch.Tensor):
                has_fourier_embeddings = (fourier_embeddings != 0).any()
                if has_fourier_embeddings:
                    self._adapter_embeddings = fourier_embeddings
                else:
                    self._adapter_embeddings = None
            else:
                self._adapter_embeddings = None
            
            # Call the original forward pass
            outputs = self.original_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
                **kwargs
            )
            
            return outputs
        
        model.forward = types.MethodType(new_llama_forward, model)
    
    # Modify each transformer block's forward method
    for block in transformer_blocks:
        if not hasattr(block, "original_forward"):
            block.original_forward = block.forward
            
            def new_block_forward(self, hidden_states, attention_mask=None, position_ids=None,
                                past_key_value=None, output_attentions=False, use_cache=False,
                                **kwargs):
                
                # Call original forward method
                outputs = self.original_forward(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    **kwargs
                )
                
                # Extract hidden states from outputs
                if isinstance(outputs, tuple):
                    out = outputs[0]
                else:
                    out = outputs
                
                # Apply adapters using external embeddings
                if hasattr(self, '_model_ref') and self._model_ref() is not None:
                    model = self._model_ref()
                    if hasattr(model, "_adapter_embeddings") and model._adapter_embeddings is not None:
                        if hasattr(self, "attn_adapter"):
                            # Ensure embeddings are on the same device as the adapter
                            adapter_embeddings = model._adapter_embeddings.to(self.attn_adapter.adapter.weight.device if hasattr(self.attn_adapter, "adapter") else self.attn_adapter.down_proj.weight.device)
                            attn_adapter_output = self.attn_adapter(adapter_embeddings)
                            out = out + attn_adapter_output
                        
                        if hasattr(self, "mlp_adapter"):
                            # Ensure embeddings are on the same device as the adapter
                            adapter_embeddings = model._adapter_embeddings.to(self.mlp_adapter.adapter.weight.device if hasattr(self.mlp_adapter, "adapter") else self.mlp_adapter.down_proj.weight.device)
                            mlp_adapter_output = self.mlp_adapter(adapter_embeddings)
                            out = out + mlp_adapter_output
                
                # Update outputs
                if isinstance(outputs, tuple):
                    outputs = (out,) + outputs[1:]
                else:
                    outputs = out
                
                return outputs
            
            # Create a closure to preserve the block reference
            def make_forward(block):
                return types.MethodType(new_block_forward, block)
            
            block.forward = make_forward(block)
            block._model_ref = weakref.ref(model)  # Use weak reference instead of direct reference
    
    # Freeze all parameters except adapters
    for name, param in model.named_parameters():
        if 'adapter' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    return model


def add_parallel_adapters_to_bert(
        model, 
        adapter_type="linear", 
        rank=8, 
        scaling=1.0
    ):
    """
    Add parallel adapters to BERT model, using external embeddings.
    
    Args:
        model: The pre-trained BERT model
        adapter_type: Type of adapter ("linear", "affine", or "low_rank")
        rank: Rank size for low-rank adapters
        scaling: Scaling factor for adapter outputs
    """
    # Get model configuration
    hidden_size = model.config.hidden_size
    
    # Get BERT's transformer blocks
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        transformer_blocks = model.encoder.layer
    else:
        raise ValueError("Could not find BERT layers in the model structure.")
    
    # Add adapters to each transformer block
    for block in transformer_blocks:
        # Add attention adapter
        if hasattr(block, "attention"):
            block.attn_adapter = ParallelAdapter(
                hidden_size,
                adapter_type=adapter_type,
                rank=rank,
                scaling=scaling
            )
        
        # Add MLP adapter
        if hasattr(block, "intermediate") or hasattr(block, "output"):
            block.mlp_adapter = ParallelAdapter(
                hidden_size,
                adapter_type=adapter_type,
                rank=rank,
                scaling=scaling
            )
    
    # Modify the model's forward method to handle Fourier embeddings
    if not hasattr(model, "original_forward"):
        model.original_forward = model.forward
        
        def new_bert_forward(self, input_ids=None, attention_mask=None, token_type_ids=None, 
                          position_ids=None, head_mask=None, inputs_embeds=None, 
                          encoder_hidden_states=None, encoder_attention_mask=None,
                          output_attentions=None, output_hidden_states=None, 
                          return_dict=None, fourier_embeddings=None, labels=None, **kwargs):
            
            # Check if fourier_embeddings exists and is a tensor
            if fourier_embeddings is not None and isinstance(fourier_embeddings, torch.Tensor):
                has_fourier_embeddings = (fourier_embeddings != 0).any()
                if has_fourier_embeddings:
                    self._adapter_embeddings = fourier_embeddings
                else:
                    self._adapter_embeddings = None
            else:
                self._adapter_embeddings = None
            
            # Call the original forward pass
            outputs = self.original_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
                **kwargs
            )
            
            return outputs
        
        model.forward = types.MethodType(new_bert_forward, model)
    
    # Modify each transformer block's forward method
    for block in transformer_blocks:
        if not hasattr(block, "original_forward"):
            block.original_forward = block.forward
            
            def new_block_forward(self, hidden_states, attention_mask=None, head_mask=None,
                                encoder_hidden_states=None, encoder_attention_mask=None,
                                output_attentions=False, **kwargs):
                
                # Call original forward method
                outputs = self.original_forward(
                    hidden_states,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    output_attentions=output_attentions,
                    **kwargs
                )
                
                # Extract hidden states from outputs
                if isinstance(outputs, tuple):
                    out = outputs[0]
                else:
                    out = outputs
                
                # Apply adapters using external embeddings
                if hasattr(self, '_model_ref') and self._model_ref() is not None:
                    model = self._model_ref()
                    if hasattr(model, "_adapter_embeddings") and model._adapter_embeddings is not None:
                        if hasattr(self, "attn_adapter"):
                            # Ensure embeddings are on the same device as the adapter
                            adapter_embeddings = model._adapter_embeddings.to(self.attn_adapter.adapter.weight.device if hasattr(self.attn_adapter, "adapter") else self.attn_adapter.down_proj.weight.device)
                            attn_adapter_output = self.attn_adapter(adapter_embeddings)
                            out = out + attn_adapter_output
                        
                        if hasattr(self, "mlp_adapter"):
                            # Ensure embeddings are on the same device as the adapter
                            adapter_embeddings = model._adapter_embeddings.to(self.mlp_adapter.adapter.weight.device if hasattr(self.mlp_adapter, "adapter") else self.mlp_adapter.down_proj.weight.device)
                            mlp_adapter_output = self.mlp_adapter(adapter_embeddings)
                            out = out + mlp_adapter_output
                
                # Update outputs
                if isinstance(outputs, tuple):
                    outputs = (out,) + outputs[1:]
                else:
                    outputs = out
                
                return outputs
            
            # Create a closure to preserve the block reference
            def make_forward(block):
                return types.MethodType(new_block_forward, block)
            
            block.forward = make_forward(block)
            block._model_ref = weakref.ref(model)  # Use weak reference instead of direct reference
    
    # Freeze all parameters except adapters
    for name, param in model.named_parameters():
        if 'adapter' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    return model


def load_model_with_parallel_adapters(
        model, 
        model_name, 
        adapter_type="linear", 
        rank=8, 
        scaling=1.0
    ):
    """
    Load a model with parallel adapters.
    """
    if model_name == "gpt2":
        model = add_parallel_adapters_to_gpt2(model, adapter_type=adapter_type, rank=rank, scaling=scaling)
    elif model_name == "llama":
        model = add_parallel_adapters_to_llama(model, adapter_type=adapter_type, rank=rank, scaling=scaling)
    elif model_name == "bert":
        model = add_parallel_adapters_to_bert(model, adapter_type=adapter_type, rank=rank, scaling=scaling)
    else:
        raise ValueError(f"Model {model_name} not supported.")

    return model
    