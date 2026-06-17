import os
import re
import torch
import logging

def is_numeric(s):
    """
    Check if a string represents a valid numeric value.
    Accepts integers, decimals, and scientific notation.
    Allows for whitespace and handles negative numbers.
    """
    # First strip whitespace
    s = s.strip()
    
    # Try to convert to float - most robust method
    try:
        float(s)
        return True
    except ValueError:
        return False
        
    # Alternative regex approach (commented out)
    # return bool(re.match(r"^\s*-?\d+(\.\d+)?([eE][-+]?\d+)?\s*$", s))

def handle_nan_loss(batch, before_decoder, data_idx, model, save_path="debug_nan_loss.pt"):
    """
    Handles NaN loss by saving the problematic batch and model state before the decoder.
    """
    debug_data = {
        "batch": batch,
        "before_decoder": before_decoder.cpu() if before_decoder is not None else None,
        "data_idx": data_idx,
        "model_state": model.state_dict(),
    }
    save_path = os.path.join("fail_case_log", save_path)
    torch.save(debug_data, save_path)
    logging.info(f"Saved debug data to {save_path}. Stopping training due to NaN loss.")

def get_regular_embeddings(model, input_ids):
    """
    Returns the token embeddings based on the model type.
    """
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
        # For GPT-2 models
        return model.transformer.wte(input_ids)
    elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        # For LLaMA models
        return model.model.embed_tokens(input_ids)
    else:
        raise AttributeError(f"Cannot find token embeddings in the model: {type(model)}")

def count_trainable_parameters(model, number_encoder, intermediate_network=None, adapter_type=None):
    """
    Count the number of trainable parameters in the entire network.
    
    Parameters:
        model: The pretrained language model
        number_encoder: The FNE module for numeric embeddings
        intermediate_network: Network to process embeddings (MLP, Linear, or Identity)
        adapter_type: Type of adapter ('linear', 'affine', 'low_rank' or None)
        
    Returns:
        A dictionary with parameter counts for each component and the total
    """
    counts = {}
    
    # Count LLM parameters
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    counts['model'] = model_params
    
    # Count number encoder parameters
    encoder_params = sum(p.numel() for p in number_encoder.parameters() if p.requires_grad)
    counts['number_encoder'] = encoder_params
    
    # Count intermediate network parameters
    if intermediate_network is not None:
        intermediate_params = sum(p.numel() for p in intermediate_network.parameters() if p.requires_grad)
        counts['intermediate_network'] = intermediate_params
    else:
        counts['intermediate_network'] = 0
    
    # Count adapter parameters if applicable
    adapter_params = 0
    if adapter_type is not None and hasattr(model, 'get_adapter_parameters'):
        # If the model has a method to get adapter parameters
        adapter_params = sum(p.numel() for p in model.get_adapter_parameters() if p.requires_grad)
    elif adapter_type is not None:
        # Try to find adapter parameters by name
        for name, param in model.named_parameters():
            if 'adapter' in name.lower() and param.requires_grad:
                adapter_params += param.numel()
    counts['adapter'] = adapter_params
    
    # Calculate total
    counts['total'] = sum(count for count in counts.values())
    
    return counts

def log_parameter_counts(counts, logger=None):
    """
    Log the parameter counts in a readable format
    
    Parameters:
        counts: Dictionary with parameter counts
        logger: Logger object (if None, print to stdout)
    """
    total = counts['total']
    
    # Format the output
    output = ["Trainable parameter counts:"]
    for component, count in counts.items():
        if component != 'total':
            percentage = (count / total * 100) if total > 0 else 0
            output.append(f"  {component}: {count:,} ({percentage:.2f}%)")
    
    output.append(f"Total trainable parameters: {total:,}")
    
    # Log or print the output
    if logger:
        for line in output:
            logger.info(line)
    else:
        for line in output:
            print(line)
