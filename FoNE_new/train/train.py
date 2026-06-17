import torch
import logging
import sys
from train.utils import get_regular_embeddings, handle_nan_loss, count_trainable_parameters, log_parameter_counts

def train_fne(model, train_loader, number_encoder, intermediate_network, optimizer, scheduler, args, int_digit_len, frac_digit_len, len_gen_size, decoder_type, adapter_type, device, tokenizer=None):
    """
    Training loop for Fourier Neural Embedding (FNE) based models with intermediate network.
    LLM parameters are now trainable along with FNE and intermediate network.
    
    This function handles two types of decoder strategies:
    1. 'fne': Uses Fourier-based approach to directly predict numeric values
    2. 'greedy': Uses token-by-token autoregressive decoding like traditional language models
    
    Parameters:
        model: The pretrained language model
        train_loader: DataLoader for training data
        number_encoder: The FNE module for numeric embeddings
        intermediate_network: Network to process embeddings (MLP, Linear, or Identity)
        optimizer: The optimizer for updating model parameters
        scheduler: Learning rate scheduler
        args: Command line arguments and configuration
        int_digit_len: Length of integer part in numeric representation
        frac_digit_len: Length of fractional part in numeric representation
        len_gen_size: Parameter for FNE to add trailing zeros
        decoder_type: 'fne' or 'greedy' - determines how number predictions are made
        adapter_type: Type of adapter for modifying model behavior (None, 'linear', 'affine', 'low_rank')
        device: Computation device (CPU/GPU)
        tokenizer: Required when decoder_type is 'greedy' to convert numbers to tokens
    """
    # === Initial Setup for Greedy Decoder ===
    # If using greedy decoder, we need to adapt the FNE encoder to work with token-based approaches
    # if decoder_type == 'greedy':
        # logging.info("Using vanilla training method for greedy decoder")
        
        # # Store references to the original FNE methods that will be patched
        # orig_compute_loss = number_encoder.fourier_compute_loss
        # orig_compute_prediction = number_encoder.fourier_compute_prediction
        
        # # Define patched methods that adapt FNE interface to work with token-based approach
        # def patched_compute_loss(self, last_hidden_state, label):
        #     # Call original method with correct parameters
        #     return orig_compute_loss(last_hidden_state, label, int_digit_len, frac_digit_len)
            
        # def patched_compute_prediction(self, last_hidden_state):
        #     # Call original method with correct parameters
        #     return orig_compute_prediction(last_hidden_state, int_digit_len, frac_digit_len)
        
        # # Save original methods and attributes for later restoration
        # number_encoder._original_compute_loss = getattr(number_encoder, 'compute_loss', None)
        # number_encoder._original_compute_prediction = getattr(number_encoder, 'compute_prediction', None)
        # number_encoder._original_frac_digit_len = getattr(number_encoder, 'frac_digit_len', None)
        # number_encoder._original_int_digit_len = getattr(number_encoder, 'int_digit_len', None)
        
        # # Add required attributes to number_encoder for vanilla training
        # number_encoder.frac_digit_len = frac_digit_len
        # number_encoder.int_digit_len = int_digit_len
        
        # # Apply the patched methods to the number encoder
        # import types
        # number_encoder.compute_loss = types.MethodType(patched_compute_loss, number_encoder)
        # number_encoder.compute_prediction = types.MethodType(patched_compute_prediction, number_encoder)
        
        # Note: The commented code below was an attempt to delegate completely to train_regular
        # but we now handle greedy decoding directly in this function
        # try:
        #     # Run training with patched encoder
        #     # model, dataloader, optimizer, scheduler, device, args# 
        #     return train_regular(model, train_loader, optimizer, scheduler, device, args, number_encoder, intermediate_network)
        # finally:
        #     # Restore original methods
        #     if number_encoder._original_compute_loss is not None:
        #         number_encoder.compute_loss = number_encoder._original_compute_loss
        #     else:
        #         delattr(number_encoder, 'compute_loss')
        #   ...
    
    # Ensure the tokenizer is provided when using greedy decoder
    if decoder_type == 'greedy' and tokenizer is None:
        raise ValueError("Tokenizer is required when decoder_type is 'greedy'")
        
    # === Model & Component Preparation ===
    # Move all components to the appropriate device
    model = model.to(device)
    number_encoder = number_encoder.to(device)
    if intermediate_network is not None:
        intermediate_network = intermediate_network.to(device)
    
    # Set training mode for all components based on configuration
    if not args.freeze_model:
        # Allow gradient updates for the core LLM if not frozen
        for param in model.parameters():
            param.requires_grad = True
        # Set training mode without triggering recursion
        if hasattr(model, 'training'):
            object.__setattr__(model, 'training', True)
    else:
        # Freeze LLM parameters when specified
        for param in model.parameters():
            param.requires_grad = False
        # Set eval mode without triggering recursion
        if hasattr(model, 'training'):
            object.__setattr__(model, 'training', False)
    
    # Always train the number encoder and intermediate network
    number_encoder.train()
    if intermediate_network is not None:
        intermediate_network.train()
    
    # Count and log the trainable parameters
    param_counts = count_trainable_parameters(model, number_encoder, intermediate_network, adapter_type)
    logging.info("=== Model Parameter Statistics ===")
    log_parameter_counts(param_counts, logger=logging)
    logging.info("================================")
    
    total_loss = 0

    # === Main Training Loop ===
    for batch_idx, batch in enumerate(train_loader):
        try:
            # Get batch data and move to device
            input_ids = batch['input_ids'].to(device)
            scatter_tensor = batch['scatter_tensor'].to(device)  # Tensor indicating [NUM] token positions
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)  # Numeric labels as float tensors
            last_token_mask = batch['last_token_mask'].to(device)  # Mask to identify position of last token
            len_gen = torch.randint(0, len_gen_size+1, (1,), device=device).item()  # Random number for length generation
            
            # Get embeddings from the base model (with gradients enabled)
            regular_embeddings = get_regular_embeddings(model, input_ids)
            
            # Generate Fourier numeric embeddings and process them
            fourier_embeddings = number_encoder(scatter_tensor, len_gen=len_gen)
            if intermediate_network is not None:
                fourier_embeddings = intermediate_network(fourier_embeddings)
            
            # Combine regular token embeddings with numeric embeddings
            combined_embeddings = regular_embeddings + fourier_embeddings
            
            # Ensure all inputs match the model's expected data type
            combined_embeddings = combined_embeddings.to(device=device, dtype=model.dtype)
            attention_mask = attention_mask.to(device=device, dtype=model.dtype)
            fourier_embeddings = fourier_embeddings.to(device=device, dtype=model.dtype)
            
            
            # === FNE Decoder Path (Direct numeric prediction) ===
            if decoder_type == 'fne':
                # Forward pass depends on adapter configuration
                if adapter_type is None:
                    # Standard forward pass without adapters
                    outputs = model(inputs_embeds=combined_embeddings, attention_mask=attention_mask, output_hidden_states=True)
                elif adapter_type == 'linear' or adapter_type == 'affine' or adapter_type == 'low_rank':
                    # Forward pass with adapter modules (needs fourier_embeddings)
                    outputs = model(inputs_embeds=combined_embeddings, attention_mask=attention_mask, output_hidden_states=True, fourier_embeddings=fourier_embeddings)
                else:
                    raise ValueError(f"Unsupported adapter type '{adapter_type}'.")
                
                # Extract the hidden state for the last token position
                before_decoder = outputs.hidden_states[-1]  # Get the final layer hidden states
                last_token_hidden_state = (before_decoder * last_token_mask.unsqueeze(-1)).sum(dim=1)  # Apply mask to get last token
                
                # Compute loss using the FNE-specific loss function
                loss = number_encoder.fourier_compute_loss(last_token_hidden_state, labels, int_digit_len, frac_digit_len, len_gen=len_gen)
                
            # === Greedy Decoder Path (Token-by-token prediction) ===
            elif decoder_type == 'greedy':
                # Convert numeric labels to token sequences
                tokenized_labels = []
                
                # Convert numeric labels to target token sequences
                for label in labels:    # labels is a tensor of shape (batch_size,)
                    # Convert float to int then to string and tokenize
                    label_str = str(int(label.item()))

                    # Tokenize the label
                    tokens = tokenizer(label_str, return_tensors="pt").input_ids.to(device)

                    tokens = tokens[:, 2:]  # Remove BOS and space token

                    # # Verify the tokenization
                    # dec_label = tokenizer.decode(tokens.squeeze(0))
                    # print(f"dec_label: {dec_label}")
                    # print(f"orig_label: {int(label.item())}")
                    # print(f"{int(float(dec_label)) == int(label.item())}")
                    
                    tokenized_labels.append(tokens.squeeze(0))
                
                # Create a forward pass
                if adapter_type is None:
                    outputs = model(
                        inputs_embeds=combined_embeddings,
                        attention_mask=attention_mask,
                        output_hidden_states=True
                    )
                elif adapter_type in ['linear', 'affine', 'low_rank']:
                    outputs = model(
                        inputs_embeds=combined_embeddings,
                        attention_mask=attention_mask,
                        fourier_embeddings=fourier_embeddings,
                        output_hidden_states=True
                    )
                else:
                    raise ValueError(f"Unsupported adapter type '{adapter_type}'.")
                
                # Get the logits from the output
                logits = outputs.logits
                
                # Find the position of the last token in each sequence
                last_positions = last_token_mask.nonzero()[:, 1]
                
                # Calculate loss
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                batch_loss = torch.tensor(0.0, device=device)
                
                # For each sample in the batch
                for i, target_tokens in enumerate(tokenized_labels):
                    # Get predictions starting from the position of interest
                    sequence_logits = logits[i]  # Shape: [1, vocab_size]

                    ## Debugging
                    # print(f"target_tokens.shape: {target_tokens.shape}")
                    # print(f"sequence_logits.shape: {sequence_logits.shape}")
                    # dec_pred = tokenizer.decode(sequence_logits.argmax(dim=-1))
                    # print(f"dec_pred: {dec_pred}")
                    # print(f"sequence_logits.argmax: {sequence_logits.argmax(dim=-1)}")
                    # print(f"target_tokens: {target_tokens}")

                    if sequence_logits.shape[0] > target_tokens.shape[0]:
                        # Pad the target tokens by appending EOS token id to match the sequence logits shape
                        target_tokens = torch.cat([target_tokens, torch.tensor([tokenizer.eos_token_id] * (sequence_logits.shape[0] - target_tokens.shape[0]), device=device)])
                        # print(f"after padding target_tokens.shape: {target_tokens.shape}")
                    elif sequence_logits.shape[0] < target_tokens.shape[0]:
                        # pad_token = tokenizer(" ", return_tensors="pt").input_ids.to(device)
                        # one-hot vectorize pad_token
                        pad_tensor = torch.zeros(1, len(tokenizer), device=device)
                        pad_tensor[0, int(tokenizer.eos_token_id)] = 1
                        # Pad the sequence logits by appending tensor of length identical to vocab size (dim 1), 
                        # which is the one-hot vector that corresponds to EOS token to match the target tokens shape
                        pad_tensor = pad_tensor.expand(target_tokens.shape[0] - sequence_logits.shape[0], -1)
                        
                        sequence_logits = torch.cat([sequence_logits, pad_tensor])
                        # print(f"after padding sequence_logits.shape: {sequence_logits.shape}")

                    # If we have target tokens, calculate loss
                    if target_tokens.size(0) > 0:
                        # Standard L2R decoding  
                        sample_loss = loss_fct(sequence_logits, target_tokens)
                        batch_loss += sample_loss

                        # # Variant R2L decoding  
                        # target_tokens = target_tokens.flip(0)
                        # sample_loss = loss_fct(sequence_logits, target_tokens)
                        # batch_loss += sample_loss
                    else:
                        raise ValueError("No target tokens found")
                
                # Average the loss over the batch
                loss = batch_loss / input_ids.size(0)
                
                # Handle zero loss edge case
                if loss.item() == 0:
                    # Fallback to a small non-zero loss to prevent training issues
                    raise ValueError("Zero loss detected")
            else:
                raise ValueError(f"Unsupported decoder type '{decoder_type}'.")

            # === Backpropagation and Optimization ===
            # Compute gradients through the entire computation graph
            loss.backward()
            
            # Apply gradient clipping if enabled
            if args.clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(number_encoder.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(intermediate_network.parameters(), max_norm=1.0)

            # Update model parameters and learning rate
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()  # Reset gradients for next iteration

            # Accumulate total loss for reporting
            total_loss += loss.item()
            
        except Exception as e:
            # Catch and log any exceptions that occur during training
            logging.error(f"Error processing batch {batch_idx}: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            continue  # Skip this batch and continue with next

    # Report average loss across all batches
    avg_loss = total_loss / len(train_loader)
    logging.info(f"avg Loss: {avg_loss}")
    return avg_loss

def train_regular(model, dataloader, optimizer, scheduler, device, args, number_encoder=None, intermediate_network=None):
    """
    Regular training loop for models without additional embedding modules.
    """
    model.train()
    
    # Count trainable parameters
    param_counts = count_trainable_parameters(model, number_encoder, intermediate_network)
    logging.info("=== Model Parameter Statistics ===")
    log_parameter_counts(param_counts, logger=logging)
    logging.info("================================")
    
    total_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()

        if args.clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    logging.info(f"avg Loss: {total_loss / len(dataloader)}")
    return total_loss / len(dataloader)

def train_xval(model, train_loader, xval, optimizer, scheduler, args, device):
    """
    Training loop for models using the xval module.
    """
    model.train()
    xval.train()
    
    # Count trainable parameters (treating xval as the "number_encoder")
    param_counts = count_trainable_parameters(model, xval)
    logging.info("=== Model Parameter Statistics ===")
    log_parameter_counts(param_counts, logger=logging)
    logging.info("================================")
    
    total_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        scatter_tensor = batch['scatter_tensor'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        last_token_mask = batch['last_token_mask'].to(device)

        regular_embeddings = get_regular_embeddings(model, input_ids)
        input_embeddings = xval(scatter_tensor, regular_embeddings)
        outputs = model(inputs_embeds=input_embeddings, attention_mask=attention_mask, output_hidden_states=True)
        before_decoder = outputs.hidden_states[-1]
        last_token_hidden_state = (before_decoder * last_token_mask.unsqueeze(-1)).sum(dim=1)

        loss = xval.compute_loss(last_token_hidden_state, labels)

        loss.backward()
        if args.clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(xval.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    logging.info(f"avg Loss: {total_loss / len(train_loader)}")
    return total_loss / len(train_loader)

def train_vanilla(model, train_loader, vanilla_model, intermediate_network, optimizer, scheduler, args, device):
    """
    Training loop for models using a vanilla embedding module with intermediate network.
    """
    model.train()
    vanilla_model.train()
    intermediate_network.train()
    
    # Count trainable parameters (treating vanilla_model as the "number_encoder")
    param_counts = count_trainable_parameters(model, vanilla_model, intermediate_network)
    logging.info("=== Model Parameter Statistics ===")
    log_parameter_counts(param_counts, logger=logging)
    logging.info("================================")
    
    total_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        scatter_tensor = batch['scatter_tensor'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        last_token_mask = batch['last_token_mask'].to(device)
        
        regular_embeddings = get_regular_embeddings(model, input_ids)
        vanilla_embeddings = vanilla_model(scatter_tensor)
        
        # Apply intermediate network to the combined embeddings
        combined_embeddings = regular_embeddings + vanilla_embeddings
        processed_embeddings = intermediate_network(combined_embeddings)

        outputs = model(inputs_embeds=processed_embeddings, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        last_token_hidden_state = (last_hidden_state * last_token_mask.unsqueeze(-1)).sum(dim=1)

        loss = vanilla_model.compute_loss(last_token_hidden_state, labels)
        
        loss.backward()
        if args.clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(vanilla_model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(intermediate_network.parameters(), max_norm=1.0)
            
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    logging.info(f"Training Loss: {avg_loss}")
    return avg_loss
