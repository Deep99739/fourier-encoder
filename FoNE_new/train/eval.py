import torch
import logging
from .utils import is_numeric, get_regular_embeddings

def evaluate_fne(model, test_loader, number_encoder, intermediate_network, int_digit_len, frac_digit_len, device, print_labels=False, max_print=10, decoder_type=None, adapter_type=None, tokenizer=None):
    """
    Evaluation loop for Fourier Neural Embedding (FNE) based models.
    
    Parameters:
        decoder_type: 'fourier' (default) or 'greedy'
        tokenizer: Required when decoder_type is 'greedy'
    """
    # # If using greedy decoder, delegate to evaluate_vanilla
    # if decoder_type == 'greedy':
    #     logging.info("Using vanilla evaluation method for greedy decoder")
        
    #     # Create adapter methods for the FNE interface
    #     orig_compute_loss = number_encoder.fourier_compute_loss
    #     orig_compute_prediction = number_encoder.fourier_compute_prediction
        
    #     # Monkey patch the methods temporarily
    #     def patched_compute_loss(self, last_hidden_state, label):
    #         return orig_compute_loss(last_hidden_state, label, int_digit_len, frac_digit_len)
            
    #     def patched_compute_prediction(self, last_hidden_state):
    #         return orig_compute_prediction(last_hidden_state, int_digit_len, frac_digit_len)
        
    #     # Save original methods and attributes to restore later
    #     number_encoder._original_compute_loss = getattr(number_encoder, 'compute_loss', None)
    #     number_encoder._original_compute_prediction = getattr(number_encoder, 'compute_prediction', None)
    #     number_encoder._original_frac_digit_len = getattr(number_encoder, 'frac_digit_len', None)
    #     number_encoder._original_int_digit_len = getattr(number_encoder, 'int_digit_len', None)
        
    #     # Add required attributes for vanilla evaluation
    #     number_encoder.frac_digit_len = frac_digit_len  # Add this attribute directly
    #     number_encoder.int_digit_len = int_digit_len    # Also add int_digit_len for completeness
        
    #     # Add the patched methods
    #     import types
    #     number_encoder.compute_loss = types.MethodType(patched_compute_loss, number_encoder)
    #     number_encoder.compute_prediction = types.MethodType(patched_compute_prediction, number_encoder)
        
        # try:
        #     # Run evaluation with patched encoder
        #     # model, dataloader, tokenizer, device, print_labels=False, max_print_examples=10
        #     return evaluate_regular(model, test_loader, tokenizer, device, print_labels, max_print, number_encoder, intermediate_network)
        # finally:
        #     # Restore original methods
        #     if number_encoder._original_compute_loss is not None:
        #         number_encoder.compute_loss = number_encoder._original_compute_loss
        #     else:
        #         delattr(number_encoder, 'compute_loss')
                
        #     if number_encoder._original_compute_prediction is not None:
        #         number_encoder.compute_prediction = number_encoder._original_compute_prediction
        #     else:
        #         delattr(number_encoder, 'compute_prediction')
                
        #     # Restore original attributes (or remove them if they didn't exist)
        #     if number_encoder._original_frac_digit_len is not None:
        #         number_encoder.frac_digit_len = number_encoder._original_frac_digit_len
        #     else:
        #         delattr(number_encoder, 'frac_digit_len')
                
        #     if number_encoder._original_int_digit_len is not None:
        #         number_encoder.int_digit_len = number_encoder._original_int_digit_len
        #     else:
        #         delattr(number_encoder, 'int_digit_len')
                
        #     # Clean up temporary attributes
        #     delattr(number_encoder, '_original_compute_loss')
        #     delattr(number_encoder, '_original_compute_prediction')
        #     delattr(number_encoder, '_original_frac_digit_len')
        #     delattr(number_encoder, '_original_int_digit_len')
    
    logging.info('Evaluation start')
    model.eval()
    number_encoder.eval()
    intermediate_network.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0
    total_squared_error = 0
    total_digits = 0
    correct_digits = 0
    all_labels = []
    all_predictions = []
    mispredictions = []
    printed_examples = 0  # Initialize counter for printed examples

    if decoder_type == 'greedy' and tokenizer is None:
        raise ValueError("Tokenizer is required when decoder_type is 'greedy'")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(device)
            scatter_tensor = batch['scatter_tensor'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            last_token_mask = batch['last_token_mask'].to(device)

            regular_embeddings = get_regular_embeddings(model, input_ids)
            fourier_embeddings = number_encoder(scatter_tensor)
            fourier_embeddings = intermediate_network(fourier_embeddings)
            input_embeddings = regular_embeddings + fourier_embeddings

            # modified part by EJ: match dtype of inputs same as model dtype
            input_embeddings = input_embeddings.to(model.dtype)
            attention_mask = attention_mask.to(model.dtype)     

            

            if decoder_type == 'fne':
                if adapter_type is None:
                    outputs = model(inputs_embeds=input_embeddings, attention_mask=attention_mask, output_hidden_states=True)
                elif adapter_type == 'linear' or adapter_type == 'affine' or adapter_type == 'low_rank':
                    outputs = model(inputs_embeds=input_embeddings, attention_mask=attention_mask, output_hidden_states=True, fourier_embeddings=fourier_embeddings)
                else:
                    raise ValueError(f"Unsupported adapter type '{adapter_type}'.")
                
                before_decoder = outputs.hidden_states[-1]
                last_token_hidden_state = (before_decoder * last_token_mask.unsqueeze(-1)).sum(dim=1)
                
                predicted_numbers = number_encoder.fourier_compute_prediction(last_token_hidden_state, int_digit_len, frac_digit_len)
                
                all_labels.append(labels.cpu())
                all_predictions.append(predicted_numbers.cpu())
                
                # tolerance = 10 ** (-frac_digit_len)
                # correct_predictions = torch.abs(predicted_numbers - labels) < tolerance
                correct_predictions = torch.abs(predicted_numbers - labels) == 0
                total_correct += correct_predictions.sum().item()
                total_samples += labels.size(0)

                for i in range(labels.size(0)):
                    actual_value = str(labels[i].item())
                    predicted_value = str(predicted_numbers[i].item())
                    min_len = len(actual_value)
                    correct_digits += sum(1 for a, p in zip(actual_value[:min_len], predicted_value[:min_len]) if a == p)
                    total_digits += len(actual_value)

                for i in range(labels.size(0)):
                    if not correct_predictions[i]:
                        mispredictions.append((predicted_numbers[i].item(), labels[i].item()))

                squared_error = torch.sum((predicted_numbers - labels) ** 2).item()
                total_squared_error += squared_error

                loss = number_encoder.fourier_compute_loss(last_token_hidden_state, labels, int_digit_len, frac_digit_len)
                total_loss += loss.item()

            elif decoder_type == 'greedy':
                try:
                    # Forward pass through model
                    if adapter_type is None:
                        outputs = model(inputs_embeds=input_embeddings, attention_mask=attention_mask, output_hidden_states=True)
                    elif adapter_type in ['linear', 'affine', 'low_rank']:
                        outputs = model(inputs_embeds=input_embeddings, attention_mask=attention_mask, output_hidden_states=True, fourier_embeddings=fourier_embeddings)
                    else:
                        raise ValueError(f"Unsupported adapter type '{adapter_type}'.")
                    
                    logits = outputs.logits
                    all_labels.append(labels.cpu())
                    
                    # Tokenize labels for evaluation - using same approach as in train.py
                    tokenized_labels = []
                    for label in labels:
                        # Convert float to int then to string and tokenize (matching train.py)
                        label_str = str(int(label.item()))
                        tokens = tokenizer(label_str, return_tensors="pt").input_ids.to(device)
                        tokens = tokens[:, 2:]  # Remove BOS and space token
                        tokenized_labels.append(tokens.squeeze(0))
                    
                    # Calculate loss
                    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                    batch_loss = torch.tensor(0.0, device=device)
                    batch_predicted_numbers = []
                    
                    # For each sample in the batch (matching train.py approach)
                    for i, target_tokens in enumerate(tokenized_labels):
                        sequence_logits = logits[i]
                        
                        # Handle sequence length mismatches
                        if sequence_logits.shape[0] > target_tokens.shape[0]:
                            # Pad the target tokens with EOS token
                            target_tokens = torch.cat([
                                target_tokens, 
                                torch.tensor([tokenizer.eos_token_id] * (sequence_logits.shape[0] - target_tokens.shape[0]), device=device)
                            ])
                        elif sequence_logits.shape[0] < target_tokens.shape[0]:
                            # Pad the sequence logits with EOS one-hot vectors
                            pad_tensor = torch.zeros(1, len(tokenizer), device=device)
                            pad_tensor[0, int(tokenizer.eos_token_id)] = 1
                            sequence_logits = torch.cat([sequence_logits, pad_tensor])
                        
                        # Calculate loss if we have target tokens
                        
                        if target_tokens.size(0) > 0:
                            # Standard L2R decoding  
                            sample_loss = loss_fct(sequence_logits, target_tokens)
                            batch_loss += sample_loss
                        
                            # # Variant R2L decoding  
                            # target_tokens = target_tokens.flip(0)
                            # sample_loss = loss_fct(sequence_logits, target_tokens)
                            # batch_loss += sample_loss

                        # Get predictions for evaluation metrics
                        pred_tokens_id = sequence_logits.argmax(dim=-1)[:target_tokens.size(0)]

                        # # If using R2L decoding, add this line
                        # pred_tokens_id = pred_tokens_id.flip(0)

                        pred_str = tokenizer.decode(pred_tokens_id, skip_special_tokens=True).strip()

                        # Convert prediction to number for metrics
                        try:
                            pred_num = int(float(pred_str)) if is_numeric(pred_str) else float('nan')

                            batch_predicted_numbers.append(pred_num)
                            
                            # Calculate accuracy metrics
                            actual_value = str(int(labels[i].item()))
                            predicted_value = str(pred_num)
                            
                            # Compare digits for accuracy
                            max_len = max(len(actual_value), len(predicted_value))
                            for i in range(max_len):
                                actual_digit = actual_value[i] if i < len(actual_value) else ' '
                                predicted_digit = predicted_value[i] if i < len(predicted_value) else ' '
                                if actual_digit == predicted_digit:
                                    correct_digits += 1
                            total_digits += max_len
                            
                            # Track whole number accuracy
                            if abs(pred_num - labels[i].item()) == 0:
                                total_correct += 1
                            else:
                                mispredictions.append((pred_num, labels[i].item()))
                            
                            total_samples += 1
                            
                            # Add squared error
                            squared_error = (pred_num - labels[i].item()) ** 2
                            total_squared_error += squared_error
                            
                        except ValueError:
                            # If conversion to float fails, count as error
                            batch_predicted_numbers.append(float('nan'))
                            total_samples += 1
                            mispredictions.append((float('nan'), labels[i].item()))
                    
                    # Average the batch loss
                    loss = batch_loss / input_ids.size(0)
                    if loss.item() == 0:
                        logging.warning("Zero loss detected during evaluation")
                    
                    total_loss += loss.item()
                    
                    # Convert predictions to tensor and add to all_predictions
                    if batch_predicted_numbers:
                        predicted_numbers = torch.tensor(batch_predicted_numbers, device=device)
                        all_predictions.append(predicted_numbers.cpu())
                    
                    # Print examples if requested
                    if print_labels and printed_examples < max_print:
                        for i in range(min(len(batch_predicted_numbers), max_print - printed_examples)):
                            if printed_examples >= max_print:
                                break
                            
                            logging.info(f"Input: {tokenizer.decode(input_ids[i])}")
                            logging.info(f"True: {labels[i].item()}, Predicted: {batch_predicted_numbers[i]}")
                            logging.info("---")
                            printed_examples += 1
                
                except Exception as e:
                    logging.error(f"Error in greedy evaluation: {str(e)}")
                    raise e
                
                total_loss += loss.item()
                
                # # Remove redundant second loop - use the metrics we already calculated above
                # if print_labels:
                #     examplelist = []
                #     for i in range(min(batch_size, max_print - printed_examples)):
                #         if printed_examples >= max_print:
                #             break
                        
                #         if i < len(batch_predicted_numbers):
                #             try:
                #                 predicted_val = batch_predicted_numbers[i]
                #                 actual_val = labels[i].item()
                #                 examplelist.append(f"({predicted_val}, {actual_val})")
                #                 printed_examples += 1
                #             except:
                #                 continue
                    
                #     if examplelist:
                #         logging.info(" ".join(examplelist))


    # Calculate metrics
    # For fourier decoder, all_labels is a list of tensors
    all_labels = torch.cat(all_labels)
    mean_label = all_labels.mean().item()
    total_variance = torch.sum((all_labels - mean_label) ** 2).item()

    # Calculate final metrics
    avg_loss = total_loss / len(test_loader)
    whole_number_accuracy = total_correct / total_samples
    digit_wise_accuracy = correct_digits / total_digits if total_digits > 0 else 0
    mse = total_squared_error / total_samples
    r2 = 1 - (total_squared_error / total_variance) if total_variance > 0 else float('nan')

    if print_labels:
        if mispredictions:
            log_count = min(len(mispredictions), max_print)
            logging.info(f"Mispredictions (up to {log_count} examples):")
            for i in range(log_count):
                predicted_val, actual_val = mispredictions[i]
                logging.info(f"Predicted: {predicted_val}, Actual: {actual_val}")
        else:
            logging.info("No mispredictions found!")

    return avg_loss, (whole_number_accuracy, digit_wise_accuracy), mse, r2

def evaluate_regular(model, dataloader, tokenizer, device, print_labels=False, max_print_examples=10, number_encoder=None, intermediate_network=None):
    """
    Evaluation loop for regular models.
    """
    logging.info('Evaluation start')
    model.eval()
    total_loss = 0
    total_examples = 0
    total_correct_examples = 0
    total_characters = 0
    correct_characters = 0
    total_squared_error = 0
    all_labels = []
    printed_examples = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            try:
                # Try normal forward pass
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
            except ValueError as e:
                if "Expected input batch_size" in str(e):
                    # If there's a batch size mismatch, try without labels
                    logging.warning("Batch size mismatch detected. Running model without labels.")
                    outputs = model(input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    
                    # Calculate loss manually if shapes match
                    if logits.size(0) == labels.size(0):
                        # Shift logits and labels for next token prediction
                        shift_logits = logits[:, :-1, :].contiguous()
                        shift_labels = labels[:, 1:].contiguous().clone()
                        
                        # Replace padding tokens with -100 to ignore them in loss
                        shift_labels[shift_labels == tokenizer.pad_token_id] = -100
                        
                        # Cross entropy loss
                        loss_fct = torch.nn.CrossEntropyLoss()
                        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    else:
                        logging.warning(f"Cannot compute loss: logits shape {logits.shape} != labels shape {labels.shape}")
                        loss = torch.tensor(0.0, device=device)  # Dummy loss for this batch
                else:
                    # Re-raise if it's a different error
                    raise e
            
            total_loss += loss.item()
            
            predictions = torch.argmax(logits, dim=-1)
            examplelist = []
            
            # Ensure predictions and input_ids have same batch size
            if predictions.size(0) != input_ids.size(0):
                logging.warning(f"Shape mismatch: predictions {predictions.shape}, input_ids {input_ids.shape}")
                # Skip further processing for this batch with mismatched shapes
                continue
                
            for i in range(len(input_ids)):
                label_indices = (labels[i] != -100).nonzero(as_tuple=True)[0]
                actual_tokens = input_ids[i, label_indices].cpu().numpy()
                
                # Ensure the index is within bounds for predictions 
                valid_indices = label_indices[label_indices <= predictions.size(1)]
                if len(valid_indices) > 0:
                    predicted_indices = valid_indices - 1  # Adjust for causal LM prediction
                    predicted_indices = predicted_indices[predicted_indices >= 0]  # Ensure non-negative
                    if len(predicted_indices) > 0:
                        predicted_tokens = predictions[i, predicted_indices].cpu().numpy()
                        
                        actual_label = tokenizer.decode(actual_tokens, skip_special_tokens=True).strip()
                        predicted_label = tokenizer.decode(predicted_tokens, skip_special_tokens=True).strip()

                        if actual_label == predicted_label:
                            total_correct_examples += 1
                        total_examples += 1

                        if is_numeric(predicted_label) and is_numeric(actual_label):
                            actual_value = float(actual_label)
                            predicted_value = float(predicted_label)
                            total_squared_error += (actual_value - predicted_value) ** 2
                            all_labels.append(actual_value)

                        max_len = max(len(actual_label), len(predicted_label))
                        padded_actual = actual_label.ljust(max_len)
                        padded_predicted = predicted_label.ljust(max_len)
                        
                        correct_characters += sum(1 for a, p in zip(padded_actual, padded_predicted) if a == p)
                        total_characters += max_len

                        if print_labels and printed_examples < max_print_examples:
                            examplelist.append(f"({predicted_label}, {actual_label})")
                            printed_examples += 1

            if print_labels and examplelist:
                logging.info(" ".join(examplelist))

    if total_examples == 0:
        logging.warning("No valid examples were processed during evaluation.")
        return 0.0, (0.0, 0.0), 0.0, 0.0
        
    avg_loss = total_loss / len(dataloader)
    whole_number_accuracy = total_correct_examples / total_examples if total_examples > 0 else 0.0
    digit_wise_accuracy = correct_characters / total_characters if total_characters > 0 else 0.0

    if all_labels:
        mean_label = sum(all_labels) / len(all_labels)
        total_variance = sum((label - mean_label) ** 2 for label in all_labels)
        mse = total_squared_error / len(all_labels)
        r2 = 1 - (total_squared_error / total_variance) if total_variance > 0 else float('nan')
    else:
        mse = -1
        r2 = float('nan')

    return avg_loss, (whole_number_accuracy, digit_wise_accuracy), mse, r2

def evaluate_xval(model, test_loader, xval, device, print_labels=False, max_print=10):
    """
    Evaluation loop for models using the xval module.
    """
    logging.info('Evaluation start')
    model.eval()
    xval.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0
    total_squared_error = 0
    total_digits = 0
    correct_digits = 0
    printed_examples = 0
    all_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
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

            predicted_numbers = xval.compute_prediction(last_token_hidden_state)
            
            tolerance = 0.5  # Example tolerance value
            correct_predictions = torch.abs(predicted_numbers - labels) < tolerance
            
            total_correct += correct_predictions.sum().item()
            total_samples += labels.size(0)
            all_labels.extend(labels.cpu().numpy())

            for i in range(labels.size(0)):
                actual_value = str(labels[i].item())
                predicted_value = str(predicted_numbers[i].item())
                min_len = len(actual_value)
                correct_digits += sum(1 for a, p in zip(actual_value[:min_len], predicted_value[:min_len]) if a == p)
                total_digits += len(actual_value)

            loss = xval.compute_loss(last_token_hidden_state, labels)
            total_loss += loss.item()
            total_squared_error += torch.sum((predicted_numbers - labels) ** 2).item()

            if print_labels and printed_examples < max_print:
                output_pairs = []
                for i in range(len(labels)):
                    if printed_examples >= max_print:
                        break
                    actual_label = labels[i].cpu().numpy()
                    predicted_label = predicted_numbers[i].cpu().numpy()
                    output_pairs.append((predicted_label, actual_label))
                    printed_examples += 1
                logging.info("Predictions and Labels: " + " ".join(f"({pred},{lbl})" for pred, lbl in output_pairs))

    avg_loss = total_loss / len(test_loader)
    whole_number_accuracy = total_correct / total_samples
    digit_wise_accuracy = correct_digits / total_digits
    mse = total_squared_error / total_samples

    if total_samples > 1:
        mean_label = sum(all_labels) / len(all_labels)
        total_variance = sum((label - mean_label) ** 2 for label in all_labels)
        r2 = 1 - (total_squared_error / total_variance) if total_variance > 0 else float('nan')
    else:
        r2 = float('nan')

    return avg_loss, (whole_number_accuracy, digit_wise_accuracy), mse, r2

def evaluate_vanilla(model, test_loader, vanilla_model, intermediate_network, device, print_labels=False, max_print=10):
    """
    Evaluation loop for models using the vanilla embedding module.
    """
    model.eval()
    vanilla_model.eval()
    intermediate_network.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0
    total_squared_error = 0
    total_digits = 0
    correct_digits = 0
    mispredictions = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            scatter_tensor = batch['scatter_tensor'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            last_token_mask = batch['last_token_mask'].to(device)

            regular_embeddings = get_regular_embeddings(model, input_ids)
            vanilla_embeddings = vanilla_model(scatter_tensor)
            vanilla_embeddings = intermediate_network(vanilla_embeddings)
            input_embeddings = regular_embeddings + vanilla_embeddings
            
            # Match dtype of inputs with model
            input_embeddings = input_embeddings.to(model.dtype)
            attention_mask = attention_mask.to(model.dtype)

            outputs = model(inputs_embeds=input_embeddings, attention_mask=attention_mask, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]
            last_token_hidden_state = (last_hidden_state * last_token_mask.unsqueeze(-1)).sum(dim=1)

            predicted_numbers = vanilla_model.compute_prediction(last_token_hidden_state)
            
            tolerance = 10 ** (-vanilla_model.frac_digit_len)
            correct = torch.abs(predicted_numbers - labels) < tolerance
            total_correct += correct.sum().item()
            total_samples += labels.size(0)
            
            scaled_labels = (labels * (10 ** vanilla_model.frac_digit_len)).long()
            scaled_preds = (predicted_numbers * (10 ** vanilla_model.frac_digit_len)).long()
            
            for i in range(labels.size(0)):
                label_digits = []
                pred_digits = []
                # Extract digits from label
                num = scaled_labels[i]
                for p in vanilla_model.powers_of_ten:
                    label_digits.append((num // p) % 10)
                # Extract digits from prediction
                num = scaled_preds[i]
                for p in vanilla_model.powers_of_ten:
                    pred_digits.append((num // p) % 10)
                # Compare digit-by-digit
                for l, p in zip(label_digits, pred_digits):
                    if l == p:
                        correct_digits += 1
                    total_digits += 1

            for i in range(labels.size(0)):
                if not correct[i]:
                    mispredictions.append((predicted_numbers[i].item(), labels[i].item()))

            loss = vanilla_model.compute_loss(last_token_hidden_state, labels)
            total_loss += loss.item()
            total_squared_error += torch.sum((predicted_numbers - labels) ** 2).item()

    avg_loss = total_loss / len(test_loader)
    accuracy = total_correct / total_samples
    digit_accuracy = correct_digits / total_digits
    mse = total_squared_error / total_samples
    all_labels_tensor = torch.cat([batch['labels'].to(device) for batch in test_loader])
    mean_label = all_labels_tensor.mean()
    total_variance = torch.sum((all_labels_tensor - mean_label) ** 2).item()
    r2 = 1 - (total_squared_error / total_variance) if total_variance != 0 else 0

    if print_labels and mispredictions:
        logging.info(f"Mispredictions (first {max_print}):")
        for pred, true in mispredictions[:max_print]:
            logging.info(f"Predicted: {pred:.5f}, True: {true:.5f}")

    return avg_loss, (accuracy, digit_accuracy), mse, r2
