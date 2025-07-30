import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import gc
from Bio import SeqIO
from tqdm import tqdm
from byprot.models.dplm2 import MultimodalDiffusionProteinLanguageModel as DPLM2
from generate_dplm2 import initialize_generation, save_results

def sample_from_categorical(logits=None, temperature=1.0):
    with torch.no_grad():
        if temperature > 0:
            dist = torch.distributions.Categorical(logits=logits.div(temperature))
            tokens = dist.sample()
            scores = dist.log_prob(tokens)
        else:
            scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)
        return tokens, scores

def stochastic_sample(logits=None, temperature=1.0, noise_scale=1.0):
    with torch.no_grad():
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        logits = logits + noise_scale * gumbel_noise
        tokens, scores = sample_from_categorical(logits, temperature)
        return tokens, scores

class Classifier(nn.Module):
    def __init__(self, hidden_size, num_labels=5, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=8, 
            dim_feedforward=hidden_size * 4, 
            dropout=dropout, 
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.feature_extractor = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_size // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_labels)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, hidden_states, attention_mask=None):
        key_padding_mask = ~attention_mask.bool() if attention_mask is not None else None
        
        extracted_features = self.feature_extractor(hidden_states, src_key_padding_mask=key_padding_mask)

        if attention_mask is not None:
            masked_hidden_states = extracted_features * attention_mask.unsqueeze(-1).float()
            sum_hidden_states = torch.sum(masked_hidden_states, dim=1)
            sum_mask = torch.sum(attention_mask, dim=1).unsqueeze(-1)
            pooled_output = sum_hidden_states / (sum_mask.clamp(min=1e-9))
        else:
            pooled_output = torch.mean(extracted_features, dim=1)

        logits = self.head(pooled_output)
        return logits

class GuidedDPLM2(DPLM2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classifier = None
        self.guidance_history = []
        self.memory_stats = []
    
    def set_classifier(self, classifier):
        self.classifier = classifier
        self.classifier.eval()
    
    def _log_memory(self, step, description=""):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            self.memory_stats.append({
                'step': step,
                'description': description,
                'allocated_gb': allocated,
                'reserved_gb': reserved
            })
    
    def _cleanup_memory(self, *tensors):
        for tensor in tensors:
            if tensor is not None:
                del tensor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _maintain_history(self, history, max_length=5):
        if isinstance(history, list) and len(history) > max_length:
            removed_items = history[:-max_length]
            for item in removed_items:
                del item
            history = history[-max_length:]
            self._cleanup_memory()
        return history
    
    def _get_guidance_schedule(self, step, max_step, schedule_type="constant"):
        if schedule_type == "constant":
            return 1.0
        elif schedule_type == "linear_decay":
            return 1.0 - (step / max_step)
        elif schedule_type == "cosine_decay":
            return 0.5 * (1 + math.cos(math.pi * step / max_step))
        elif schedule_type == "warm_up":
            warmup_steps = max_step // 4
            if step < warmup_steps:
                return step / warmup_steps
            else:
                return 1.0
        else:
            return 1.0
    
    def _get_lm_head_weight(self):
        if hasattr(self.net, 'get_output_embeddings'):
            output_embeddings = self.net.get_output_embeddings()
            if output_embeddings is not None:
                return output_embeddings.weight
        return None
    
    def _gradient_guided_forward(self, output_tokens, target_class, classification_weight, region_mask=None):
        batch_size = output_tokens.size(0)
        target_tensor = torch.tensor([target_class] * batch_size, 
                                   dtype=torch.long, device=output_tokens.device)
        
        try:
            with torch.no_grad():
                net_out = self.forward(input_ids=output_tokens)
                hidden_states_detached = net_out["last_hidden_state"]
                logits = net_out["logits"]
            hidden_states = hidden_states_detached.detach().requires_grad_(True)

            attention_mask = output_tokens.ne(self.pad_id)
            if region_mask is not None:
                classifier_attention_mask = attention_mask & region_mask
            else:
                classifier_attention_mask = attention_mask

            classifier_logits = self.classifier(hidden_states, classifier_attention_mask)

            classification_loss = F.cross_entropy(classifier_logits, target_tensor)

            hidden_grad = torch.autograd.grad(
                outputs=classification_loss,
                inputs=hidden_states,
                grad_outputs=torch.ones_like(classification_loss),
                retain_graph=False,
                create_graph=False,
                only_inputs=True,
                allow_unused=True
            )[0]

            if hidden_grad is not None:
                grad_norm = hidden_grad.norm(dim=-1, keepdim=True) + 1e-8
                hidden_grad_unit = hidden_grad / grad_norm

                lm_head_weight = self._get_lm_head_weight()
                if lm_head_weight is not None:
                    logits_adjustment = torch.matmul(hidden_grad_unit, lm_head_weight.T)
                    
                    guided_logits = logits.detach() - classification_weight * logits_adjustment           #############修改
                else:
                    guided_logits = logits.detach()
            else:
                guided_logits = logits.detach()

            with torch.no_grad():
                pred_class = classifier_logits.argmax(dim=-1)
                pred_probs = torch.softmax(classifier_logits, dim=-1)
                target_prob = pred_probs[torch.arange(batch_size), target_tensor]
                
                guidance_info = {
                    'classification_loss': classification_loss.item(),
                    'predicted_class': pred_class.cpu().tolist() if pred_class.numel() > 1 else pred_class.item(),
                    'target_probability': target_prob.cpu().tolist() if target_prob.numel() > 1 else target_prob.item(),
                    'gradient_norm': hidden_grad.norm(dim=-1).mean().item() if hidden_grad is not None else 0.0,
                    'guidance_applied': hidden_grad is not None,
                    'max_target_prob': target_prob.max().item() if target_prob.numel() > 1 else target_prob.item(),
                    'mean_target_prob': target_prob.mean().item() if target_prob.numel() > 1 else target_prob.item()
                }
            
            return guided_logits, guidance_info
            
        except Exception as e:
            print(f"梯度引导计算出错: {e}")
            import traceback
            traceback.print_exc()
            with torch.no_grad():
                net_out = self.forward(input_ids=output_tokens)
                return net_out["logits"], {'error': str(e)}
        
        finally:
            self._cleanup_memory()
    
    def _energy_guided_forward(self, output_tokens, target_class, classification_weight, region_mask=None):
        with torch.no_grad():
            batch_size = output_tokens.size(0)
            target_tensor = torch.tensor([target_class] * batch_size, 
                                    dtype=torch.long, device=output_tokens.device)
            
            net_out = self.forward(input_ids=output_tokens)
            logits = net_out["logits"]
            hidden_states = net_out["last_hidden_state"]
            
            attention_mask = output_tokens.ne(self.pad_id)
            if region_mask is not None:
                classifier_attention_mask = attention_mask & region_mask
            else:
                classifier_attention_mask = attention_mask
            
            classifier_logits = self.classifier(hidden_states, classifier_attention_mask)
            classifier_probs = torch.softmax(classifier_logits, dim=-1)

            target_probs = classifier_probs[torch.arange(batch_size), target_tensor]
            energy_bias = classification_weight * target_probs.unsqueeze(-1).unsqueeze(-1)
            
            guided_logits = logits + energy_bias
            
            pred_class = classifier_logits.argmax(dim=-1)
            classifier_probs = torch.softmax(classifier_logits, dim=-1)
            target_probs = classifier_probs[torch.arange(batch_size), target_tensor]
            
            guidance_info = {
                'predicted_class': pred_class.cpu().tolist() if pred_class.numel() > 1 else pred_class.item(),
                'target_probability': target_probs.cpu().tolist() if target_probs.numel() > 1 else target_probs.item(),
                #'energy_bias': target_log_probs.cpu().tolist() if target_log_probs.numel() > 1 else target_log_probs.item(),
                'max_target_prob': target_probs.max().item() if target_probs.numel() > 1 else target_probs.item(),
                'mean_target_prob': target_probs.mean().item() if target_probs.numel() > 1 else target_probs.item()
            }
            return guided_logits, guidance_info

    def forward_decoder(
        self,
        prev_decoder_out,
        target_class=None,
        need_attn_weights=False,
        partial_masks=None,
        sampling_strategy="annealing@2.2:1.0",
        classification_weight=0.1,
        guidance_mode="gradient",
        region_mask=None,
    ):
        output_tokens = prev_decoder_out["output_tokens"].clone()
        output_scores = prev_decoder_out["output_scores"].clone()
        step, max_step = prev_decoder_out["step"], prev_decoder_out["max_step"]
        temperature = prev_decoder_out["temperature"]
        history = prev_decoder_out["history"]

        output_masks = self.get_non_special_symbol_mask(
            output_tokens, partial_masks=partial_masks
        )
        
        guidance_info = {}
        
        if self.classifier is not None and target_class is not None:
            if guidance_mode == "gradient":
                logits, guidance_info = self._gradient_guided_forward(
                    output_tokens, target_class, classification_weight, region_mask=region_mask
                )
                attentions = None
            else:
                logits, guidance_info = self._energy_guided_forward(
                    output_tokens, target_class, classification_weight, region_mask=region_mask
                )
                attentions = None
        else:
            with torch.no_grad():
                net_out = self.forward(input_ids=output_tokens)
                logits = net_out["logits"]
                attentions = net_out["attentions"] if need_attn_weights else None

        if logits.dtype != output_scores.dtype:
            logits = logits.type_as(output_scores)

        type_ids = self.get_modality_type(output_tokens)
        aa_position = type_ids.eq(self.aa_type) & output_masks
        struct_position = type_ids.eq(self.struct_type) & output_masks
        indices_aa = torch.where(aa_position)
        indices_struct = torch.where(struct_position)

        logits[indices_aa[0], indices_aa[1], 33:] = -math.inf
        logits[indices_struct[0], indices_struct[1], :33] = -math.inf
        logits[..., self.special_token_list] = -math.inf

        if sampling_strategy == "argmax":
            log_probs = F.log_softmax(logits, dim=-1)
            _scores, _tokens = log_probs.max(-1)
        elif sampling_strategy == "gumbel_argmax":
            noise_scale = temperature
            _tokens, _scores = stochastic_sample(
                logits, temperature=0.0, noise_scale=noise_scale
            )
            _tokens.masked_scatter_(~output_masks, output_tokens[~output_masks])
        elif sampling_strategy.startswith("annealing"):
            max_temp, min_temp = map(float, sampling_strategy.split("@")[1].split(":"))
            rate = 1 - step / max_step
            current_temp = min_temp + (max_temp - min_temp) * rate
            _tokens, _scores = sample_from_categorical(logits, temperature=current_temp)
        else:
            _tokens, _scores = sample_from_categorical(logits, temperature=temperature)

        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])
        
        history = self._maintain_history(history, max_length=5)
        history.append(output_tokens.clone())

        self._cleanup_memory()

        return dict(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attentions=attentions,
            step=step + 1,
            max_step=max_step,
            history=history,
            guidance_info=guidance_info,
        )


    def generate_guided(
        self,
        input_tokens,
        target_class,
        template_tokens=None,
        max_iter=500,
        temperature=1.0,
        classification_weight=0.1,
        guidance_mode="gradient",
        guidance_schedule="constant",
        apply_guidance_steps=None,
        unmasking_strategy="stochastic1.0",
        sampling_strategy="annealing@2.2:1.0",
        partial_masks=None,
        region_mask=None,
        max_history_length=5,
        clear_cache_frequency=5,
        enable_memory_monitoring=True,
        **kwargs
    ):
        if self.classifier is None:
            raise ValueError("Classifier not set")
        self.eval()
        self.classifier.eval()

        if template_tokens is None:
            template_tokens = input_tokens
            
        self._cleanup_memory()
        
        encoder_out = self.forward_encoder(input_tokens)
        initial_output_tokens, initial_output_scores = self.initialize_output_tokens(
            input_tokens, encoder_out=encoder_out, partial_masks=partial_masks
        )
        
        prev_decoder_out = dict(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            output_masks=None,
            attentions=None,
            step=0,
            max_step=max_iter,
            history=[initial_output_tokens.clone()],
            temperature=temperature,
            type_ids=self.get_modality_type(initial_output_tokens),
        )
        
        prev_decoder_out["output_masks"] = self.get_non_special_symbol_mask(
            prev_decoder_out["output_tokens"], partial_masks=partial_masks
        )

        self.guidance_history = []
        
        for step in tqdm(range(max_iter), desc="Guided Generation"):
            should_apply_guidance = (
                apply_guidance_steps is None or 
                step in apply_guidance_steps
            )
            
            current_guidance_weight = classification_weight * self._get_guidance_schedule(
                step, max_iter, guidance_schedule
            )
            
            decoder_out = self.forward_decoder(
                prev_decoder_out=prev_decoder_out,
                target_class=target_class if should_apply_guidance else None,
                partial_masks=partial_masks,
                sampling_strategy=sampling_strategy,
                classification_weight=current_guidance_weight,
                guidance_mode=guidance_mode,
                region_mask=region_mask if should_apply_guidance else None,
            )
            
            output_tokens = decoder_out["output_tokens"]
            output_scores = decoder_out["output_scores"]
            guidance_info = decoder_out.get("guidance_info", {})

            if partial_masks is not None:
                fixed_region_mask = partial_masks
                output_tokens.masked_scatter_(fixed_region_mask, template_tokens[fixed_region_mask])
                
            if should_apply_guidance and len(self.guidance_history) < 100:
                history_entry = {
                    'step': step,
                    'guidance_weight': current_guidance_weight,
                    'guidance_mode': guidance_mode,
                    **guidance_info
                }
                self.guidance_history.append(history_entry)

                if step % 50 == 0:
                    print(f"Step {step}: Target prob: {guidance_info.get('mean_target_prob', 0):.4f}, "
                          f"Pred class: {guidance_info.get('predicted_class', 'N/A')}")
            
            non_special_sym_mask = self.get_non_special_symbol_mask(
                prev_decoder_out["output_tokens"], partial_masks=partial_masks
            )
            
            output_masks, result_tokens, result_scores = self._reparam_decoding(
                output_tokens=prev_decoder_out["output_tokens"].clone(),
                output_scores=prev_decoder_out["output_scores"].clone(),
                cur_tokens=output_tokens.clone(),
                cur_scores=output_scores.clone(),
                decoding_strategy=f"reparam-uncond-{unmasking_strategy}-linear",
                xt_neq_x0=prev_decoder_out["output_masks"],
                type_ids=prev_decoder_out["type_ids"].clone(),
                non_special_sym_mask=non_special_sym_mask,
                t=step + 1,
                max_step=max_iter,
            )
            
            if partial_masks is not None:
                fixed_region_mask = partial_masks
                result_tokens.masked_scatter_(fixed_region_mask, input_tokens[fixed_region_mask])

            old_tokens = prev_decoder_out["output_tokens"]
            old_scores = prev_decoder_out["output_scores"]
            
            prev_decoder_out.update(
                output_masks=output_masks,
                output_tokens=result_tokens,
                output_scores=result_scores,
                step=step + 1,
                history=decoder_out["history"],
            )
            
            self._cleanup_memory(old_tokens, old_scores, output_tokens, output_scores)
            
        return prev_decoder_out["output_tokens"]
    
    def get_guidance_history(self):
        return self.guidance_history

def initialize_generation_for_region(
    struct_fasta_path,
    aa_sequence,
    region, 
    tokenizer, 
    device, 
    model
):

    record = next(SeqIO.parse(struct_fasta_path, "fasta"))
    struct_tokens_list = str(record.seq).split(",")
    seq_len = len(struct_tokens_list)
    
    if len(aa_sequence) != seq_len:
        raise ValueError(
            f"氨基酸序列长度 ({len(aa_sequence)}) 与结构序列长度 ({seq_len}) 不匹配。"
        )

    struct_tokens_str = "".join(struct_tokens_list)
    struct_seq_tokenized = tokenizer.struct_cls_token + struct_tokens_str + tokenizer.struct_eos_token

    unmasked_aa_seq = tokenizer.aa_cls_token + aa_sequence + tokenizer.aa_eos_token

    aa_list = list(aa_sequence)
    region_start, region_end = region
    start_idx_in_seq = region_start - 1
    end_idx_in_seq = region_end - 1

    for i in range(start_idx_in_seq, end_idx_in_seq + 1):
        if i < len(aa_list):
            aa_list[i] = tokenizer.aa_mask_token
            
    masked_aa_seq = "".join(aa_list)
    masked_aa_tokenized = tokenizer.aa_cls_token + masked_aa_seq + tokenizer.aa_eos_token

    batch_struct = tokenizer.batch_encode_plus([struct_seq_tokenized], add_special_tokens=False, return_tensors="pt")
    batch_aa_masked = tokenizer.batch_encode_plus([masked_aa_tokenized], add_special_tokens=False, return_tensors="pt")
    batch_aa_unmasked = tokenizer.batch_encode_plus([unmasked_aa_seq], add_special_tokens=False, return_tensors="pt")

    input_tokens = torch.cat([batch_struct["input_ids"], batch_aa_masked["input_ids"]], dim=1).to(device)
    template_tokens = torch.cat([batch_struct["input_ids"], batch_aa_unmasked["input_ids"]], dim=1).to(device)

    type_ids = model.get_modality_type(input_tokens)
    aa_type_mask = type_ids.eq(model.aa_type)

    full_region_mask = torch.zeros_like(input_tokens, dtype=torch.bool)
    struct_part_len = batch_struct["input_ids"].shape[1]

    start_idx_in_tensor = struct_part_len + 1 + start_idx_in_seq
    end_idx_in_tensor   = struct_part_len + 1 + end_idx_in_seq
    full_region_mask[:, start_idx_in_tensor : end_idx_in_tensor + 1] = True

    region_mask  = aa_type_mask & full_region_mask
    partial_masks = aa_type_mask & ~full_region_mask

    return {
        "input_tokens": input_tokens,
        "template_tokens": template_tokens,
        "partial_masks": partial_masks,
        "region_mask": region_mask
    }

def test_guided_generation():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    model = GuidedDPLM2.from_pretrained("airkingbd/dplm2_650m").cuda()
    
    hidden_size = model.net.config.hidden_size
    classifier = Classifier(
        hidden_size, 
        num_labels=5, 
        dropout=0.0
    ).cuda()
    
    checkpoint = torch.load('./training_outputs_20250626-111729/pretrained_model_best.pth')
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    
    model.set_classifier(classifier)
    
    print("测试分类器性能...")
    test_classifier_performance(model, classifier)
    
    # eGFP sequence
    original_aa_sequence = "MSRGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
    
    generation_input = initialize_generation_for_region(
        struct_fasta_path="./results/structure.fasta",
        aa_sequence=original_aa_sequence,
        region=(30, 50),  # 指定要生成和引导的区域
        tokenizer=model.tokenizer,
        device=next(model.parameters()).device,
        model=model
    )
    
    input_tokens = generation_input["input_tokens"]
    template_tokens = generation_input["template_tokens"]
    partial_masks = generation_input["partial_masks"]
    region_mask = generation_input["region_mask"]
    num_samples = 32

    input_tokens = input_tokens.repeat(num_samples, 1).clone()
    template_tokens = template_tokens.repeat(num_samples, 1).clone()
    partial_masks = partial_masks.repeat(num_samples, 1).clone()
    region_mask = region_mask.repeat(num_samples, 1).clone()

    samples = model.generate_guided(
        input_tokens=input_tokens,
        template_tokens=template_tokens,
        partial_masks=partial_masks,
        region_mask=None,  # 使用region_mask来聚焦引导
        target_class=0,
        max_iter=100,
        temperature=0.8,
        classification_weight=10,
        guidance_mode="gradient",
        guidance_schedule="constant",
        unmasking_strategy="stochastic1.0",
        sampling_strategy="annealing@2.0:1.0",
        max_history_length=5,
        clear_cache_frequency=10,
        enable_memory_monitoring=True
    )
    
    samples_dict = {"output_tokens": samples}
    save_results(
        outputs=samples_dict,
        task='inverse_folding',
        save_dir="./results",
        tokenizer=model.tokenizer,
        struct_tokenizer=model.struct_tokenizer,
        save_pdb=False
    )
    
    guidance_history = model.get_guidance_history()
    print("生成完成!")
    
    print("\n生成结果分析:")
    analyze_generated_samples(model, classifier, samples)
    
    return samples, guidance_history

def test_classifier_performance(model, classifier):
    """测试分类器性能"""
    print("测试分类器在原始序列上的表现:")
    original_aa_sequence = "MSRGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
    
    generation_input = initialize_generation_for_region(
        struct_fasta_path="./results/structure.fasta",
        aa_sequence=original_aa_sequence,
        region=(10, 50),
        tokenizer=model.tokenizer,
        device=next(model.parameters()).device,
        model=model
    )
    
    test_tokens = generation_input["template_tokens"]
    
    with torch.no_grad():
        net_out = model(input_ids=test_tokens)
        hidden_states = net_out["last_hidden_state"]
        attention_mask = test_tokens.ne(model.pad_id)
        
        logits = classifier(hidden_states, attention_mask)
        probs = torch.softmax(logits, dim=-1)
        
        print("原始序列的分类概率分布:")
        for i in range(5):
            print(f"类别 {i}: {probs[0, i].item():.4f}")
        
        predicted_class = probs.argmax(dim=-1).item()
        confidence = probs[0, predicted_class].item()
        print(f"预测类别: {predicted_class}, 置信度: {confidence:.4f}")

def analyze_generated_samples(model, classifier, samples):
    """分析生成的样本"""
    model.eval()
    classifier.eval()

    with torch.no_grad():
        net_out = model(input_ids=samples)
        hidden_states = net_out["last_hidden_state"]
        attention_mask = samples.ne(model.pad_id)

        logits = classifier(hidden_states, attention_mask)
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        
        class_counts = torch.bincount(preds, minlength=5)
        print("各类别生成统计:")
        for i in range(5):
            print(f"类别 {i}: {class_counts[i].item()} 个样本")
        
        confidences = []
        for i, (cls_id, prob_vec) in enumerate(zip(preds.cpu(), probs.cpu())):
            conf = prob_vec[cls_id].item()
            confidences.append(conf)
            if i < 10:
                print(f"Sample {i:02d}: 预测类别 = {cls_id}, 置信度 = {conf:.4f}")
        
        print(f"\n置信度统计: 平均 = {sum(confidences)/len(confidences):.4f}, "
              f"最高 = {max(confidences):.4f}, 最低 = {min(confidences):.4f}")


def test_classifier_performance():
    print("测试分类器在各类别上的表现:")
    model = GuidedDPLM2.from_pretrained("airkingbd/dplm2_650m").cuda()
    original_aa_sequence = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
    
    generation_input = initialize_generation_for_region(
        struct_fasta_path="./results/structure.fasta",
        aa_sequence=original_aa_sequence,
        region=(1, 15),
        tokenizer=model.tokenizer,
        device=next(model.parameters()).device,
        model=model
    )
    
    test_tokens = generation_input["template_tokens"]
    
    hidden_size = model.net.config.hidden_size
    
    classifier = Classifier(
        hidden_size, 
        num_labels=5, 
        dropout=0.0
    ).cuda()
    
    checkpoint = torch.load('./training_outputs_20250626-111729/pretrained_model_best.pth')          # path to classifier model
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    
    with torch.no_grad():
        net_out = model(input_ids=test_tokens)
        hidden_states = net_out["last_hidden_state"]
        attention_mask = test_tokens.ne(model.pad_id)
        
        logits = classifier(hidden_states, attention_mask)
        probs = torch.softmax(logits, dim=-1)
        
        print("原始序列的分类概率分布:")
        for i in range(5):
            print(f"类别 {i}: {probs[0, i].item():.4f}")
        
        predicted_class = probs.argmax(dim=-1).item()
        confidence = probs[0, predicted_class].item()
        print(f"预测类别: {predicted_class}, 置信度: {confidence:.4f}")

samples, guidance_history = test_guided_generation()
