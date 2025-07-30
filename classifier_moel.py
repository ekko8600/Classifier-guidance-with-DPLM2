import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import json
from datetime import datetime
warnings.filterwarnings('ignore')

from byprot.models.dplm2 import MultimodalDiffusionProteinLanguageModel as DPLM2

class ProteinDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=600):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.sequences)
    
    def _create_input(self, sequence):
        seq_len = len(sequence)
        struct_token = self.tokenizer.all_tokens[50] if len(self.tokenizer.all_tokens) > 50 else self.tokenizer.struct_mask_token
        seq_struct = struct_token * seq_len
        seq_struct = (
            self.tokenizer.struct_cls_token + 
            seq_struct + 
            self.tokenizer.struct_eos_token
        )
        seq_aa = (
            self.tokenizer.aa_cls_token + 
            sequence + 
            self.tokenizer.aa_eos_token
        )
        return seq_struct, seq_aa
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        if sequence is None or len(sequence) == 0:
            sequence = "A" * 10
        
        try:
            seq_struct, seq_aa = self._create_input(sequence)
            
            batch_struct = self.tokenizer.batch_encode_plus(
                [seq_struct], add_special_tokens=False, padding="max_length",
                max_length=self.max_length // 2, truncation=True, return_tensors="pt"
            )
            batch_aatype = self.tokenizer.batch_encode_plus(
                [seq_aa], add_special_tokens=False, padding="max_length", 
                max_length=self.max_length // 2, truncation=True, return_tensors="pt"
            )

            input_tokens = torch.concat([batch_struct["input_ids"], batch_aatype["input_ids"]], dim=1)
            attention_mask = torch.concat([batch_struct["attention_mask"], batch_aatype["attention_mask"]], dim=1)

            input_tokens = input_tokens.squeeze(0)
            attention_mask = attention_mask.squeeze(0)
            
        except Exception:
            input_tokens = torch.full((self.max_length,), self.tokenizer.pad_token_id, dtype=torch.long)
            attention_mask = torch.zeros(self.max_length, dtype=torch.bool)
        
        return {
            'input_ids': input_tokens,
            'labels': torch.tensor(label, dtype=torch.long),
            'attention_mask': attention_mask.bool()
        }

class Classifier(nn.Module):
    def __init__(self, hidden_size, num_labels=5, dropout=0.0):
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


class PretrainFinetuneTrainer:
    def __init__(self, model, classifier, device='cuda'):
        self.model = model
        self.classifier = classifier
        self.device = device
        
        self.output_dir = f"./training_outputs_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def prep_data(self, sequences, labels, test_size=0.2, batch_size=4):
        clean_seqs, clean_labs = [], []
        for seq, lab in zip(sequences, labels):
            if seq and len(seq) > 0:
                clean_seqs.append(seq)
                clean_labs.append(lab)

        if len(clean_seqs) < 5 or test_size == 0.0:
            train_seqs, val_seqs = clean_seqs, clean_seqs
            train_labs, val_labs = clean_labs, clean_labs
        else:
            train_seqs, val_seqs, train_labs, val_labs = train_test_split(
                clean_seqs, clean_labs, test_size=test_size, random_state=42, stratify=clean_labs
            )
        
        train_dataset = ProteinDataset(train_seqs, train_labs, self.model.tokenizer)
        val_dataset = ProteinDataset(val_seqs, val_labs, self.model.tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        return train_loader, val_loader
    
    def pretrain_phase(self, train_loader, val_loader, epochs=50, lr=1e-4, save_name="pretrained_model"):
        print("="*60)
        print(f"开始预训练阶段 - {epochs} epochs")
        print("="*60)
        
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        self.classifier.to(self.device)

        optimizer = AdamW(self.classifier.parameters(), lr=lr, weight_decay=1e-5)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
        print(f"Optimizer: AdamW, Scheduler: Linear Warmup, Total steps: {total_steps}")
        
        all_labels = [b['labels'].tolist() for b in train_loader]
        all_labels = [item for sublist in all_labels for item in sublist]
        
        class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'val_loss': [], 'val_acc': [], 'val_f1': []
        }
        
        best_f1 = 0
        patience = 20 # Reduced patience
        no_improve = 0
        
        for epoch in range(epochs):
            self.classifier.train()
            train_loss = 0
            train_preds, train_labels = [], []
            
            progress_bar = tqdm(train_loader, desc=f"预训练 Epoch {epoch+1}")
            for batch in progress_bar:
                try:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model.forward(input_ids=input_ids)
                        hidden_states = outputs["last_hidden_state"]
                    
                    logits = self.classifier(hidden_states, attention_mask)
                    loss = criterion(logits, labels)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    
                    train_loss += loss.item()
                    preds = torch.argmax(logits, dim=-1)
                    train_preds.extend(preds.cpu().numpy())
                    train_labels.extend(labels.cpu().numpy())
                    progress_bar.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])
                    
                except RuntimeError as e:
                    print(f"训练批次跳过: {e}")
                    continue

            val_metrics = self._validate(val_loader, criterion)
            
            if len(train_labels) > 0:
                train_loss /= len(train_loader)
                train_acc = accuracy_score(train_labels, train_preds)
                train_f1 = f1_score(train_labels, train_preds, average='weighted')

                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['train_f1'].append(train_f1)
                history['val_loss'].append(val_metrics['loss'])
                history['val_acc'].append(val_metrics['acc'])
                history['val_f1'].append(val_metrics['f1'])
                
                print(f"预训练 训练 Loss: {train_loss:.3f}, Acc: {train_acc:.3f}, F1: {train_f1:.3f} | 验证 Loss: {val_metrics['loss']:.3f}, Acc: {val_metrics['acc']:.3f}, F1: {val_metrics['f1']:.3f}")
                
                if val_metrics['f1'] > best_f1:
                    best_f1 = val_metrics['f1']
                    no_improve = 0
                    self.save_model(f"{save_name}_best.pth", epoch, val_metrics['f1'], history, is_finetune=False)
                    print(f"预训练最佳模型已保存 (F1: {val_metrics['f1']:.3f})")
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print("预训练早停")
                        break

        self.save_model(f"{save_name}_final.pth", epoch, val_metrics['f1'], history, is_finetune=False)
        print("预训练阶段完成!")
        return history
    
    def finetune_phase(self, train_loader, val_loader, pretrained_path, epochs=30, lr=1e-5, save_name="finetuned_model"):
        print("="*60)
        print(f"开始微调阶段 - {epochs} epochs")
        print("="*60)
        
        self.load_model(pretrained_path, is_finetune=False)
        print(f"已加载预训练权重: {pretrained_path}")

        for param in self.model.parameters():
            param.requires_grad = False
        self.model.train()
        self.classifier.to(self.device)
        self.classifier.train()
        
        optimizer = AdamW([
            {'params': self.model.parameters(), 'lr': lr * 0.1},
            {'params': self.classifier.parameters(), 'lr': lr}
        ], weight_decay=1e-5)

        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
        print(f"Optimizer: AdamW, Scheduler: Linear Warmup, Total steps: {total_steps}")
        
        all_labels = [b['labels'].tolist() for b in train_loader]
        all_labels = [item for sublist in all_labels for item in sublist]
        
        class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'val_loss': [], 'val_acc': [], 'val_f1': []
        }
        
        best_f1 = 0
        patience = 25 # Patience for finetuning
        no_improve = 0
        
        for epoch in range(epochs):
            self.model.train()
            self.classifier.train()
            train_loss = 0
            train_preds, train_labels = [], []
            
            progress_bar = tqdm(train_loader, desc=f"微调 Epoch {epoch+1}")
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model.forward(input_ids=input_ids)
                hidden_states = outputs["last_hidden_state"]
                
                logits = self.classifier(hidden_states, attention_mask)
                loss = criterion(logits, labels)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.model.parameters()) + list(self.classifier.parameters()), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
                progress_bar.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])
                    
            val_metrics = self._validate(val_loader, criterion)
            
            if len(train_labels) > 0:
                train_loss /= len(train_loader)
                train_acc = accuracy_score(train_labels, train_preds)
                train_f1 = f1_score(train_labels, train_preds, average='weighted')
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['train_f1'].append(train_f1)
                history['val_loss'].append(val_metrics['loss'])
                history['val_acc'].append(val_metrics['acc'])
                history['val_f1'].append(val_metrics['f1'])
                
                print(f"微调 训练 Loss: {train_loss:.3f}, Acc: {train_acc:.3f}, F1: {train_f1:.3f} | 验证 Loss: {val_metrics['loss']:.3f}, Acc: {val_metrics['acc']:.3f}, F1: {val_metrics['f1']:.3f}")

                if val_metrics['f1'] > best_f1:
                    best_f1 = val_metrics['f1']
                    no_improve = 0
                    self.save_model(f"{save_name}_best.pth", epoch, val_metrics['f1'], history, is_finetune=True)
                    print(f"微调最佳模型已保存 (F1: {val_metrics['f1']:.3f})")
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print("微调早停")
                        break

        self.save_model(f"{save_name}_final.pth", epoch, val_metrics['f1'], history, is_finetune=True)
        print("微调阶段完成!")
        return history
    
    def _validate(self, val_loader, criterion):
        self.model.eval()
        self.classifier.eval()
        val_loss = 0
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model.forward(input_ids=input_ids)
                    hidden_states = outputs["last_hidden_state"]
                    
                    logits = self.classifier(hidden_states, attention_mask)
                    loss = criterion(logits, labels)
                    
                    val_loss += loss.item()
                    preds = torch.argmax(logits, dim=-1)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
                    
                except RuntimeError:
                    continue
        
        if len(val_labels) > 0:
            val_loss /= len(val_loader)
            val_acc = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)
        else:
            val_loss, val_acc, val_f1 = 0, 0, 0
            
        return {'loss': val_loss, 'acc': val_acc, 'f1': val_f1}
    
    def save_model(self, filename, epoch, f1_score, history, is_finetune=False):
        save_path = os.path.join(self.output_dir, filename)
        save_obj = {
            'classifier_state_dict': self.classifier.state_dict(),
            'epoch': epoch,
            'f1_score': f1_score,
            'history': history,
            'config': {
                'hidden_size': self.classifier.hidden_size,
                'num_labels': self.classifier.num_labels
            }
        }
        if is_finetune:
            save_obj['model_state_dict'] = self.model.state_dict()
        
        torch.save(save_obj, save_path)

        history_path = os.path.join(self.output_dir, filename.replace('.pth', '_history.json'))
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
    
    def load_model(self, filepath, is_finetune=False):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        
        if is_finetune and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])

        print(f"模型加载成功，epoch: {checkpoint.get('epoch', 'unknown')}, F1: {checkpoint.get('f1_score', 'unknown')}")

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"数据文件未找到: {path}")
    df = pd.read_csv(path)
    sequences = df['mutated_sequence'].tolist()
    labels = df['lable'].tolist()
    
    clean_seqs, clean_labs = [], []
    for seq, lab in zip(sequences, labels):
        if isinstance(seq, str) and len(seq) > 0:
            clean_seqs.append(seq)
            clean_labs.append(lab)
    
    return clean_seqs, clean_labs

def plot_training_comparison(pretrain_history, finetune_history, output_dir):
    plt.figure(figsize=(18, 6))
    plt.rcParams['axes.unicode_minus'] = False 

    plt.subplot(1, 3, 1)
    if pretrain_history and pretrain_history.get('val_f1'): plt.plot(pretrain_history['val_f1'], 'b-', label='pretrain val F1', linewidth=2)
    if finetune_history and finetune_history.get('val_f1'): plt.plot(finetune_history['val_f1'], 'r-', label='finetune val F1', linewidth=2)
    plt.title('F1', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    if pretrain_history and pretrain_history.get('val_loss'): plt.plot(pretrain_history['val_loss'], 'b-', label='pretrain val Loss', linewidth=2)
    if finetune_history and finetune_history.get('val_loss'): plt.plot(finetune_history['val_loss'], 'r-', label='finetune val Loss', linewidth=2)
    plt.title('loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    if pretrain_history and pretrain_history.get('val_acc'): plt.plot(pretrain_history['val_acc'], 'b-', label='pretrain val acc', linewidth=2)
    if finetune_history and finetune_history.get('val_acc'): plt.plot(finetune_history['val_acc'], 'r-', label='finetune val acc', linewidth=2)
    plt.title('acc', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    model = DPLM2.from_pretrained("airkingbd/dplm2_650m").to(device)
    hidden_size = model.net.config.hidden_size
    classifier = Classifier(hidden_size=hidden_size, num_labels=5, dropout=0.1)
    trainer = PretrainFinetuneTrainer(model, classifier, device)
    large_sequences, large_labels = load_data('./data_classify/GFP_AEQVI_Sarkisyan_2016.csv')

    print("\n" + "="*60)
    print("第一阶段：在大数据集上预训练")
    print("="*60)
    pretrain_train_loader, pretrain_val_loader = trainer.prep_data(
        large_sequences, large_labels, batch_size=32, test_size=0.1
    )
    pretrain_history = trainer.pretrain_phase(
        pretrain_train_loader, pretrain_val_loader, 
        epochs=100, lr=3e-4, save_name="pretrained_model"
    )

    print("\n" + "="*60)
    print("第二阶段：在小数据集上微调")
    print("="*60)
    small_sequences, small_labels = load_data('./data_classify/GFP_AEQVI_Sarkisyan_2016.csv')
    finetune_train_loader, finetune_val_loader = trainer.prep_data(
        small_sequences, small_labels, batch_size=32, test_size=0.1
    )
    
    pretrained_model_path = os.path.join(trainer.output_dir, "pretrained_model_best.pth")
    if not os.path.exists(pretrained_model_path):
        pretrained_model_path = os.path.join(trainer.output_dir, "pretrained_model_final.pth")


    finetune_history = trainer.finetune_phase(
        finetune_train_loader, finetune_val_loader, 
        pretrained_model_path, epochs=30, lr=1e-4, save_name="finetuned_model"
    )

    plot_training_comparison(pretrain_history, finetune_history, trainer.output_dir)
    
    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)

if __name__ == "__main__":
    main()
