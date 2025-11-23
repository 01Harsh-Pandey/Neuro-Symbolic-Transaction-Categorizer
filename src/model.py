import pandas as pd
import torch
import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

# CONFIG
DATA_PATH = "data/synthetic_dataset.csv"
MODEL_DIR = "models/"
os.makedirs(MODEL_DIR, exist_ok=True)

class TransactionDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # Calculate Macro F1 (The Winning Metric)
    f1 = f1_score(labels, preds, average='macro')
    return {'macro_f1': f1}

def main():
    print("⚡ LOAD: Reading synthetic dataset...")
    df = pd.read_csv(DATA_PATH)
    
    # 1. Prepare Labels (Category > Subcategory)
    # We predict the granular "Subcategory" directly
    labels = df['label'].unique().tolist()
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for i, l in enumerate(labels)}
    
    # Save the label map for the App later
    joblib.dump(label2id, os.path.join(MODEL_DIR, "label_map.joblib"))
    joblib.dump(id2label, os.path.join(MODEL_DIR, "id2label.joblib"))
    
    df['label_id'] = df['label'].map(label2id)
    
    # 2. Split Data (Stratified to ensure rare categories are in both sets)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['raw_text'].tolist(), 
        df['label_id'].tolist(), 
        test_size=0.2, 
        stratify=df['label_id'], # Crucial for Imbalanced Data
        random_state=42
    )

    # 3. Tokenization (DistilBERT)
    print("🧩 TOKENIZE: Processing text...")
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    tokenizer.save_pretrained(MODEL_DIR) # Save for inference
    
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=64)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=64)

    train_dataset = TransactionDataset(train_encodings, train_labels)
    val_dataset = TransactionDataset(val_encodings, val_labels)

    # 4. Training (The Heavy Lifting)
    print("🏋️ TRAINING: Fine-tuning DistilBERT...")
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', 
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,              # 3 Epochs is enough for synthetic data
        per_device_train_batch_size=16,  # Adjust if GPU memory is low
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        eval_strategy="epoch",     # Check F1 at end of every epoch
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none"  # Disable wandb/tensorboard if not needed
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Save the best model from training
    trainer.save_model(os.path.join(MODEL_DIR, "distilbert_final"))
    
    # 5. Evaluation Report
    print("📊 EVALUATING: Generating Report...")
    preds_output = trainer.predict(val_dataset)
    y_preds = np.argmax(preds_output.predictions, axis=1)
    
    # Calculate metrics
    macro_f1 = f1_score(val_labels, y_preds, average='macro')
    micro_f1 = f1_score(val_labels, y_preds, average='micro')
    weighted_f1 = f1_score(val_labels, y_preds, average='weighted')
    
    report = classification_report(val_labels, y_preds, target_names=labels, digits=4)
    
    print("\n" + "="*60)
    print("FINAL CLASSIFICATION REPORT")
    print("="*60)
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Micro F1: {micro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print("\nDetailed Report:")
    print(report)
    
    # Write report to file (Deliverable)
    with open("metrics_report.txt", "w") as f:
        f.write("FINAL CLASSIFICATION REPORT\n")
        f.write("="*50 + "\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n")
        f.write(f"Micro F1: {micro_f1:.4f}\n")
        f.write(f"Weighted F1: {weighted_f1:.4f}\n\n")
        f.write(report)

    # 6. Quantization (The "Pro Move")
    print("\n📦 QUANTIZATION: Compressing model for CPU inference...")
    
    # Load the best model for quantization
    model_path = os.path.join(MODEL_DIR, "distilbert_final")
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    
    # Move to CPU for quantization
    model.to('cpu')
    
    # Set model to evaluation mode for quantization
    model.eval()
    
    # Apply dynamic quantization to linear layers
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear},  # Quantize the linear layers
        dtype=torch.qint8
    )
    
    # Save Quantized Model
    quantized_model_path = os.path.join(MODEL_DIR, "quantized_model")
    os.makedirs(quantized_model_path, exist_ok=True)
    
    # Save both the state dict and the model config
    torch.save(quantized_model.state_dict(), os.path.join(quantized_model_path, "pytorch_model.bin"))
    model.config.save_pretrained(quantized_model_path)
    
    # Also save the full quantized model for easy loading
    torch.save(quantized_model, os.path.join(MODEL_DIR, "quantized_model.pt"))
    
    print(f"✅ SUCCESS: Quantized model saved to {MODEL_DIR}quantized_model/")
    
    # Model size comparison
    original_size = sum(p.numel() * p.element_size() for p in model.parameters())
    quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())
    
    print(f"📏 Model Size Reduction: {original_size / 1024 / 1024:.2f}MB → {quantized_size / 1024 / 1024:.2f}MB")
    print(f"💾 Compression Ratio: {original_size/quantized_size:.2f}x")
    
    # Final validation with quantized model
    print("\n🔍 Validating quantized model performance...")
    quantized_model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i in range(min(100, len(val_dataset))):
            sample = val_dataset[i]
            inputs = {key: val.unsqueeze(0) for key, val in sample.items() if key != 'labels'}
            outputs = quantized_model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1)
            correct += (pred == sample['labels']).item()
            total += 1
    
    quantized_accuracy = correct / total
    print(f"✅ Quantized Model Sample Accuracy: {quantized_accuracy:.4f} ({correct}/{total})")
    
    # Final success message
    print("\n🎯 TRAINING COMPLETE!")
    print(f"📈 Target Metric: Macro F1 = {macro_f1:.4f}")
    if macro_f1 > 0.90:
        print("🏆 GOAL ACHIEVED: Macro F1 > 0.90!")
    else:
        print("⚠️  Goal not yet achieved. Consider training for more epochs or tuning hyperparameters.")

if __name__ == "__main__":
    main()