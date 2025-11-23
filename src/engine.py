import sys
import os

# Add the project root to the python path so imports work correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch
import joblib
import faiss
import pickle
import time
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sentence_transformers import SentenceTransformer

# Robust Import Logic
try:
    # Try importing as a module first (e.g., from main.py)
    from src.rules import RuleEngine
except ImportError:
    # Fallback for running script directly inside src/
    try:
        from rules import RuleEngine
    except ImportError:
        # Fallback for running script from root
        from src.rules import RuleEngine

# CONFIG
MODEL_DIR = "models/"
QUANTIZED_MODEL_PATH = os.path.join(MODEL_DIR, "quantized_model", "pytorch_model.bin")
# Fallback if the folder structure is flat
if not os.path.exists(QUANTIZED_MODEL_PATH):
    QUANTIZED_MODEL_PATH = os.path.join(MODEL_DIR, "quantized_model.pt")

class TransactionEngine:
    def __init__(self):
        print("🚀 ENGINE: Initializing Neuro-Symbolic Core...")
        
        # 1. Load Rules
        self.rule_engine = RuleEngine()
        
        # 2. Load AI (Quantized DistilBERT)
        self.device = 'cpu'
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
        self.label_map = joblib.load(os.path.join(MODEL_DIR, "label_map.joblib"))
        self.id2label = {i: l for l, i in self.label_map.items()}
        
        # Load Architecture & Weights
        try:
            # Try to load the full quantized model first
            if os.path.exists(os.path.join(MODEL_DIR, "quantized_model.pt")):
                self.model = torch.load(os.path.join(MODEL_DIR, "quantized_model.pt"), map_location=self.device, weights_only=False)
                print("✅ AI: Loaded full quantized model")
            else:
                # Fallback: Load architecture and apply quantization
                self.model = DistilBertForSequenceClassification.from_pretrained(
                    'distilbert-base-uncased',
                    num_labels=len(self.label_map)
                )
                self.model.to(self.device)
                
                # Apply Dynamic Quantization
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
                
                # Try to load state dict
                try:
                    state_dict = torch.load(QUANTIZED_MODEL_PATH, map_location=self.device)
                    self.model.load_state_dict(state_dict)
                    print("✅ AI: Loaded quantized state dict")
                except Exception as e:
                    print(f"⚠️ Warning: Could not load quantized weights: {e}")
                    # Load from the fine-tuned model
                    try:
                        model_path = os.path.join(MODEL_DIR, "distilbert_final")
                        if os.path.exists(model_path):
                            fine_tuned_model = DistilBertForSequenceClassification.from_pretrained(model_path)
                            self.model.load_state_dict(fine_tuned_model.state_dict())
                            print("✅ AI: Loaded fine-tuned model weights")
                    except Exception as e2:
                        print(f"⚠️ Warning: Could not load fine-tuned weights: {e2}. Using base model.")
                        
        except Exception as e:
            print(f"❌ AI: Failed to load model - {e}")
            raise
            
        self.model.eval()
        
        # 3. Load Semantic Memory (FAISS)
        print("🧠 MEMORY: Loading Vector Store...")
        try:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.index = faiss.read_index(os.path.join(MODEL_DIR, "faiss_index.bin"))
            with open(os.path.join(MODEL_DIR, "faiss_meta.pkl"), "rb") as f:
                self.memory_meta = pickle.load(f)
            print(f"✅ MEMORY: Loaded {len(self.memory_meta)} reference transactions")
        except Exception as e:
            print(f"❌ MEMORY: Failed to load FAISS index - {e}")
            self.index = None
            self.memory_meta = []
            
        print("✅ ENGINE: Ready.")

    def predict(self, text):
        """
        Main prediction method with three-tier architecture
        
        Args:
            text: Raw transaction string
            
        Returns:
            Dictionary with classification results
        """
        start_time = time.time()
        result = {}

        # LEVEL 1: RULES (O(1) matching)
        rule_hit = self.rule_engine.apply(text)
        if rule_hit:
            result = rule_hit
            result['latency_ms'] = round((time.time() - start_time) * 1000, 2)
            result['tier'] = 1
            return result

        # LEVEL 2: AI MODEL (DistilBERT with confidence threshold)
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                confidence, predicted_id = torch.max(probs, dim=1)
                
            conf_score = confidence.item()
            
            # Threshold Check (60% confidence)
            if conf_score > 0.60:
                full_label = self.id2label[predicted_id.item()]
                cat, subcat = full_label.split(" > ")
                result = {
                    "category": cat,
                    "subcategory": subcat,
                    "confidence": round(conf_score, 4),
                    "source": "AI_MODEL",
                    "reason": f"Model Confidence: {conf_score:.1%}",
                    "tier": 2
                }
            else:
                # LEVEL 3: SEMANTIC FALLBACK
                result = self._semantic_fallback(text)
                result['tier'] = 3
                
        except Exception as e:
            print(f"❌ AI prediction failed: {e}")
            # Fallback to semantic search
            result = self._semantic_fallback(text)
            result['tier'] = 3

        result['latency_ms'] = round((time.time() - start_time) * 1000, 2)
        return result

    def _semantic_fallback(self, text):
        """
        Semantic fallback using FAISS similarity search
        
        Args:
            text: Transaction text to classify
            
        Returns:
            Classification result based on similar transactions
        """
        if self.index is None or len(self.memory_meta) == 0:
            return {
                "category": "Unknown",
                "subcategory": "Unknown",
                "confidence": 0.0,
                "source": "FALLBACK_FAILED",
                "reason": "Semantic memory not available"
            }
        
        try:
            # Encode query
            embedding = self.embedder.encode([text])
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embedding)
            
            # Search for nearest neighbor
            D, I = self.index.search(embedding.astype('float32'), 1)
            
            neighbor_idx = I[0][0]
            if neighbor_idx < len(self.memory_meta):
                neighbor_data = self.memory_meta[neighbor_idx]
                similarity_score = D[0][0]  # Cosine similarity (higher is better)
                
                # Convert similarity to confidence (0.0 to 0.8 scale)
                confidence = min(0.8, similarity_score * 0.9)
                
                return {
                    "category": neighbor_data['category'],
                    "subcategory": neighbor_data['subcategory'],
                    "confidence": round(confidence, 4),
                    "source": "SEMANTIC_MEMORY",
                    "reason": f"Similar to: '{neighbor_data['raw_text'][:50]}...' (similarity: {similarity_score:.3f})"
                }
            else:
                return {
                    "category": "Unknown",
                    "subcategory": "Unknown", 
                    "confidence": 0.0,
                    "source": "FALLBACK_FAILED",
                    "reason": "No similar transactions found"
                }
                
        except Exception as e:
            print(f"❌ Semantic fallback failed: {e}")
            return {
                "category": "Unknown",
                "subcategory": "Unknown",
                "confidence": 0.0,
                "source": "FALLBACK_FAILED",
                "reason": f"Semantic search error: {str(e)}"
            }

    def get_engine_info(self):
        """Return information about the loaded engine components"""
        return {
            "rules_loaded": len(self.rule_engine.rules),
            "ai_model_loaded": self.model is not None,
            "semantic_memory_loaded": self.index is not None,
            "semantic_references": len(self.memory_meta),
            "total_categories": len(self.label_map)
        }

# Global instance for easy access
transaction_engine = TransactionEngine()

# Simple Test
if __name__ == "__main__":
    engine = TransactionEngine()
    
    test_cases = [
        "Uber Trip",           # Should be Rule
        "Starbucks 0229",      # Should be AI  
        "Hulu Subscription",   # Should be Semantic (if not in training)
        "SQ *MERCHANT 231",    # Should be Semantic
        "UNKNOWN TRANSACTION"  # Should be Unknown
    ]
    
    print("\n🧪 TESTING TRANSACTION ENGINE")
    print("=" * 60)
    
    for text in test_cases:
        print(f"\n📥 Input: '{text}'")
        result = engine.predict(text)
        
        print(f"   📊 Category: {result['category']} > {result['subcategory']}")
        print(f"   🎯 Confidence: {result['confidence']:.1%}")
        print(f"   🔧 Source: {result['source']} (Tier {result['tier']})")
        print(f"   ⚡ Latency: {result['latency_ms']}ms")
        print(f"   💡 Reason: {result['reason']}")
    
    # Print engine info
    print("\n" + "=" * 60)
    print("🔧 ENGINE INFORMATION")
    info = engine.get_engine_info()
    print(f"   Rules: {info['rules_loaded']} patterns")
    print(f"   AI Model: {'✅' if info['ai_model_loaded'] else '❌'}")
    print(f"   Semantic Memory: {info['semantic_references']} references")
    print(f"   Total Categories: {info['total_categories']}")