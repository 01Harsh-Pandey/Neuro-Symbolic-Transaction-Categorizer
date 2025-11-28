import sys
import os

# --- PATH SETUP (CRITICAL FOR CLOUD) ---
# Get the absolute path to the project root (one level up from src/)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR) # This is the project root

# Add root to python path
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

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
    from src.rules import RuleEngine
except ImportError:
    try:
        from rules import RuleEngine
    except ImportError:
        # Last resort fallback
        sys.path.append(os.path.join(BASE_DIR, 'src'))
        from rules import RuleEngine

# --- CONFIGURATION ---
# Use absolute paths to prevent "File Not Found" errors on Cloud
MODEL_DIR = os.path.join(BASE_DIR, "models")
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
        
        # --- FIX: USE HUB TOKENIZER ---
        # We load the tokenizer from HuggingFace Hub to avoid missing local file errors
        # This is safe because the tokenizer vocabulary doesn't change for fine-tuning
        try:
            print("⏳ AI: Downloading standard tokenizer...")
            self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        except Exception:
            # Fallback to local if internet fails (unlikely on cloud)
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)

        # Load Label Map
        label_map_path = os.path.join(MODEL_DIR, "label_map.joblib")
        if os.path.exists(label_map_path):
            self.label_map = joblib.load(label_map_path)
            self.id2label = {i: l for l, i in self.label_map.items()}
        else:
            print("❌ AI: Label map not found! Check models/ folder.")
            self.label_map = {}
            self.id2label = {}
        
        # Load Architecture & Weights
        try:
            # Try to load the full quantized model first
            full_model_path = os.path.join(MODEL_DIR, "quantized_model.pt")
            if os.path.exists(full_model_path):
                self.model = torch.load(full_model_path, map_location=self.device)
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
            # Don't raise here, allow the app to run in "Rules Only" mode if AI fails
            self.model = None
            
        if self.model:
            self.model.eval()
        
        # 3. Load Semantic Memory (FAISS)
        print("🧠 MEMORY: Loading Vector Store...")
        try:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            
            index_path = os.path.join(MODEL_DIR, "faiss_index.bin")
            meta_path = os.path.join(MODEL_DIR, "faiss_meta.pkl")
            
            if os.path.exists(index_path) and os.path.exists(meta_path):
                self.index = faiss.read_index(index_path)
                with open(meta_path, "rb") as f:
                    self.memory_meta = pickle.load(f)
                print(f"✅ MEMORY: Loaded {len(self.memory_meta)} reference transactions")
            else:
                print("⚠️ MEMORY: FAISS index or meta file missing. Semantic fallback disabled.")
                self.index = None
                self.memory_meta = []
                
        except Exception as e:
            print(f"❌ MEMORY: Failed to load FAISS index - {e}")
            self.index = None
            self.memory_meta = []
            
        print("✅ ENGINE: Ready.")

    def predict(self, text):
        """
        Main prediction method with three-tier architecture
        """
        start_time = time.time()
        result = {}

        # LEVEL 1: RULES (O(1) matching)
        try:
            rule_hit = self.rule_engine.apply(text)
            if rule_hit:
                result = rule_hit
                result['latency_ms'] = round((time.time() - start_time) * 1000, 2)
                result['tier'] = 1
                return result
        except Exception as e:
            print(f"⚠️ Rule Engine Error: {e}")

        # LEVEL 2: AI MODEL (DistilBERT)
        if self.model is not None:
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
                result = self._semantic_fallback(text)
                result['tier'] = 3
        else:
            # If Model failed to load, go straight to fallback
            result = self._semantic_fallback(text)
            result['tier'] = 3

        result['latency_ms'] = round((time.time() - start_time) * 1000, 2)
        return result

    def _semantic_fallback(self, text):
        """Semantic fallback using FAISS similarity search"""
        if self.index is None or len(self.memory_meta) == 0:
            return {
                "category": "Unknown",
                "subcategory": "Unknown",
                "confidence": 0.0,
                "source": "FALLBACK_FAILED",
                "reason": "Semantic memory not available",
                "tier": 3
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
                similarity_score = float(D[0][0])  # Cosine similarity
                
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

# --- GLOBAL INSTANCE REMOVED ---
# Do not instantiate here! It causes double loading in Streamlit.
# transaction_engine = TransactionEngine() 

# Simple Test
if __name__ == "__main__":
    # Only instantiate when running this script directly
    engine = TransactionEngine()
    
    test_cases = [
        "Uber Trip",           # Should be Rule
        "Starbucks 0229",      # Should be AI  
        "Hulu Subscription",   # Should be Semantic
        "SQ *MERCHANT 231",    # Should be Semantic
        "UNKNOWN TRANSACTION"  # Should be Unknown
    ]
    
    print("\n🧪 TESTING TRANSACTION ENGINE")
    print("=" * 60)
    
    for text in test_cases:
        print(f"\n📥 Input: '{text}'")
        result = engine.predict(text)
        
        print(f"   📊 Category: {result.get('category')} > {result.get('subcategory')}")
        print(f"   🎯 Confidence: {result.get('confidence', 0):.1%}")
        print(f"   🔧 Source: {result.get('source')} (Tier {result.get('tier')})")
        print(f"   ⚡ Latency: {result.get('latency_ms')}ms")
        print(f"   💡 Reason: {result.get('reason')}")
