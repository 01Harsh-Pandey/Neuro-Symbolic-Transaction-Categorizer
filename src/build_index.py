import pandas as pd
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer

# CONFIG
DATA_PATH = "data/synthetic_dataset.csv"
MODEL_PATH = "models/"
INDEX_FILE = os.path.join(MODEL_PATH, "faiss_index.bin")
META_FILE = os.path.join(MODEL_PATH, "faiss_meta.pkl")

def main():
    if not os.path.exists(DATA_PATH):
        print("❌ Dataset not found.")
        return

    print("🧠 MEMORY: Loading Sentence Transformer...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("📂 DATA: Indexing high-quality references...")
    df = pd.read_csv(DATA_PATH)
    
    # Only index clean/light noise data to ensure good references
    ref_df = df[df['noise_type'].isin(['clean_template', 'typo_light'])].drop_duplicates('raw_text')
    
    print(f"📊 Selected {len(ref_df)} high-quality transactions for indexing")
    
    # Generate embeddings
    embeddings = model.encode(ref_df['raw_text'].tolist(), show_progress_bar=True)
    
    print(f"🔢 Generated embeddings with dimension: {embeddings.shape[1]}")

    # Build FAISS index (using Inner Product for cosine similarity)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings.astype('float32'))
    
    # Save index
    faiss.write_index(index, INDEX_FILE)
    
    # Save Metadata
    meta = ref_df[['raw_text', 'category', 'subcategory']].to_dict('records')
    with open(META_FILE, "wb") as f:
        pickle.dump(meta, f)
        
    print(f"✅ SUCCESS: Indexed {len(ref_df)} high-quality transactions.")
    print(f"💾 Saved index to: {INDEX_FILE}")
    print(f"💾 Saved metadata to: {META_FILE}")
    
    # Print some statistics
    category_counts = ref_df['category'].value_counts()
    print("\n📊 Index Composition by Category:")
    for category, count in category_counts.items():
        print(f"   {category}: {count} transactions")

if __name__ == "__main__":
    main()