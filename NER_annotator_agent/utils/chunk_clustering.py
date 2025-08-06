import json
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

def load_data(path):
    """Carica i dati dal file .ejsonl"""
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def compute_embeddings(texts, model_name):
    """Calcola gli embeddings per i testi"""
    print(f"[INFO] Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"[INFO] Computing embeddings for {len(texts)} texts...")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return embeddings, model

def plot_elbow_method(embeddings, max_k=20):
    """Visualizza il grafico per l'elbow method e restituisce le inerzie"""
    print(f"[INFO] Computing elbow method for k from 2 to {max_k}...")
    inertias = []
    k_range = range(2, max_k + 1)
    
    for k in tqdm(k_range, desc="Computing K-Means"):
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        km.fit(embeddings)
        inertias.append(km.inertia_)
    
    # Calcola la seconda derivata per suggerire k ottimale
    if len(inertias) >= 2:
        second_derivative = np.diff(inertias, n=2)
        suggested_k = np.argmin(second_derivative) + 4  # +4 perché partiamo da k=2 e diff riduce di 2
        if suggested_k > max_k:
            suggested_k = max_k
    else:
        suggested_k = 3
    
    # Crea il grafico
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, 'bo-', markersize=8, linewidth=2)
    plt.axvline(x=suggested_k, color='red', linestyle='--', alpha=0.7, 
                label=f'Suggested k = {suggested_k}')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia (Within-cluster Sum of Squares)')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Aggiungi annotazioni per alcuni punti
    for i, (k, inertia) in enumerate(zip(k_range, inertias)):
        if i % 3 == 0 or k == suggested_k:  # Mostra ogni 3° punto o il suggerito
            plt.annotate(f'k={k}', (k, inertia), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    print(f"[INFO] Suggested optimal k based on second derivative: {suggested_k}")
    return inertias, suggested_k

def cluster_embeddings(embeddings, k):
    """Esegue il clustering con K-Means"""
    print(f"[INFO] Clustering with k = {k}...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(embeddings)
    
    # Stampa informazioni sui cluster
    unique, counts = np.unique(labels, return_counts=True)
    print("[INFO] Cluster distribution:")
    for cluster_id, count in zip(unique, counts):
        print(f"  Cluster {cluster_id}: {count} documents")
    
    return labels, kmeans.cluster_centers_

def retrieve_top_k_per_centroid(embeddings, entries, labels, centroids, top_k=5):
    """Recupera i top-k documenti più vicini a ogni centroide"""
    print(f"[INFO] Retrieving top-{top_k} documents per centroid...")
    results = []
    
    for cluster_id in range(len(centroids)):
        centroid = centroids[cluster_id]
        # Trova tutti i documenti in questo cluster
        indices_in_cluster = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
        
        if len(indices_in_cluster) == 0:
            print(f"[WARNING] Cluster {cluster_id} is empty!")
            continue
        
        # Calcola similarità coseno con il centroide
        cluster_embeddings = embeddings[indices_in_cluster]
        similarities = cosine_similarity([centroid], cluster_embeddings)[0]
        
        # Prendi i top-k (o tutti se sono meno di k)
        k_to_take = min(top_k, len(similarities))
        top_indices = np.argsort(similarities)[-k_to_take:][::-1]  # Dal più simile al meno simile
        
        print(f"  Cluster {cluster_id}: selected {len(top_indices)} documents (similarities: {similarities[top_indices]})")
        
        for rank, idx in enumerate(top_indices):
            global_idx = indices_in_cluster[idx]
            entry = entries[global_idx].copy()  # Copia tutti i metadati originali
            entry["cluster_id"] = int(cluster_id)
            entry["similarity_to_centroid"] = float(similarities[idx])
            entry["rank_in_cluster"] = rank + 1
            entry["embedding"] = embeddings[global_idx].tolist()
            results.append(entry)
    
    return results

def save_output(data, path):
    """Salva i risultati in formato JSON"""
    print(f"[INFO] Saving {len(data)} documents to {path}...")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description="Cluster documents and extract top-k per centroid")
    
    parser.add_argument("--input", type=str, default="/home/tiziano/langgraph_agents/NER_annotator_agent/data/output/gemini_tender_ner_dataset.jsonl", help="Path to input .jsonl file")
    parser.add_argument("--output0", type=str, default="/home/tiziano/langgraph_agents/NER_annotator_agent/data/output/tender0_rag_db.json", help="Path to output .jsonl file for chunk_id 0")
    parser.add_argument("--output1", type=str, default="/home/tiziano/langgraph_agents/NER_annotator_agent/data/output/tender1_rag_db.json", help="Path to output .jsonl file for chunk_id 1")
    
    parser.add_argument("--model", type=str, default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", help="Embedding model name")
    parser.add_argument("--top_k", type=int, default=2, help="Top-K documents to retrieve per cluster centroid")
    parser.add_argument("--max_k", type=int, default=6, help="Maximum number of clusters to test for elbow method")
    parser.add_argument("--auto_k", action="store_true", help="Use automatically suggested k instead of manual selection")

    args = parser.parse_args()

    # Ora puoi usare l'oggetto args come faresti normalmente
    print(f"File di input: {args.input}")
    print(f"File di output per chunk 0: {args.output0}")
    print(f"File di output per chunk 1: {args.output1}")
    print(f"Modello: {args.model}")
    print(f"Top K: {args.top_k}")
    print(f"Max K: {args.max_k}")
    print(f"Auto K: {args.auto_k}")
    
    # Carica i dati
    print(f"[INFO] Loading data from {args.input}...")
    entries = load_data(args.input)
    print(f"[INFO] Loaded {len(entries)} documents")

    # --- TODO: Modifica la logica qui ---
    # Splitta i dati in base al 'chunk_id'
    entries0 = [entry for entry in entries if entry.get("chunk_id") == "0"]
    entries1 = [entry for entry in entries if entry.get("chunk_id") == "1"]
    
    print(f"[INFO] Found {len(entries0)} documents for chunk_id 0")
    print(f"[INFO] Found {len(entries1)} documents for chunk_id 1")

    # Funzione per eseguire il clustering e salvare i risultati
    def process_chunk(entries_list, output_file, args):
        if not entries_list:
            print(f"[WARNING] No documents to process for this chunk. Skipping...")
            return

        texts = [entry["text"] for entry in entries_list]
        
        # Calcola embeddings
        embeddings, model = compute_embeddings(texts, args.model)
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        
        # Mostra elbow method
        inertias, suggested_k = plot_elbow_method(embeddings, args.max_k)
        
        # Scegli k
        if args.auto_k:
            k_chosen = suggested_k
            print(f"[INFO] Using automatically suggested k = {k_chosen}")
        else:
            while True:
                try:
                    k_input = input(f"\nEnter number of clusters (suggested: {suggested_k}, range: 2-{args.max_k}) for {output_file}: ")
                    k_chosen = int(k_input)
                    if 2 <= k_chosen <= args.max_k:
                        break
                    else:
                        print(f"Please enter a number between 2 and {args.max_k}")
                except ValueError:
                    print("Please enter a valid integer")
        
        print(f"[INFO] Using k = {k_chosen} clusters")
        
        # Esegui clustering
        labels, centroids = cluster_embeddings(embeddings, k_chosen)
        
        # Recupera top-k per centroide
        results = retrieve_top_k_per_centroid(embeddings, entries_list, labels, centroids, args.top_k)
        
        # Salva risultati
        save_output(results, output_file)
        
        print(f"[DONE] Successfully processed {len(entries_list)} documents into {k_chosen} clusters")
        print(f"[DONE] Saved {len(results)} top-k documents to {output_file}")
        
        # Statistiche finali
        total_docs_saved = len(results)
        expected_docs = k_chosen * args.top_k
        print(f"[INFO] Expected max documents: {expected_docs}, Actual: {total_docs_saved}")

    # Processa il primo gruppo
    print("\n--- Processing chunk_id 0 ---")
    process_chunk(entries0, args.output0, args)

    # Processa il secondo gruppo
    print("\n--- Processing chunk_id 1 ---")
    process_chunk(entries1, args.output1, args)
    
if __name__ == "__main__":
    main()