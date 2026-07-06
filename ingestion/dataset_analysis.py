import os
import pandas as pd
import fitz  # PyMuPDF
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

def analyze_dataset():
    data_dir = os.path.join("data", "raw")
    csv_path = os.path.join(data_dir, "paper_list.csv")
    
    if not os.path.exists(csv_path):
        print("No paper_list.csv found.")
        return
        
    df = pd.read_csv(csv_path)
    
    total_docs = 0
    total_pages = 0
    total_tokens = 0
    
    section_counts = {
        "Abstract": 0,
        "Introduction": 0,
        "Method/Methodology": 0,
        "Results": 0,
        "Related Work": 0
    }
    
    total_references = 0
    total_figures = 0
    total_tables = 0
    
    documents_texts = [] # for clustering
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        arxiv_id = str(row['arxiv_id'])
        pdf_path = os.path.join(data_dir, f"{arxiv_id}.pdf")
        
        if not os.path.exists(pdf_path):
            continue
            
        try:
            doc = fitz.open(pdf_path)
            num_pages = len(doc)
            total_pages += num_pages
            total_docs += 1
            
            doc_text = ""
            
            for page_num in range(num_pages):
                page = doc[page_num]
                text = page.get_text()
                doc_text += text + "\n"
            
            # Words & tokens
            words = doc_text.split()
            tokens = int(len(words) * 1.3)
            total_tokens += tokens
            
            text_lower = doc_text.lower()
            
            # Sections
            if "abstract" in text_lower: section_counts["Abstract"] += 1
            if "introduction" in text_lower: section_counts["Introduction"] += 1
            if "method" in text_lower or "methodology" in text_lower: section_counts["Method/Methodology"] += 1
            if "results" in text_lower: section_counts["Results"] += 1
            if "related work" in text_lower: section_counts["Related Work"] += 1
            
            # Figure & Table counting
            # Find unique Figure N and Table N mentions
            fig_matches = re.findall(r'(?i)\b(?:figure|fig\.)\s+(\d+)\b', doc_text)
            table_matches = re.findall(r'(?i)\btable\s+(\d+)\b', doc_text)
            
            total_figures += len(set(fig_matches))
            total_tables += len(set(table_matches))
            
            # Reference counting
            # Find the last occurrence of "References" or "Bibliography" heading
            ref_match = list(re.finditer(r'(?im)^[ \t]*(?:\d+\.?\s*)?(?:References|Bibliography)[ \t]*$', doc_text))
            
            ref_count = 0
            if ref_match:
                # Get the text after the last references heading
                ref_text = doc_text[ref_match[-1].end():]
                # Count unique numbered entries like [1], [2]
                bracket_refs = set(re.findall(r'\[(\d+)\]', ref_text))
                if bracket_refs:
                    ref_count = len(bracket_refs)
                else:
                    # Fallback: Count lines starting with a number e.g. "1. Smith et al."
                    line_refs = re.findall(r'(?m)^\s*\d+\.\s+', ref_text)
                    if line_refs:
                        ref_count = len(line_refs)
                    else:
                        # Fallback: count text blocks separated by double newlines in ref_text
                        blocks = [b for b in ref_text.split('\n\n') if len(b.strip()) > 20]
                        if len(blocks) > 0:
                            ref_count = len(blocks)
            total_references += ref_count
            
            # Topic clustering extraction
            # Extract Abstract text
            abstract_text = ""
            abstract_match = re.search(r'(?i)\babstract\b(.*?)(?:\bintroduction\b|\b1\.\s+introduction\b)', doc_text[:15000], flags=re.DOTALL)
            if abstract_match:
                abstract_text = abstract_match.group(1)
            else:
                # Just take the first 1500 characters of the doc text if no abstract found
                abstract_text = doc_text[:1500]
                
            cluster_text = str(row.get('title', '')) + " " + abstract_text
            documents_texts.append(cluster_text)
            
            doc.close()
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            continue

    if total_docs == 0:
        print("No valid PDFs processed.")
        return

    avg_pages = total_pages / total_docs
    avg_tokens = total_tokens / total_docs
    avg_references = total_references / total_docs if total_docs else 0
    avg_figures = total_figures / total_docs if total_docs else 0
    avg_tables = total_tables / total_docs if total_docs else 0
    
    chunks_300 = total_tokens / 300
    chunks_500 = total_tokens / 500
    chunks_800 = total_tokens / 800
    
    num_clusters = min(5, len(documents_texts))
    custom_stop = ['et', 'al', '2023', '2024', '2022', 'figure', 'table', 'arxiv', 'preprint', 'abstract', 'introduction']
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    stop_words = list(ENGLISH_STOP_WORDS) + custom_stop
    
    vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=1000)
    X = vectorizer.fit_transform(documents_texts)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    
    topics = []
    for i in range(num_clusters):
        top_words = [terms[ind] for ind in order_centroids[i, :5]]
        topics.append(f"Cluster {i+1}: " + ", ".join(top_words))
        
    md_content = f"""# Dataset Analysis

## Overview
- **Number of documents:** {total_docs}
- **Total pages:** {total_pages}
- **Average pages per document:** {avg_pages:.1f}
- **Average tokens per document:** {avg_tokens:.1f}
- **Average references per paper:** {avg_references:.1f}
- **Average figures per paper:** {avg_figures:.1f}
- **Average tables per paper:** {avg_tables:.1f}

### So What?
The corpus consists of {total_docs} papers with an average of {avg_pages:.1f} pages and {avg_tokens:.1f} tokens per document. A citation density of {avg_references:.1f} references per paper indicates that multi-hop questions across cited papers will be a common occurrence, making robust citation tracking essential. The presence of figures and tables ({avg_figures:.1f} figures and {avg_tables:.1f} tables per paper on average) suggests that multimodal parsing or table extraction capabilities will be beneficial in Phase 3.

## Chunking Estimates
Estimated total tokens: {total_tokens}
- **300-token chunks:** ~{int(chunks_300)} chunks
- **500-token chunks:** ~{int(chunks_500)} chunks
- **800-token chunks:** ~{int(chunks_800)} chunks

### So What?
These chunk counts provide a baseline for vector database sizing and retrieval latency tests. A chunk size of 500 tokens gives a manageable number of chunks overall, balancing context retention and retrieval speed for local or API-based embeddings.

## Topic Distribution
Rough clustering of titles and abstracts:
"""
    for t in topics:
        md_content += f"- {t}\n"

    md_content += f"""
### So What?
The topics reflect a varied distribution among the chosen domain keywords. Query classification (Phase 4) and metadata filtering (Phase 3) could use these topic clusters to route questions or narrow down context effectively.

## Section Distribution
Papers containing identifiable sections:
- Abstract: {section_counts['Abstract']}
- Introduction: {section_counts['Introduction']}
- Method/Methodology: {section_counts['Method/Methodology']}
- Results: {section_counts['Results']}
- Related Work: {section_counts['Related Work']}

### So What?
Most papers consistently contain standard academic sections. Section-aware or semantic chunking (Phase 3) could be highly effective since core methodology and results are explicitly demarcated, helping avoid context-cutting across logical boundaries.
"""

    with open(os.path.join("docs", "dataset_analysis.md"), "w", encoding='utf-8') as f:
        f.write(md_content)
        
    print("Dataset analysis complete. Results saved to docs/dataset_analysis.md.")

if __name__ == "__main__":
    analyze_dataset()
