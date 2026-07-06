import arxiv
import time
import csv
import os
import requests
import pandas as pd

def main():
    queries = [
        "function calling large language models agents",
        "tool augmented LLM agents"
    ]
    
    max_results_per_query = 20
    output_dir = os.path.join("data", "raw")
    csv_file_path = os.path.join(output_dir, "paper_list.csv")
    
    os.makedirs(output_dir, exist_ok=True)
    
    downloaded_ids = set()
    if os.path.exists(csv_file_path):
        df_existing = pd.read_csv(csv_file_path)
        downloaded_ids.update(df_existing['arxiv_id'].astype(str).tolist())
    
    records = []
    client = arxiv.Client()
    
    for query in queries:
        print(f"\nSearching for: '{query}'...")
        search = arxiv.Search(
            query=query,
            max_results=max_results_per_query,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        try:
            results = list(client.results(search))
        except Exception as e:
            print(f"Error searching for '{query}': {e}")
            time.sleep(3)
            continue
            
        for paper in results:
            arxiv_id = paper.get_short_id()
            if arxiv_id in downloaded_ids:
                print(f"  [-] Skipping duplicate: {paper.title} ({arxiv_id})")
                continue
                
            print(f"  [+] Downloading: {paper.title} ({arxiv_id})")
            try:
                response = requests.get(paper.pdf_url, stream=True, timeout=30)
                if response.status_code == 200:
                    pdf_path = os.path.join(output_dir, f"{arxiv_id}.pdf")
                    with open(pdf_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                else:
                    print(f"  [!] Failed to download {arxiv_id}, status code: {response.status_code}")
                    continue
                
                authors = ", ".join([author.name for author in paper.authors])
                records.append({
                    "arxiv_id": arxiv_id,
                    "title": paper.title,
                    "authors": authors,
                    "published_date": paper.published.isoformat(),
                    "pdf_url": paper.pdf_url,
                    "search_query": query
                })
                downloaded_ids.add(arxiv_id)
                time.sleep(3)
            except Exception as e:
                print(f"  [!] Error downloading {arxiv_id}: {e}")
        time.sleep(3)

    if records:
        print(f"\nAppending metadata to {csv_file_path}...")
        df_new = pd.DataFrame(records)
        df_new.to_csv(csv_file_path, mode='a', header=not os.path.exists(csv_file_path), index=False)
        print(f"\nDone! Downloaded {len(records)} unique papers.")
    else:
        print("\nNo new papers downloaded.")

if __name__ == "__main__":
    main()
