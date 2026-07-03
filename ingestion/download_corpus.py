import arxiv
import time
import csv
import os
import requests

def main():
    queries = [
        "retrieval augmented generation",
        "reranking large language models",
        "agentic AI reasoning",
        "self-reflective RAG",
        "tool use language models"
    ]
    
    max_results_per_query = 20
    output_dir = os.path.join("data", "raw")
    csv_file_path = os.path.join(output_dir, "paper_list.csv")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    downloaded_ids = set()
    records = []
    
    # Initialize arxiv Client
    # We will still use explicit time.sleep to ensure 3-second delay between downloads
    client = arxiv.Client()
    
    for query in queries:
        print(f"\nSearching for: '{query}'...")
        search = arxiv.Search(
            query=query,
            max_results=max_results_per_query,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        try:
            # Fetch results
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
                # Download PDF
                response = requests.get(paper.pdf_url, stream=True, timeout=30)
                if response.status_code == 200:
                    pdf_path = os.path.join(output_dir, f"{arxiv_id}.pdf")
                    with open(pdf_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                else:
                    print(f"  [!] Failed to download {arxiv_id}, status code: {response.status_code}")
                    continue
                
                # Add to records
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
                
                # Delay between downloads to respect rate limits
                time.sleep(3)
            except Exception as e:
                print(f"  [!] Error downloading {arxiv_id}: {e}")
                
        # Delay between queries
        time.sleep(3)

    # Write to CSV
    print(f"\nWriting metadata to {csv_file_path}...")
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as f:
        fieldnames = ["arxiv_id", "title", "authors", "published_date", "pdf_url", "search_query"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        for record in records:
            writer.writerow(record)
            
    print(f"\nDone! Downloaded {len(records)} unique papers.")

if __name__ == "__main__":
    main()
