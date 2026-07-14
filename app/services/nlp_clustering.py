import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import numpy as np

logger = logging.getLogger(__name__)

def cluster_incident_reports(reports: list) -> list:
    """
    Takes a list of database incident dictionaries and clusters them using ML.
    Filters out noise and groups similar reports into verified 'threat clusters'.
    """
    if len(reports) < 2:
        return [] # Need at least 2 similar reports to form a cluster

    # Extract the text descriptions
    texts = [r["description"] for r in reports]
    
    # 1. Vectorize: Convert text descriptions to numerical vectors
    # (Removes common English stop words like 'the', 'and', 'is')
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        X = vectorizer.fit_transform(texts)
    except ValueError:
        return [] # Failsafe if text is empty or lacks vocabulary
    
    # 2. Cluster: Use Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
    # eps controls how similar texts must be. min_samples=2 means 2 reports verify a threat.
    db = DBSCAN(eps=0.8, min_samples=2, metric='euclidean').fit(X.toarray())
    labels = db.labels_
    
    # 3. Group and Format the Results
    clusters = {}
    for idx, label in enumerate(labels):
        if label == -1:
            continue # -1 means 'Noise' (a unique report that doesn't match others)
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(reports[idx])
        
    result = []
    for label, items in clusters.items():
        # Calculate the geographical center (centroid) of the clustered reports
        avg_lat = sum(item["latitude"] for item in items) / len(items)
        avg_lon = sum(item["longitude"] for item in items) / len(items)
        
        result.append({
            "cluster_id": int(label),
            "threat_summary": items[0]["description"], # Uses the first report as a summary
            "verified_report_count": len(items),
            "center_latitude": avg_lat,
            "center_longitude": avg_lon,
            "report_ids": [item["id"] for item in items]
        })
        
    logger.info(f"Generated {len(result)} verified threat clusters from {len(reports)} raw reports.")
    return result