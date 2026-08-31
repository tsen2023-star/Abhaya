import logging

logger = logging.getLogger(__name__)

def cluster_incident_reports(reports: list) -> list:
    """
    Takes a list of database incident dictionaries and clusters them using ML.
    Imports are lazy to avoid slow server cold-start times.
    """
    if len(reports) < 2:
        return []

    try:
        # Lazy imports — only load sklearn when this function is actually called
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import DBSCAN
        import numpy as np

        texts = [r["description"] for r in reports]

        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            X = vectorizer.fit_transform(texts)
        except ValueError:
            return []

        db = DBSCAN(eps=0.8, min_samples=2, metric='euclidean').fit(X.toarray())
        labels = db.labels_

        clusters = {}
        for idx, label in enumerate(labels):
            if label == -1:
                continue
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(reports[idx])

        result = []
        for label, items in clusters.items():
            avg_lat = sum(item["latitude"] for item in items) / len(items)
            avg_lon = sum(item["longitude"] for item in items) / len(items)
            result.append({
                "cluster_id": int(label),
                "threat_summary": items[0]["description"],
                "verified_report_count": len(items),
                "center_latitude": avg_lat,
                "center_longitude": avg_lon,
                "report_ids": [item["id"] for item in items]
            })

        logger.info(f"Generated {len(result)} verified threat clusters from {len(reports)} raw reports.")
        return result

    except Exception as e:
        logger.error(f"Clustering failed: {e}")
        return []