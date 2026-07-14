import requests
import logging

logger = logging.getLogger(__name__)

def get_safety_score(lat: float, lon: float, radius: int = 500):
    """
    Queries OpenStreetMap for street metadata around a specific location.
    Calculates a safety score based on street lighting and path types.
    """
    # Overpass API query to find ways (streets) with specific safety tags within a radius
    overpass_url = "http://overpass-api.de/api/interpreter"
    query = f"""
    [out:json];
    (
      way(around:{radius},{lat},{lon})["highway"];
    );
    out tags;
    """
    
    try:
        response = requests.post(overpass_url, data={'data': query})
        data = response.json()
        
        total_streets = 0
        safe_signals = 0
        
        for element in data.get('elements', []):
            tags = element.get('tags', {})
            total_streets += 1
            
            # Boost score if the street is explicitly lit
            if tags.get('lit') == 'yes':
                safe_signals += 2
                
            # Boost score if it's a pedestrian or residential zone (generally more foot traffic)
            if tags.get('highway') in ['pedestrian', 'residential', 'footway']:
                safe_signals += 1
                
        # Calculate a baseline safety score out of 100
        if total_streets == 0:
            return 50.0  # Neutral score if no data is found
            
        # Normalizing the score (just a basic heuristic for the prototype)
        raw_score = (safe_signals / (total_streets * 2)) * 100
        final_score = min(max(raw_score + 30, 0), 100) # Base padding so it's rarely 0
        
        logger.info(f"Analyzed {total_streets} streets near {lat}, {lon}. Safety Score: {final_score:.2f}/100")
        return round(final_score, 2)
        
    except Exception as e:
        logger.error(f"Failed to fetch OSM data: {e}")
        return 50.0  # Fallback score