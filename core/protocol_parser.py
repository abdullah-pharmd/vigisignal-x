import re
from typing import Optional, Dict, List

def parse_protocol(text: str) -> Optional[Dict[str, List[str]]]:
    """
    Parses a clinical trial protocol text to extract inclusion criteria,
    exclusion criteria, objectives/endpoints, and safety stopping boundaries.
    """
    if text is None or not isinstance(text, str):
        return None
    if not text.strip():
        return None

    
    # Clean and split into lines
    lines = text.split("\n")
    
    # Placeholders
    inclusion = []
    exclusion = []
    endpoints = []
    safety_boundaries = []
    
    current_section = None
    
    for line in lines:
        l_strip = line.strip()
        l_upper = l_strip.upper()
        
        # Section detection
        if "INCLUSION CRITERIA:" in l_upper or "PATIENTS MUST MEET" in l_upper:
            current_section = "inclusion"
            continue
        elif "EXCLUSION CRITERIA:" in l_upper or "PATIENTS MEETING ANY OF THE FOLLOWING EXCLUSION" in l_upper:
            current_section = "exclusion"
            continue
        elif "OBJECTIVES AND ENDPOINTS" in l_upper or "ENDPOINTS" in l_upper:
            current_section = "endpoints"
        elif "SAFETY BOUNDARIES" in l_upper or "STOPPING RULES" in l_upper or "SAFETY MONITORING" in l_upper:
            current_section = "safety"
            
        # Extract items
        if current_section == "inclusion":
            if l_strip.startswith("-") or l_strip.startswith("*") or re.match(r'^\d+\.', l_strip):
                inclusion.append(l_strip)
        elif current_section == "exclusion":
            if l_strip.startswith("-") or l_strip.startswith("*") or re.match(r'^\d+\.', l_strip):
                exclusion.append(l_strip)
        elif current_section == "endpoints":
            if "endpoint" in l_strip.lower() or "objective" in l_strip.lower():
                endpoints.append(l_strip)
        elif current_section == "safety":
            if l_strip.startswith("-") or l_strip.startswith("*") or "discontinuation" in l_strip.lower() or "withdrawal" in l_strip.lower() or "safety" in l_strip.lower():
                safety_boundaries.append(l_strip)
                
    # Fallbacks/heuristics if bullet points are not structured
    if not inclusion:
        inclusion = [line for line in lines if any(x in line.lower() for x in ["aged", "inclusion", "hypertension", "must meet"])]
    if not exclusion:
        exclusion = [line for line in lines if any(x in line.lower() for x in ["history of", "exclusion", "hyperkalemia", "excluded"])]
    if not safety_boundaries:
        safety_boundaries = [line for line in lines if any(x in line.lower() for x in ["stopping", "discontinuation", "monitoring", "withdraw"])]

    return {
        "inclusion": inclusion[:6],
        "exclusion": exclusion[:6],
        "endpoints": endpoints[:5],
        "safety_boundaries": safety_boundaries[:6]
    }
