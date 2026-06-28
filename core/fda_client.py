import asyncio
import aiohttp
import json
import os
import pandas as pd
from typing import Dict, Any, List

# API URL for openFDA drug events
FDA_API_URL = "https://api.fda.gov/drug/event.json"

class FDAClient:
    def __init__(self, api_key: str = None, rate_limit_semaphore: int = 3, delay_between_batches: float = 1.5):
        self.api_key = api_key
        self.sem = asyncio.Semaphore(rate_limit_semaphore)
        self.delay = delay_between_batches

    def _build_url(self, limit: int, skip: int) -> str:
        # Base query for patients >= 60 years old (unit 801 is years)
        query = "patient.patientonsetage:[60 TO 125] AND patient.patientonsetageunit:801"
        url = f"{FDA_API_URL}?search={query}&limit={limit}&skip={skip}"
        if self.api_key:
            url += f"&api_key={self.api_key}"
        return url

    async def fetch_page(self, session: aiohttp.ClientSession, limit: int, skip: int, max_retries: int = 5) -> List[Dict[str, Any]]:
        url = self._build_url(limit, skip)
        
        for retry in range(max_retries):
            async with self.sem:
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            return data.get("results", [])
                        elif response.status == 429:
                            # Rate limit hit, apply exponential backoff
                            wait_time = (2 ** retry) * self.delay
                            print(f"[FDAClient] Rate limit 429. Retrying page skip={skip} in {wait_time:.2f}s...")
                            await asyncio.sleep(wait_time)
                        elif response.status == 403:
                            # Forbidden could mean API key missing for limit size
                            data = await response.json()
                            err_msg = data.get("error", {}).get("message", "")
                            print(f"[FDAClient] HTTP 403 Forbidden for page skip={skip}. Msg: {err_msg}")
                            return []
                        else:
                            # Other HTTP errors
                            print(f"[FDAClient] HTTP {response.status} for page skip={skip}. Retrying...")
                            await asyncio.sleep(self.delay)
                except Exception as e:
                    print(f"[FDAClient] Error fetching page skip={skip}: {e}. Retrying...")
                    await asyncio.sleep(self.delay)
        
        print(f"[FDAClient] Failed to fetch page skip={skip} after {max_retries} retries.")
        return []

    async def fetch_all_reports(self, num_pages: int = None, page_size: int = None) -> List[Dict[str, Any]]:
        # Auto-configure based on API key presence
        if self.api_key:
            limit = page_size if page_size is not None else 1000
            pages = num_pages if num_pages is not None else 10
        else:
            limit = page_size if page_size is not None else 100
            pages = num_pages if num_pages is not None else 100
            
        print(f"[FDAClient] Starting parallel fetch of {pages} pages (size={limit})...")
        
        # Open aiohttp session
        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(pages):
                skip = i * limit
                # openFDA skips are restricted to skip + limit <= 10000 keyless or up to 25000 with key
                # So we clamp skip to stay within bounds
                max_skip = 25000 if self.api_key else 10000
                if skip + limit > max_skip:
                    print(f"[FDAClient] Reached maximum pagination index ({skip + limit} > {max_skip}). Stopping queue.")
                    break
                    
                tasks.append(self.fetch_page(session, limit, skip))
                # Small staggered start to prevent initial spike
                await asyncio.sleep(0.15)
                
            results = await asyncio.gather(*tasks)
            
            # Flatten lists of reports
            all_reports = []
            for page in results:
                all_reports.extend(page)
                
            print(f"[FDAClient] Completed fetch. Total raw reports retrieved: {len(all_reports)}")
            return all_reports

    def process_and_filter_reports(self, raw_reports: List[Dict[str, Any]]) -> pd.DataFrame:
        print("[FDAClient] Processing and filtering raw reports...")
        processed_records = []
        
        for r in raw_reports:
            report_id = r.get("safetyreportid", "")
            receive_date = r.get("receivedate", "")  # YYYYMMDD format
            patient = r.get("patient", {})
            if not patient:
                continue
                
            # Double check age criteria (60+)
            age_str = patient.get("patientonsetage", "")
            age_unit = patient.get("patientonsetageunit", "")
            
            try:
                age = float(age_str) if age_str else 0.0
            except ValueError:
                age = 0.0
                
            # Filter: age >= 60 (assuming years / unit 801, or fallback check)
            if age < 60 or age_unit != "801":
                continue
                
            # Extract drugs list
            drugs_raw = patient.get("drug", [])
            if not isinstance(drugs_raw, list):
                continue
                
            # Clean and extract medicinal product names
            drug_names = []
            for d in drugs_raw:
                prod = d.get("medicinalproduct", "")
                if prod:
                    # Normalize: uppercase and stripped
                    drug_names.append(prod.strip().upper())
            
            # Polypharmacy filter: strictly concomitant drug count > 4
            if len(drug_names) <= 4:
                continue
                
            # Extract reactions list
            reactions_raw = patient.get("reaction", [])
            if not isinstance(reactions_raw, list):
                continue
                
            reaction_names = []
            for re in reactions_raw:
                term = re.get("reactionmeddrapt", "")
                if term:
                    reaction_names.append(term.strip().upper())
                    
            if not reaction_names:
                continue
                
            # Deduplicate lists
            drug_names = list(set(drug_names))
            reaction_names = list(set(reaction_names))
            
            processed_records.append({
                "safetyreportid": report_id,
                "report_date": receive_date,
                "age": age,
                "concomitant_count": len(drug_names),
                "drugs": ";".join(drug_names),
                "reactions": ";".join(reaction_names)
            })
            
        df = pd.DataFrame(processed_records)
        print(f"[FDAClient] Processing complete. Filtered geriatric polypharmacy reports: {len(df)}")
        return df

    def save_data(self, raw_reports: List[Dict[str, Any]], processed_df: pd.DataFrame, base_dir: str = "."):
        # Make directories
        raw_dir = os.path.join(base_dir, "data", "raw")
        proc_dir = os.path.join(base_dir, "data", "processed")
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(proc_dir, exist_ok=True)
        
        # Save raw JSON
        raw_path = os.path.join(raw_dir, "raw_reports.json")
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(raw_reports, f, indent=2)
        print(f"[FDAClient] Saved raw reports to {raw_path}")
        
        # Save processed CSV
        proc_path = os.path.join(proc_dir, "processed_reports.csv")
        processed_df.to_csv(proc_path, index=False, encoding="utf-8")
        print(f"[FDAClient] Saved processed reports to {proc_path}")

# Standalone execution helper
if __name__ == "__main__":
    async def main():
        client = FDAClient()
        # Keyless testing: fetches up to 10 pages of size 100 = 1000 reports
        raw = await client.fetch_all_reports(num_pages=10, page_size=100)
        df = client.process_and_filter_reports(raw)
        client.save_data(raw, df)
        
    asyncio.run(main())
