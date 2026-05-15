import json
import os
from fastapi.encoders import isoformat
import requests
import hashlib
from bs4 import BeautifulSoup
from datetime import datetime, timezone
import google.generativeai as genai
from typing import List

# Import our strict Pydantic schemas
from models import ExtractedContent, Entity, EntitySentiment, LLMCallLog

# Global list to hold LLM calls for the audit requirement
llm_calls_log = []

class FinancialPipeline:
    def __init__(self):
        self.sources: List[str] = []
        self.raw_content: dict = {}
        self.extracted_content: List[ExtractedContent] = []
        self.entities: List[Entity] = []
        self.entity_sentiments: List[EntitySentiment] = []
        
        # Configure Gemini API
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("WARNING: GOOGLE_API_KEY environment variable not set.")
        genai.configure(api_key=api_key)

    def log_stage(self, stage_name: str):
        """Logs the pipeline stage to satisfy the auditable requirements."""
        print(f"-> {stage_name}")

    def load_sources(self):
        try:
            with open('sources.json', 'r') as f:
                data = json.load(f)
                self.sources = data.get('sources', [])
                print(f"Loaded {len(self.sources)} sources.")
        except FileNotFoundError:
            print("Error: sources.json not found.")

    def fetch_content(self):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        for url in self.sources:
            try:
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Strip out noisy tags to save tokens
                for element in soup(["script", "style", "header", "footer", "nav", "aside"]):
                    element.extract()
                
                # Get clean text
                clean_text = soup.get_text(separator=' ', strip=True)
                self.raw_content[url] = clean_text
                print(f"Successfully fetched and cleaned content from {url}")
                
            except Exception as e:
                print(f"Failed to fetch content from {url}: {e}")

    def extract_and_normalize(self):
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        for url, text in self.raw_content.items():
            prompt = "Extract the core financial information, headlines, numerical data (with source spans), and context from the following text."
            full_prompt = f"{prompt}\n\nText:\n{text[:15000]}" # Truncating to avoid massive context blowouts if page is huge
            
            try:
                # Force structured JSON output mapping to our Pydantic schema
                response = model.generate_content(
                    full_prompt,
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json",
                        response_schema=ExtractedContent,
                        temperature=0.1 # Keep it deterministic
                    )
                )
                
                # Parse the response back into our Pydantic model
                extracted_data = ExtractedContent.model_validate_json(response.text)
                
                # Enforce required fields that the LLM might miss
                extracted_data.source_url = url
                extracted_data.extracted_at = datetime.now(timezone.utc)
                
                self.extracted_content.append(extracted_data)
                
                # --- Audit Trail Logging ---
                # Estimate tokens (rough approximation: 4 chars per token)
                est_in_tokens = len(full_prompt) // 4
                est_out_tokens = len(response.text) // 4
                
                log_entry = LLMCallLog(
                    stage="CONTENT_EXTRACTED",
                    source_url=url,
                    content_ids=[extracted_data.content_id],
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    provider="Google",
                    model="gemini-2.5-flash",
                    prompt_hash=hashlib.md5(full_prompt.encode()).hexdigest(),
                    input_artifacts=["raw_content dictionary"],
                    output_artifact="extracted_content.json",
                    estimated_input_tokens=est_in_tokens,
                    estimated_output_tokens=est_out_tokens
                )
                
                # Append to JSONL log
                with open('llm_calls.jsonl', 'a') as log_file:
                    log_file.write(log_entry.model_dump_json() + '\n')
                    
                print(f"Successfully extracted content and logged LLM call for {url}")
                
            except Exception as e:
                print(f"Failed to extract content for {url}: {e}")

        # Save the final extracted list to disk
        with open('extracted_content.json', 'w') as f:
            json.dump([item.model_dump(mode='json') for item in self.extracted_content], f, indent=2)

    def resolve_entities(self):
        if not self.extracted_content:
            print("No extracted content to resolve entities from.")
            return

        print("Starting Autonomous Entity Resolution...")
        
        # Aggregate text from all extracted content
        aggregated_text = ""
        content_ids = []
        for item in self.extracted_content:
            aggregated_text += f"--- Content ID: {item.content_id} ---\nTitle: {item.title}\nBody: {item.body}\n\n"
            content_ids.append(item.content_id)

        model = genai.GenerativeModel("gemini-2.5-flash")
        
        prompt = """
        You are an expert financial Named Entity Recognition (NER) system. 
        Analyze the following extracted financial text and identify all relevant entities autonomously (without a glossary).
        Group aliases together (e.g., 'Fed', 'Federal Reserve', 'US central bank' should map to the canonical name 'Federal Reserve').
        Return a strict list of entities with their source mentions and confidence scores.
        """
        
        full_prompt = f"{prompt}\n\nText to analyze:\n{aggregated_text[:30000]}" # Guardrail context length
        
        try:
            response = model.generate_content(
                full_prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=list[Entity], # Expecting a list of our Entity Pydantic model
                    temperature=0.1
                )
            )
            
            # Parse response into a list of Entity objects
            import json as _json # Ensure json is available for raw parsing if needed
            entities_data = _json.loads(response.text)
            self.entities = [Entity(**e) for e in entities_data]
            
            # Save to entities.json
            with open('entities.json', 'w') as f:
                _json.dump([e.model_dump(mode='json') for e in self.entities], f, indent=2)
                
            # Log LLM Call
            est_in_tokens = len(full_prompt) // 4
            est_out_tokens = len(response.text) // 4
            
            log_entry = LLMCallLog(
                stage="ENTITIES_RESOLVED",
                source_url=None, # Global pipeline stage
                content_ids=content_ids,
                timestamp=datetime.now(timezone.utc).isoformat(),
                provider="Google",
                model="gemini-2.5-flash",
                prompt_hash=hashlib.md5(full_prompt.encode()).hexdigest(),
                input_artifacts=["extracted_content list"],
                output_artifact="entities.json",
                estimated_input_tokens=est_in_tokens,
                estimated_output_tokens=est_out_tokens
            )
            
            with open('llm_calls.jsonl', 'a') as log_file:
                log_file.write(log_entry.model_dump_json() + '\n')
                
            print(f"Successfully resolved {len(self.entities)} unique entities.")

        except Exception as e:
            print(f"Failed during entity resolution: {e}")

    def score_sentiment(self):
        if not self.entities or not self.extracted_content:
            print("Missing entities or extracted content for sentiment scoring.")
            return

        print("Starting Entity-Specific Sentiment Scoring...")
        
        # Prepare the context payload
        aggregated_text = ""
        content_ids = []
        for item in self.extracted_content:
            aggregated_text += f"--- Content ID: {item.content_id} ---\nTitle: {item.title}\nBody: {item.body}\n\n"
            content_ids.append(item.content_id)
            
        entities_list_str = "\n".join([f"- ID: {e.entity_id} | Name: {e.canonical_name} (Aliases: {', '.join(e.aliases)})" for e in self.entities])
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        prompt = f"""
        You are an expert financial analyst. Analyze the sentiment for specific entities based ONLY on the provided text.
        Do not provide page-level sentiment. Provide sentiment strictly for the entities listed below.
        CRITICAL: You MUST provide exact quotes from the text as 'source_span' evidence to justify the sentiment.
        CRITICAL: You MUST include the exact 'entity_id' provided below for each sentiment record.
        Unsupported sentiment claims are invalid.
        
        Entities to analyze:
        {entities_list_str}
        """
        
        full_prompt = f"{prompt}\n\nText to analyze:\n{aggregated_text[:30000]}"
        
        try:
            response = model.generate_content(
                full_prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=list[EntitySentiment], 
                    temperature=0.1
                )
            )
            
            # Parse response
            import json as _json
            sentiment_data = _json.loads(response.text)
            self.entity_sentiments = [EntitySentiment(**s) for s in sentiment_data]
            
            # Save to entity_sentiment.json
            with open('entity_sentiment.json', 'w') as f:
                _json.dump([s.model_dump(mode='json') for s in self.entity_sentiments], f, indent=2)
                
            # Log LLM Call
            est_in_tokens = len(full_prompt) // 4
            est_out_tokens = len(response.text) // 4
            
            log_entry = LLMCallLog(
                stage="ENTITY_SENTIMENT_SCORED",
                source_url=None,
                content_ids=content_ids,
                timestamp=datetime.now(timezone.utc).isoformat(),
                provider="Google",
                model="gemini-2.5-flash",
                prompt_hash=hashlib.md5(full_prompt.encode()).hexdigest(),
                input_artifacts=["extracted_content list", "entities list"],
                output_artifact="entity_sentiment.json",
                estimated_input_tokens=est_in_tokens,
                estimated_output_tokens=est_out_tokens
            )
            
            with open('llm_calls.jsonl', 'a') as log_file:
                log_file.write(log_entry.model_dump_json() + '\n')
                
            print(f"Successfully scored sentiment for {len(self.entity_sentiments)} entities.")

        except Exception as e:
            print(f"Failed during sentiment scoring: {e}")
        
    def run_qa_and_conflicts(self):
        print("Running QA and Conflict Detection...")
        import uuid
        import json as _json
        
        qa_issues = []
        
        # Group sentiments by entity_id
        sentiment_map = {}
        for sentiment_record in self.entity_sentiments:
            eid = sentiment_record.entity_id
            if eid not in sentiment_map:
                sentiment_map[eid] = []
            sentiment_map[eid].append(sentiment_record)
            
        # Detect conflicts
        for eid, records in sentiment_map.items():
            unique_sentiments = set([r.sentiment for r in records])
            if len(unique_sentiments) > 1:
                # We have a conflict!
                source_ids = [ev.content_id for r in records for ev in r.evidence]
                
                issue = {
                    "issue_id": str(uuid.uuid4()),
                    "severity": "warning",
                    "issue_type": "conflicting_sentiment",
                    "entities": [eid],
                    "source_content_ids": list(set(source_ids)),
                    "details": f"Found conflicting sentiments ({', '.join(unique_sentiments)}) for entity {eid} across different sources."
                }
                qa_issues.append(issue)
                
        # Save QA Report
        try:
            with open('qa_report.json', 'w') as f:
                _json.dump(qa_issues, f, indent=2)
            print(f"QA complete. Found {len(qa_issues)} conflicts.")
        except Exception as e:
            print(f"Failed to write QA report: {e}")

    def generate_reports(self):
        print("Generating Cost Report and Final Briefings...")
        import json as _json
        
        # --- 1. COST REPORT ---
        total_in_tokens = 0
        total_out_tokens = 0
        
        try:
            if os.path.exists('llm_calls.jsonl'):
                with open('llm_calls.jsonl', 'r') as f:
                    for line in f:
                        if line.strip():
                            log_data = _json.loads(line)
                            total_in_tokens += log_data.get('estimated_input_tokens', 0)
                            total_out_tokens += log_data.get('estimated_output_tokens', 0)
                            
            # Rough cost calculation based on standard Gemini 1.5 Flash pricing
            cost_in = (total_in_tokens / 1_000_000) * 0.075
            cost_out = (total_out_tokens / 1_000_000) * 0.30
            total_cost = cost_in + cost_out
            
            cost_report = {
                "total_estimated_input_tokens": total_in_tokens,
                "total_estimated_output_tokens": total_out_tokens,
                "estimated_total_cost_usd": round(total_cost, 6),
                "efficiency_strategy": "Deduplicated HTML tags (removed scripts, styles, navs) before sending to LLM to heavily reduce input token size and save costs."
            }
            
            with open('cost_report.json', 'w') as f:
                _json.dump(cost_report, f, indent=2)
            print("Cost report generated.")
            
        except Exception as e:
            print(f"Failed to generate cost report: {e}")

        # --- 2. MULTI-AUDIENCE REPORTS ---
        if not self.entities or not self.entity_sentiments:
            print("No data available to generate reports.")
            return
            
        os.makedirs('reports', exist_ok=True)
        
        entities_str = _json.dumps([e.model_dump(mode='json') for e in self.entities])
        sentiments_str = _json.dumps([s.model_dump(mode='json') for s in self.entity_sentiments])
        
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        # Generate Trader Brief
        try:
            print("Drafting Trader Brief...")
            trader_prompt = f"Act as a quantitative trader. Based strictly on the following entity and sentiment data, write a concise, action-oriented 'Trader Brief' focusing on price levels and momentum. Use Markdown formatting.\n\nEntities: {entities_str[:15000]}\n\nSentiments: {sentiments_str[:15000]}"
            trader_response = model.generate_content(trader_prompt)
            with open('reports/trader_brief.md', 'w') as f:
                f.write(trader_response.text)
        except Exception as e:
            print(f"Failed to draft Trader Brief: {e}")
            
        # Generate Executive Summary
        try:
            print("Drafting Executive Summary...")
            exec_prompt = f"Act as a Chief Risk Officer. Based strictly on the following entity and sentiment data, write a high-level 'Executive Summary' focusing on macro trends and risks. Use Markdown formatting.\n\nEntities: {entities_str[:15000]}\n\nSentiments: {sentiments_str[:15000]}"
            exec_response = model.generate_content(exec_prompt)
            with open('reports/executive_summary.md', 'w') as f:
                f.write(exec_response.text)
            print("Multi-audience reports generated successfully in reports/ directory.")
        except Exception as e:
            print(f"Failed to draft Executive Summary: {e}")

    def run(self):
        """Executes the strict sequential pipeline."""
        self.log_stage("INIT")
        
        self.load_sources()
        self.log_stage("SOURCES_LOADED")
        
        self.fetch_content()
        self.log_stage("CONTENT_FETCHED")
        
        self.extract_and_normalize()
        self.log_stage("CONTENT_EXTRACTED")
        self.log_stage("CONTENT_NORMALISED")
        
        self.resolve_entities()
        self.log_stage("ENTITIES_EXTRACTED")
        self.log_stage("ENTITIES_RESOLVED")
        
        self.score_sentiment()
        self.log_stage("ENTITY_SENTIMENT_SCORED")
        
        self.run_qa_and_conflicts()
        self.log_stage("QA_AND_CONFLICTS_CHECKED")
        
        self.generate_reports()
        self.log_stage("REPORTS_GENERATED")
        
        self.log_stage("RESULTS_FINALISED")

if __name__ == "__main__":
    pipeline = FinancialPipeline()
    pipeline.run()