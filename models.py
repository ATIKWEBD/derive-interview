from typing import List, Optional
from pydantic import BaseModel

class NumericalData(BaseModel):
    label: str
    value: str
    unit: Optional[str]
    source_span: str

class ExtractedContent(BaseModel):
    content_id: str
    source_url: str
    source_name: str
    content_type: str
    title: str
    body: str
    published_at: Optional[str]
    extracted_at: str
    numerical_data: List[NumericalData]

class SourceMention(BaseModel):
    content_id: str
    source_url: str
    mention_text: str
    source_span: str

class Entity(BaseModel):
    entity_id: str
    canonical_name: str
    entity_type: str
    aliases: List[str]
    source_mentions: List[SourceMention]
    resolution_confidence: float

class Evidence(BaseModel):
    content_id: str
    source_url: str
    source_span: str
    reason: str

class EntitySentiment(BaseModel):
    entity_id: str
    canonical_name: str
    sentiment: str
    sentiment_score: float
    confidence: float
    evidence: List[Evidence]

class LLMCallLog(BaseModel):
    stage: str
    source_url: Optional[str]
    content_ids: List[str]
    timestamp: str
    provider: str
    model: str
    prompt_hash: str
    input_artifacts: List[str]
    output_artifact: str
    estimated_input_tokens: int
    estimated_output_tokens: int