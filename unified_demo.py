import os
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ------------------------- Request/Response Models ------------------------- #
class Message(BaseModel):
    original_form: str
    speaker: Optional[int] = 0
    label: Optional[int] = None

class ChatAnalysisRequest(BaseModel):
    messages: List[Message]
    model_name: Optional[str] = "roberta-base"

class WindowAnalysisRequest(BaseModel):
    messages: List[Message]
    message_indices: Optional[List[int]] = None
    model_name: Optional[str] = "roberta-base"

class ConversationAnalysisRequest(BaseModel):
    conversation: Dict[str, Any]
    model_name: Optional[str] = "roberta-base"

class WindowPrediction(BaseModel):
    start: int
    end: int
    prediction: int
    confidence: float
    anomaly_probability: float
    message_indices: Optional[List[int]] = None
    message_anomaly_scores: Optional[List[float]] = None

class AnalysisResponse(BaseModel):
    prediction: int
    confidence: float
    anomaly_probability: float
    model_used: str
    windows: Optional[List[WindowPrediction]] = None
    anomaly_windows: Optional[int] = None
    normal_windows: Optional[int] = None
    total_windows: Optional[int] = None
    detailed_results: Optional[List[Dict[str, Any]]] = None
    message_anomaly_map: Optional[Dict[int, float]] = None

class WindowAnalysisResponse(BaseModel):
    prediction: int
    confidence: float
    anomaly_probability: float
    model_used: str
    anomaly_indices: List[int]
    message_scores: Optional[Dict[int, float]] = None

class FileAnalysisResponse(BaseModel):
    total_conversations: int
    anomaly_conversations: int
    normal_conversations: int
    detection_rate: float
    model_used: str
    conversations: List[Dict[str, Any]]

# ------------------------- Model Architecture ------------------------- #
class ConversationAnomalyDetector(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int = 2,
        hidden_dropout: float = 0.3,
        attention_dropout: float = 0.1,
        use_attention_pooling: bool = True,
        msg_token_id: Optional[int] = None,
        attn_supervision_weight: float = 0.5
    ):
        super().__init__()
        
        self.transformer = AutoModel.from_pretrained(model_name)
        self.config = self.transformer.config
        hidden_size = self.config.hidden_size
        
        self.dropout = nn.Dropout(hidden_dropout)
        self.msg_token_id = msg_token_id
        self.attn_supervision_weight = attn_supervision_weight
        
        self.use_attention_pooling = use_attention_pooling
        if use_attention_pooling:
            self.attention_weights = nn.Linear(hidden_size, 1)
            self.attention_dropout = nn.Dropout(attention_dropout)
        
        # Message-level attention scoring layer
        self.attn_scoring = nn.Linear(hidden_size, 1)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(hidden_size // 2, num_labels)
        )
    
    def forward(self, input_ids, attention_mask, return_message_attention=False):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )
        
        sequence_output = outputs.last_hidden_state
        
        # Message-level attention scores
        message_attention_weights = None
        if return_message_attention and self.msg_token_id is not None:
            msg_positions = (input_ids == self.msg_token_id).float()
            
            if msg_positions.sum() > 0:
                attn_scores = self.attn_scoring(sequence_output).squeeze(-1)
                attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)
                
                msg_indices = msg_positions.nonzero(as_tuple=True)
                if len(msg_indices[0]) > 0:
                    message_scores = []
                    for batch_idx in range(input_ids.shape[0]):
                        batch_msg_mask = msg_positions[batch_idx] > 0
                        if batch_msg_mask.any():
                            batch_scores = attn_scores[batch_idx][batch_msg_mask]
                            batch_scores = F.softmax(batch_scores, dim=-1)
                            message_scores.append(batch_scores)
                    
                    if message_scores:
                        message_attention_weights = message_scores[0] if len(message_scores) == 1 else torch.cat(message_scores)
        
        # Global pooling for classification
        if self.use_attention_pooling:
            attention_scores = self.attention_weights(sequence_output).squeeze(-1)
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
            attention_weights = F.softmax(attention_scores, dim=-1)
            attention_weights = self.attention_dropout(attention_weights)
            
            pooled_output = torch.bmm(
                attention_weights.unsqueeze(1),
                sequence_output
            ).squeeze(1)
        else:
            pooled_output = sequence_output[:, 0]
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        if return_message_attention:
            return logits, outputs.attentions, message_attention_weights
        return logits, outputs.attentions

# ------------------------- FastAPI App ------------------------- #
app = FastAPI(title="Unified Drug Chat Detection API", version="3.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and tokenizers
models = {}
tokenizers = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configuration for different models
MODEL_CONFIGS = {
    "roberta-base": {
        "model_name": "klue/roberta-base",
        "model_path": "roberta_base/models_pure_lm/best_model.pt",
        "window_size": 10,
        "stride": 5,
        "has_msg_token": True  # Set based on actual training
    },
    "roberta-large": {
        "model_name": "klue/roberta-large",
        "model_path": "roberta_large/models_pure_lm/best_model.pt", 
        "window_size": 10,
        "stride": 5,
        "has_msg_token": True  # Set based on actual training
    },
    "bert-base": {
        "model_name": "klue/bert-base",
        "model_path": "bert_base/models_pure_lm_attn/best_model.pt",
        "window_size": 10,
        "stride": 5,
        "has_msg_token": True   # Only this one uses MSG tokens
    }
}

MAX_LENGTH = 512

def load_model(model_key: str):
    """Load a specific model and tokenizer"""
    config = MODEL_CONFIGS[model_key]
    model_name = config["model_name"]
    model_path = config["model_path"]
    has_msg_token = config["has_msg_token"]
    
    print(f"Loading {model_key} tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add special tokens
    special_tokens = {
        'additional_special_tokens': [
            '[SEP]', '[SPEAKER_0]', '[SPEAKER_1]', '[MASK]'
        ]
    }
    
    if has_msg_token:
        special_tokens['additional_special_tokens'].append('[MSG]')
    
    tokenizer.add_special_tokens(special_tokens)
    msg_token_id = tokenizer.convert_tokens_to_ids('[MSG]') if has_msg_token else None
    
    print(f"Loading {model_key} model...")
    model = ConversationAnomalyDetector(
        model_name,
        num_labels=2,
        hidden_dropout=0.3,
        attention_dropout=0.1,
        use_attention_pooling=True,
        msg_token_id=msg_token_id,
        attn_supervision_weight=0.5
    )
    
    # Resize token embeddings
    model.transformer.resize_token_embeddings(len(tokenizer))
    
    # Load weights
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                checkpoint = checkpoint['model_state_dict']
            
            missing, unexpected = model.load_state_dict(checkpoint, strict=False)
            if missing:
                print(f"⚠️  {model_key} missing keys (newly initialized): {missing}")
            if unexpected:
                print(f"⚠️  {model_key} unexpected keys (ignored): {unexpected}")
            print(f"{model_key} model loaded from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load {model_key} from {model_path}: {e}")
    else:
        print(f"Warning: Model file not found at {model_path} for {model_key}. Using untrained model.")
    
    model.to(device)
    model.eval()
    
    tokenizers[model_key] = tokenizer
    models[model_key] = model
    print(f"{model_key} model loaded on {device}")

def load_all_models():
    """Load all available models"""
    for model_key in MODEL_CONFIGS.keys():
        try:
            load_model(model_key)
        except Exception as e:
            print(f"Failed to load {model_key}: {e}")

def preprocess_messages(messages: List[Message], window_size: int, has_msg_token: bool = True) -> str:
    """Convert messages to model input format - Always use MSG token for attention-based scoring"""
    conversation_parts = []
    
    for i, msg in enumerate(messages[:window_size]):
        speaker = f"[SPEAKER_{msg.speaker if msg.speaker is not None else i % 2}]"
        if has_msg_token:
            conversation_parts.append(f"[MSG] {speaker} {msg.original_form}")
        else:
            conversation_parts.append(f"{speaker} {msg.original_form}")
    
    return " [SEP] ".join(conversation_parts)

def predict_window(messages: List[Message], model_key: str = "roberta-base", return_message_scores=False) -> Dict[str, Any]:
    """Predict anomaly for a window of messages"""
    if model_key not in models:
        raise HTTPException(status_code=400, detail=f"Model {model_key} not available")
    
    model = models[model_key]
    tokenizer = tokenizers[model_key]
    config = MODEL_CONFIGS[model_key]
    
    conversation_text = preprocess_messages(messages, config["window_size"], config["has_msg_token"])
    
    # Tokenize
    encoding = tokenizer(
        conversation_text,
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Predict with message-level attention if requested and model supports it
    with torch.no_grad():
        if config["has_msg_token"] and return_message_scores:
            logits, _, message_attention = model(
                input_ids, 
                attention_mask, 
                return_message_attention=True
            )
        else:
            logits, _ = model(input_ids, attention_mask)
            message_attention = None
        
        probs = F.softmax(logits, dim=-1)
        pred = torch.argmax(logits, dim=-1).item()
        confidence = probs[0, pred].item()
        anomaly_prob = probs[0, 1].item()
    
    result = {
        'prediction': pred,
        'confidence': confidence,
        'anomaly_probability': anomaly_prob,
        'model_used': model_key
    }
    
    if return_message_scores and message_attention is not None:
        message_scores = message_attention.cpu().numpy().tolist()
        result['message_scores'] = message_scores
    
    return result

def generate_conversation_id(session_id: Optional[str] = None) -> str:
    """Generate unique conversation ID using SHA-256"""
    timestamp = datetime.now().isoformat()
    base_string = f"{session_id or 'anonymous'}_{timestamp}"
    return hashlib.sha256(base_string.encode()).hexdigest()[:16].upper()

# ------------------------- API Endpoints ------------------------- #
@app.on_event("startup")
async def startup_event():
    """Load all models on startup"""
    load_all_models()

@app.get("/")
async def root():
    return {"message": "Unified Drug Chat Detection API v3.0 is running"}

@app.get("/api/models")
async def get_available_models():
    """Get list of available models"""
    available = []
    model_info = {}
    for model_key in MODEL_CONFIGS.keys():
        if model_key in models:
            config = MODEL_CONFIGS[model_key]
            model_data = {
                "key": model_key,
                "name": config["model_name"],
                "window_size": config["window_size"],
                "stride": config["stride"],
                "has_msg_token": config["has_msg_token"]
            }
            available.append(model_data)
            
            # Create display info for frontend
            model_info[model_key] = f"{config['model_name']} 모델: 윈도우 크기 {config['window_size']}, 스트라이드 {config['stride']}, 메시지 단위 주의도 분석"
    
    return {"models": available, "model_info": model_info}

@app.post("/api/analyze-window", response_model=WindowAnalysisResponse)
async def analyze_window(request: WindowAnalysisRequest):
    """Analyze a single window and return which messages are anomalous"""
    model_key = request.model_name or "roberta-base"
    
    if model_key not in models:
        raise HTTPException(status_code=400, detail=f"Model {model_key} not available")
    
    # Predict for the window with message scores
    result = predict_window(request.messages, model_key, return_message_scores=True)
    
    # Determine which messages are likely anomalous based on attention scores
    anomaly_indices = []
    message_scores = {}
    
    if result['prediction'] == 1 and 'message_scores' in result and request.message_indices:
        # Use attention scores to identify anomalous messages
        scores = result['message_scores']
        
        if scores:
            total_score = sum(scores)
            if total_score > 0:
                normalized_scores = [s / total_score for s in scores]
                
                # Consider messages with above-average attention as potentially anomalous
                avg_score = 1.0 / len(scores)
                threshold = avg_score * 1.5
                
                for i, (score, idx) in enumerate(zip(normalized_scores, request.message_indices)):
                    if idx >= 0:
                        message_scores[idx] = score
                        if score > threshold and request.messages[i].original_form.strip():
                            anomaly_indices.append(idx)
    
    return WindowAnalysisResponse(
        anomaly_indices=anomaly_indices,
        message_scores=message_scores,
        **{k: v for k, v in result.items() if k != 'message_scores'}
    )

@app.post("/api/analyze-chat", response_model=AnalysisResponse)
async def analyze_chat(request: ChatAnalysisRequest):
    """Analyze a chat window for anomalies"""
    model_key = request.model_name or "roberta-base"
    
    if model_key not in models:
        raise HTTPException(status_code=400, detail=f"Model {model_key} not available")
    
    if len(request.messages) < 3:
        raise HTTPException(status_code=400, detail="At least 3 messages required")
    
    config = MODEL_CONFIGS[model_key]
    window_size = config["window_size"]
    
    # Pad messages if needed
    messages = list(request.messages)
    while len(messages) < window_size:
        messages.append(Message(original_form="", speaker=len(messages) % 2))
    
    # Predict for the window
    result = predict_window(messages[:window_size], model_key, return_message_scores=True)
    
    return AnalysisResponse(**result)

@app.post("/api/analyze-conversation", response_model=AnalysisResponse)
async def analyze_conversation(request: ConversationAnalysisRequest):
    """Analyze a full conversation using sliding windows with V1-style attention-based message scoring"""
    model_key = request.model_name or "roberta-base"
    
    if model_key not in models:
        raise HTTPException(status_code=400, detail=f"Model {model_key} not available")

    utterances = request.conversation.get("utterance", request.conversation.get("utterances", []))
    if not utterances:
        raise HTTPException(status_code=400, detail="At least 1 utterance required")

    config = MODEL_CONFIGS[model_key]
    window_size = config["window_size"]
    stride = config["stride"]

    # 1) Build Message objects and pad to window_size
    messages = [
        Message(original_form=utt.get("original_form", ""), speaker=utt.get("speaker", i % 2))
        for i, utt in enumerate(utterances)
    ]
    while len(messages) < window_size:
        messages.append(Message(original_form="", speaker=len(messages) % 2))

    # 2) Compute all slide-window start positions (fix overlap issue)
    starts = list(range(0, len(messages) - window_size + 1, stride))
    # Don't add tail if it creates too much overlap
    tail = max(0, len(messages) - window_size)
    if tail not in starts and tail > starts[-1] + stride // 2:
        starts.append(tail)

    windows = []
    message_anomaly_scores = {}   # idx -> accumulated score
    message_window_counts = {}    # idx -> number of windows seen

    # 3) For each window: predict & accumulate using V1-style attention scoring
    for start_idx in starts:
        end_idx = start_idx + window_size
        window_msgs = messages[start_idx:end_idx]

        # map local positions to global utterance indices (or -1 for padding)
        msg_indices = [
            i if i < len(utterances) else -1
            for i in range(start_idx, end_idx)
        ]

        result = predict_window(window_msgs, model_key, return_message_scores=True)

        # store window-level prediction
        windows.append(WindowPrediction(
            start=start_idx,
            end=end_idx,
            prediction=result["prediction"],
            confidence=result["confidence"],
            anomaly_probability=result["anomaly_probability"],
            message_indices=msg_indices,
            message_anomaly_scores=result.get("message_scores", [])
        ))

        # V1-style attention-based scoring: accumulate weighted scores
        scores = result.get("message_scores")
        if scores and config["has_msg_token"]:
            # Use attention-based scoring for models with MSG tokens
            total = sum(scores)
            if total > 0:
                normalized = [s / total for s in scores]
                alpha = 0.3  # Same as V1
                for local_pos, global_idx in enumerate(msg_indices):
                    if global_idx >= 0:
                        weight = alpha * normalized[local_pos] + (1 - alpha) * result["anomaly_probability"]
                        message_anomaly_scores.setdefault(global_idx, 0.0)
                        message_window_counts.setdefault(global_idx, 0)
                        message_anomaly_scores[global_idx] += weight
                        message_window_counts[global_idx] += 1
        else:
            # For models without MSG tokens, use anomaly probability weighting
            # Accumulate scores for all windows (no threshold filtering)
            for local_pos, global_idx in enumerate(msg_indices):
                if global_idx >= 0 and utterances[global_idx].get("original_form", "").strip():
                    # Weight by position in window and anomaly probability
                    center_pos = window_size // 2
                    distance_weight = 1.0 - abs(local_pos - center_pos) / center_pos
                    weight = distance_weight * result["anomaly_probability"]
                    
                    message_anomaly_scores.setdefault(global_idx, 0.0)
                    message_window_counts.setdefault(global_idx, 0)
                    message_anomaly_scores[global_idx] += weight
                    message_window_counts[global_idx] += 1

    # 4) Compute per-message average scores
    message_anomaly_map = {}
    for idx, total_score in message_anomaly_scores.items():
        count = message_window_counts.get(idx, 1)
        message_anomaly_map[idx] = total_score / count

    # 5) V1-style dynamic threshold: mean + 1.0 * std over all messages
    anomaly_indices = []
    if message_anomaly_map:
        scores_arr = np.array(list(message_anomaly_map.values()))
        mean, std = scores_arr.mean(), scores_arr.std()
        
        # Dynamic threshold based on score distribution
        if std > 0.01:  # If there's meaningful variation in scores
            thresh = max(mean + 1.0 * std, 0.1)
        else:
            # If scores are very uniform, use percentile-based threshold
            thresh = np.percentile(scores_arr, 80)  # Top 20% of scores
        
        anomaly_indices = [
            idx for idx, sc in message_anomaly_map.items()
            if sc > thresh and utterances[idx].get("original_form", "").strip()
        ]

    # 6) Overall stats
    anomaly_windows = sum(w.prediction == 1 for w in windows)
    total_windows = len(windows)
    normal_windows = total_windows - anomaly_windows
    avg_anomaly_prob = sum(w.anomaly_probability for w in windows) / total_windows if total_windows else 0.0
    overall_prediction = 1 if anomaly_indices else 0
    best_confidence = max((w.confidence for w in windows), default=0.0)

    return AnalysisResponse(
        prediction=overall_prediction,
        confidence=best_confidence,
        anomaly_probability=avg_anomaly_prob,
        model_used=model_key,
        windows=windows,
        anomaly_windows=anomaly_windows,
        normal_windows=normal_windows,
        total_windows=total_windows,
        detailed_results=[{
            "start": w.start,
            "end": w.end,
            "prediction": w.prediction,
            "confidence": w.confidence,
            "anomaly_probability": w.anomaly_probability,
            "message_indices": w.message_indices,
            "message_anomaly_scores": w.message_anomaly_scores
        } for w in windows],
        message_anomaly_map=message_anomaly_map
    )

@app.post("/api/analyze-file", response_model=FileAnalysisResponse)
async def analyze_file(file: UploadFile = File(...), model_name: str = "roberta-base"):
    """Analyze uploaded JSON file using model-appropriate strategy"""
    if model_name not in models:
        raise HTTPException(status_code=400, detail=f"Model {model_name} not available")
    
    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Only JSON files are supported")

    content = await file.read()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file")

    if not isinstance(data, list):
        data = [data]

    results = []
    anomaly_conversations = 0
    config = MODEL_CONFIGS[model_name]
    window_size = config["window_size"]
    stride = config["stride"]
    has_msg_token = config["has_msg_token"]

    for conv in data:
        conv_id = conv.get("id") or f"CONV_{generate_conversation_id()}"
        utterances = conv.get("utterance", conv.get("utterances", []))
        if not utterances:
            continue

        # Choose strategy based on model type
        if has_msg_token:
            # BERT-style: Use V1 sliding window approach with attention
            anomaly_indices = analyze_with_attention(utterances, model_name, window_size, stride)
        else:
            # RoBERTa-style: Use per-message centered windows (like original V1)
            anomaly_indices = analyze_per_message_windows(utterances, model_name, window_size)
        
        logger.info(f"[DEBUG][{conv_id}] flagged_indices={anomaly_indices}")

        has_anomaly = bool(anomaly_indices)
        if has_anomaly:
            anomaly_conversations += 1

        # Annotate each utterance
        for i, utt in enumerate(utterances):
            utt["predicted_label"] = 1 if i in anomaly_indices else 0

        results.append({
            "id": conv_id,
            "has_anomaly": has_anomaly,
            "utterances": utterances,
            "anomaly_indices": anomaly_indices,
            "anomaly_scores": {}  # Simplified for now
        })

    total = len(results)
    detection_rate = (anomaly_conversations / total * 100) if total else 0.0

    return FileAnalysisResponse(
        total_conversations=total,
        anomaly_conversations=anomaly_conversations,
        normal_conversations=total - anomaly_conversations,
        detection_rate=detection_rate,
        model_used=model_name,
        conversations=results
    )

def analyze_with_attention(utterances, model_name, window_size, stride):
    """V1-style sliding window analysis with attention for BERT models"""
    # Build Message list and pad
    messages = [
        Message(original_form=utt.get("original_form", ""), speaker=utt.get("speaker", i % 2))
        for i, utt in enumerate(utterances)
    ]
    while len(messages) < window_size:
        messages.append(Message(original_form="", speaker=len(messages) % 2))

    # Sliding windows
    starts = list(range(0, len(messages) - window_size + 1, stride))
    tail = max(0, len(messages) - window_size)
    if tail not in starts and tail > starts[-1] + stride // 2:
        starts.append(tail)

    message_anomaly_scores = {}
    message_window_counts = {}

    for start_idx in starts:
        end_idx = start_idx + window_size
        window_msgs = messages[start_idx:end_idx]
        
        msg_indices = [i if i < len(utterances) else -1 for i in range(start_idx, end_idx)]
        result = predict_window(window_msgs, model_name, return_message_scores=True)

        # Use attention-based scoring
        scores = result.get("message_scores", [])
        if scores:
            total = sum(scores)
            if total > 0:
                normalized = [s / total for s in scores]
                alpha = 0.3
                for local_pos, global_idx in enumerate(msg_indices):
                    if global_idx >= 0:
                        weight = alpha * normalized[local_pos] + (1 - alpha) * result["anomaly_probability"]
                        message_anomaly_scores.setdefault(global_idx, 0.0)
                        message_window_counts.setdefault(global_idx, 0)
                        message_anomaly_scores[global_idx] += weight
                        message_window_counts[global_idx] += 1

    # Compute average and threshold
    message_anomaly_map = {}
    for idx, total_score in message_anomaly_scores.items():
        count = message_window_counts.get(idx, 1)
        message_anomaly_map[idx] = total_score / count

    if message_anomaly_map:
        scores_arr = np.array(list(message_anomaly_map.values()))
        mean, std = scores_arr.mean(), scores_arr.std()
        thresh = max(mean + 1.0 * std, 0.1)
        
        return [
            idx for idx, sc in message_anomaly_map.items()
            if sc > thresh and utterances[idx].get("original_form", "").strip()
        ]
    return []

def analyze_per_message_windows(utterances, model_name, window_size):
    """V1-style per-message window analysis for RoBERTa models"""
    anomaly_indices = []
    
    for idx in range(len(utterances)):
        # Build centered window around each message
        half = window_size // 2
        start = max(0, idx - half)
        end = min(len(utterances), start + window_size)
        
        if end - start < window_size:
            start = max(0, end - window_size)
        
        window_msgs = []
        for i in range(start, end):
            if i < len(utterances):
                utt = utterances[i]
                window_msgs.append(Message(
                    original_form=utt.get('original_form', ''),
                    speaker=utt.get('speaker', i % 2)
                ))
        
        # Pad if needed
        while len(window_msgs) < window_size:
            window_msgs.append(Message(original_form="", speaker=len(window_msgs) % 2))
        
        result = predict_window(window_msgs, model_name, return_message_scores=False)
        
        # If window is anomalous and center message is non-empty, flag it
        if (result["prediction"] == 1 and 
            result["anomaly_probability"] > 0.3 and  # Add threshold back
            utterances[idx].get("original_form", "").strip()):
            anomaly_indices.append(idx)
    
    return anomaly_indices

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": list(models.keys()),
        "device": str(device),
        "version": "3.0.0"
    }

# ------------------------- Run the server ------------------------- #
if __name__ == "__main__":
    # Create necessary directories
    for model_key, config in MODEL_CONFIGS.items():
        model_dir = os.path.dirname(config["model_path"])
        os.makedirs(model_dir, exist_ok=True)
    
    # Run the server
    uvicorn.run(
        "unified_demo:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )