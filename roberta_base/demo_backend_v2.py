import os
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
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
#save log into log_demo_backend_v2.log
log_file = Path("log_demo_backend_v2.log")
# ------------------------- Request/Response Models ------------------------- #
class Message(BaseModel):
    original_form: str
    speaker: Optional[int] = 0
    label: Optional[int] = None

class ChatAnalysisRequest(BaseModel):
    messages: List[Message]

class WindowAnalysisRequest(BaseModel):
    messages: List[Message]
    message_indices: Optional[List[int]] = None

class ConversationAnalysisRequest(BaseModel):
    conversation: Dict[str, Any]

class WindowPrediction(BaseModel):
    start: int
    end: int
    prediction: int
    confidence: float
    anomaly_probability: float
    message_indices: Optional[List[int]] = None
    message_anomaly_scores: Optional[List[float]] = None  # New: per-message scores

class AnalysisResponse(BaseModel):
    prediction: int
    confidence: float
    anomaly_probability: float
    windows: Optional[List[WindowPrediction]] = None
    anomaly_windows: Optional[int] = None
    normal_windows: Optional[int] = None
    total_windows: Optional[int] = None
    detailed_results: Optional[List[Dict[str, Any]]] = None
    message_anomaly_map: Optional[Dict[int, float]] = None  # New: map of message index to anomaly score

class WindowAnalysisResponse(BaseModel):
    prediction: int
    confidence: float
    anomaly_probability: float
    anomaly_indices: List[int]
    message_scores: Optional[Dict[int, float]] = None  # New: individual message scores

class FileAnalysisResponse(BaseModel):
    total_conversations: int
    anomaly_conversations: int
    normal_conversations: int
    detection_rate: float
    conversations: List[Dict[str, Any]]

# ------------------------- Model Architecture (Updated with Attention) ------------------------- #
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
        
        # Message-level attention scoring layer (matching training script)
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
            # Find [MSG] token positions
            msg_positions = (input_ids == self.msg_token_id).float()
            
            if msg_positions.sum() > 0:
                # Get attention scores for all positions
                attn_scores = self.attn_scoring(sequence_output).squeeze(-1)  # [B, L]
                attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)
                
                # Extract scores at message positions
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
app = FastAPI(title="Drug Chat Detection API", version="2.0.0")



# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and tokenizer
model = None
tokenizer = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
msg_token_id = None

# Configuration
MODEL_PATH = "models_pure_lm_attn/best_model.pt"  # Updated path
MODEL_NAME = "klue/roberta-base"  # Updated model name
WINDOW_SIZE = 10  # Updated from 2 to 10
STRIDE = 3  # Updated from 2 to 5
MAX_LENGTH = 512

def load_model():
    """Load the trained model and tokenizer"""
    global model, tokenizer, msg_token_id
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Add special tokens
    special_tokens = {
        'additional_special_tokens': [
            '[SEP]', '[SPEAKER_0]', '[SPEAKER_1]', '[MASK]', '[MSG]'
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    msg_token_id = tokenizer.convert_tokens_to_ids('[MSG]')
    
    print("Loading model...")
    model = ConversationAnomalyDetector(
        MODEL_NAME,
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
    if os.path.exists(MODEL_PATH):
        # 1) load the raw checkpoint
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        # 2) if you accidentally pointed at a full-training-checkpoint, unwrap it:
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            checkpoint = checkpoint['model_state_dict']
        # 3) load with strict=False so missing/new layers are initialized
        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
        if missing:
            print(f"⚠️  Missing keys (newly initialized): {missing}")
        if unexpected:
            print(f"⚠️  Unexpected keys (ignored):     {unexpected}")
        print(f"Model loaded from {MODEL_PATH}")

    else:
        print(f"Warning: Model file not found at {MODEL_PATH}. Using untrained model.")
    
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}")

def preprocess_messages(messages: List[Message], window_size: int = WINDOW_SIZE) -> str:
    """Convert messages to model input format with [MSG] tokens"""
    conversation_parts = []
    
    for i, msg in enumerate(messages[:window_size]):
        speaker = f"[SPEAKER_{msg.speaker if msg.speaker is not None else i % 2}]"
        # Add [MSG] token at the beginning of each message
        conversation_parts.append(f"[MSG] {speaker} {msg.original_form}")
    
    return " [SEP] ".join(conversation_parts)

def extract_message_scores(input_ids, attention_weights, msg_token_id):
    """Extract attention scores for each message position"""
    # This function is no longer needed as the model directly returns message scores
    pass

def build_padded_window(utterances: List[Dict[str, Any]], center_idx: int) -> List[Message]:
    """Build a padded window centered around a specific index"""
    half = WINDOW_SIZE // 2
    start = max(0, center_idx - half)
    end = min(len(utterances), start + WINDOW_SIZE)
    
    # Adjust start if we're at the end
    if end - start < WINDOW_SIZE:
        start = max(0, end - WINDOW_SIZE)
    
    msgs = []
    for i in range(start, end):
        utt = utterances[i]
        msgs.append(Message(
            original_form=utt.get('original_form', ''),
            speaker=utt.get('speaker', i % 2)
        ))
    
    # Pad with empty messages if needed
    while len(msgs) < WINDOW_SIZE:
        msgs.append(Message(original_form="", speaker=len(msgs) % 2))
    
    return msgs

def predict_window(messages: List[Message], return_message_scores=False) -> Dict[str, Any]:
    """Predict anomaly for a window of messages with message-level scores"""
    conversation_text = preprocess_messages(messages)
    
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
    
    # Predict with message-level attention
    with torch.no_grad():
        logits, _, message_attention = model(
            input_ids, 
            attention_mask, 
            return_message_attention=True
        )
        probs = F.softmax(logits, dim=-1)
        print(f"[DEBUG] Conversation: {conversation_text}, probs: {probs.cpu().numpy()}, message_attention: {message_attention.cpu().numpy() if message_attention is not None else None}")
        pred = torch.argmax(logits, dim=-1).item()
        confidence = probs[0, pred].item()
        anomaly_prob = probs[0, 1].item()
    
    result = {
        'prediction': pred,
        'confidence': confidence,
        'anomaly_probability': anomaly_prob
    }
    
    if return_message_scores and message_attention is not None:
        # Convert tensor to list of scores
        message_scores = message_attention.cpu().numpy().tolist()
        result['message_scores'] = message_scores
    
    return result

def generate_conversation_id(session_id: Optional[str] = None) -> str:
    """Generate unique conversation ID using SHA-256"""
    timestamp = datetime.now().isoformat()
    base_string = f"{session_id or 'anonymous'}_{timestamp}"
    return hashlib.sha256(base_string.encode()).hexdigest()[:16].upper()

def mask_labels(data: Any) -> Any:
    """Recursively mask all label fields in the data"""
    if isinstance(data, dict):
        return {k: mask_labels(v) if k != 'label' else None for k, v in data.items()}
    elif isinstance(data, list):
        return [mask_labels(item) for item in data]
    else:
        return data

# ------------------------- API Endpoints ------------------------- #
@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/")
async def root():
    return {"message": "Drug Chat Detection API v2.0 is running"}

@app.post("/api/analyze-window", response_model=WindowAnalysisResponse)
async def analyze_window(request: WindowAnalysisRequest):
    """Analyze a single window and return which messages are anomalous"""
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Predict for the window with message scores
    result = predict_window(request.messages, return_message_scores=True)
    
    # Determine which messages are likely anomalous based on attention scores
    anomaly_indices = []
    message_scores = {}
    
    if result['prediction'] == 1 and 'message_scores' in result and request.message_indices:
        # Use attention scores to identify anomalous messages
        scores = result['message_scores']
        
        # Normalize scores
        if scores:
            total_score = sum(scores)
            if total_score > 0:
                normalized_scores = [s / total_score for s in scores]
                
                # Consider messages with above-average attention as potentially anomalous
                avg_score = 1.0 / len(scores)
                threshold = avg_score * 1.5  # 1.5x average attention
                
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
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(request.messages) < 3:
        raise HTTPException(status_code=400, detail="At least 3 messages required")
    
    # Pad messages if needed
    messages = list(request.messages)
    while len(messages) < WINDOW_SIZE:
        messages.append(Message(original_form="", speaker=len(messages) % 2))
    
    # Predict for the window
    result = predict_window(messages[:WINDOW_SIZE], return_message_scores=True)
    
    return AnalysisResponse(**result)

@app.post("/api/analyze-conversation", response_model=AnalysisResponse)
async def analyze_conversation(request: ConversationAnalysisRequest):
    """Analyze a full conversation using sliding windows with message-level detection."""
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    utterances = request.conversation.get("utterance", request.conversation.get("utterances", []))
    if not utterances:
        raise HTTPException(status_code=400, detail="At least 1 utterance required")

    # 1) Build Message objects and pad to WINDOW_SIZE
    messages = [
        Message(original_form=utt.get("original_form", ""), speaker=utt.get("speaker", i % 2))
        for i, utt in enumerate(utterances)
    ]
    while len(messages) < WINDOW_SIZE:
        messages.append(Message(original_form="", speaker=len(messages) % 2))

    # 2) Compute all slide‐window start positions (including final “tail”)
    starts = list(range(0, len(messages) - WINDOW_SIZE + 1, STRIDE))
    tail = max(0, len(messages) - WINDOW_SIZE)
    if tail not in starts:
        starts.append(tail)

    windows = []
    message_anomaly_scores = {}   # idx -> accumulated score
    message_window_counts = {}    # idx -> number of windows seen

    # 3) For each window: predict & accumulate
    for start_idx in starts:
        end_idx = start_idx + WINDOW_SIZE
        window_msgs = messages[start_idx:end_idx]

        # map local positions to global utterance indices (or -1 for padding)
        msg_indices = [
            i if i < len(utterances) else -1
            for i in range(start_idx, end_idx)
        ]

        result = predict_window(window_msgs, return_message_scores=True)

        # store window‐level prediction
        windows.append(WindowPrediction(
            start=start_idx,
            end=end_idx,
            prediction=result["prediction"],
            confidence=result["confidence"],
            anomaly_probability=result["anomaly_probability"],
            message_indices=msg_indices,
            message_anomaly_scores=result.get("message_scores", [])
        ))

        # if we have per‐message attention scores, normalize & weight by anomaly_probability
        scores = result.get("message_scores")
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

    # 4) Compute per‐message average scores
    message_anomaly_map = {}
    for idx, total_score in message_anomaly_scores.items():
        count = message_window_counts.get(idx, 1)
        message_anomaly_map[idx] = total_score / count

    # 5) Dynamic threshold: mean + 1.0 * std over all messages
    if message_anomaly_map:
        avg_anom_prob = sum(w.anomaly_probability for w in windows) / len(windows)
        if avg_anom_prob < 0.1:   # tune this to your ROC curve
            anomaly_indices = []
        else:
            # apply your threshold + min_votes logic
            import numpy as np
            scores_arr = np.array(list(message_anomaly_map.values()))
            mean, std = scores_arr.mean(), scores_arr.std()
            thresh = mean + 1.0 * std
            abs_floor = 0.1
            thresh = max(mean + 1.0 * std, abs_floor)

            anomaly_indices = [
                idx for idx, sc in message_anomaly_map.items()
                if sc > thresh and utterances[idx].get("original_form", "").strip()
            ]
    else:
        anomaly_indices = []

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
async def analyze_file(file: UploadFile = File(...)):
    """Analyze uploaded JSON file with improved message-level detection."""
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
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

    for conv in data:
        conv_id = conv.get("id") or f"CONV_{generate_conversation_id()}"
        utterances = conv.get("utterance", conv.get("utterances", []))
        if not utterances:
            continue

        # 1) Build Message list and pad
        messages = [
            Message(original_form=utt.get("original_form", ""), speaker=utt.get("speaker", i % 2))
            for i, utt in enumerate(utterances)
        ]
        while len(messages) < WINDOW_SIZE:
            messages.append(Message(original_form="", speaker=len(messages) % 2))

        # 2) Compute sliding‐window start positions
        starts = list(range(0, len(messages) - WINDOW_SIZE + 1, STRIDE))
        tail = max(0, len(messages) - WINDOW_SIZE)
        if tail not in starts:
            starts.append(tail)

        # 3) Accumulate per‐message anomaly scores
        message_anomaly_scores = {}
        message_window_counts = {}

        for start_idx in starts:
            end_idx = start_idx + WINDOW_SIZE
            window_msgs = messages[start_idx:end_idx]
            # Map to global indices
            msg_indices = [i if i < len(utterances) else -1
                           for i in range(start_idx, end_idx)]

            result = predict_window(window_msgs, return_message_scores=True)

            scores = result.get("message_scores", [])
            if scores:
            # if result["anomaly_probability"] > 0.05 and scores:
            # normalize & accumulate as before…
                total = sum(scores)
                if total > 0:
                    normalized = [s / total for s in scores]
                    alpha = 0.5
                    for local_pos, global_idx in enumerate(msg_indices):
                        if global_idx >= 0:
                            weight = alpha * normalized[local_pos] + \
                                     (1 - alpha) * result["anomaly_probability"]
                            message_anomaly_scores.setdefault(global_idx, 0.0)
                            message_window_counts.setdefault(global_idx, 0)
                            message_anomaly_scores[global_idx] += weight
                            message_window_counts[global_idx] += 1

                            print(normalized, normalized[local_pos], result["anomaly_probability"], weight, global_idx, utterances[global_idx].get("original_form", ""))

        # 4) Compute average per‐message scores
        message_anomaly_map = {}
        for idx, total_score in message_anomaly_scores.items():
            count = message_window_counts.get(idx, 1)
            print(f"[DEBUG][{conv_id}] idx={idx} total_score={total_score} count={count}")
            message_anomaly_map[idx] = total_score / count

        logger.info(f"[DEBUG][{conv_id}] raw_scores: {message_anomaly_map}")

        # 5) Dynamic thresholding (mean + 1.0 * std)
        if message_anomaly_map:
            scores_arr = np.array(list(message_anomaly_map.values()))
            mean, std = scores_arr.mean(), scores_arr.std()
            thresh = mean + 1.0 * std
            logger.info(f"[DEBUG][{conv_id}] mean={mean:.4f} std={std:.4f} thresh={thresh:.4f}")
            # percentile = 95
            # thresh = np.percentile(scores_arr, percentile)
            abs_floor = 0.1
            thresh = max(mean + 1.0 * std, abs_floor)
            anomaly_indices = [
                idx for idx, sc in message_anomaly_map.items()
                if sc > thresh and utterances[idx].get("original_form", "").strip()
            ]
            # min_votes = 2
            # anomaly_indices = [
            #     idx
            #     for idx, sc in message_anomaly_map.items()
            #     if sc > thresh and message_window_counts.get(idx, 0) >= min_votes
            # ]
        else:
            anomaly_indices = []

        logger.info(f"[DEBUG][{conv_id}] flagged_indices={anomaly_indices}")

        has_anomaly = bool(anomaly_indices)
        if has_anomaly:
            anomaly_conversations += 1

        # 6) Annotate each utterance
        for i, utt in enumerate(utterances):
            utt["predicted_label"] = 1 if i in anomaly_indices else 0

        results.append({
            "id": conv_id,
            "has_anomaly": has_anomaly,
            "utterances": utterances,
            "anomaly_indices": anomaly_indices,
            "anomaly_scores": {
                str(idx): message_anomaly_map[idx]
                for idx in message_anomaly_map
            }
        })

    total_conversations = len(results)
    detection_rate = (anomaly_conversations / total_conversations * 100) \
                     if total_conversations else 0.0

    return FileAnalysisResponse(
        total_conversations=total_conversations,
        anomaly_conversations=anomaly_conversations,
        normal_conversations=total_conversations - anomaly_conversations,
        detection_rate=detection_rate,
        conversations=results
    )


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "version": "2.0.0"
    }

# ------------------------- Run the server ------------------------- #
if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("models_pure_lm_attn", exist_ok=True)
    
    # Run the server
    uvicorn.run(
        "demo_backend_v2:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )