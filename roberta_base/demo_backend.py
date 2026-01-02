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

class AnalysisResponse(BaseModel):
    prediction: int
    confidence: float
    anomaly_probability: float
    windows: Optional[List[WindowPrediction]] = None
    anomaly_windows: Optional[int] = None
    normal_windows: Optional[int] = None
    total_windows: Optional[int] = None
    detailed_results: Optional[List[Dict[str, Any]]] = None

class WindowAnalysisResponse(BaseModel):
    prediction: int
    confidence: float
    anomaly_probability: float
    anomaly_indices: List[int]  # Indices of messages detected as anomalous

class FileAnalysisResponse(BaseModel):
    total_conversations: int
    anomaly_conversations: int
    normal_conversations: int
    detection_rate: float
    conversations: List[Dict[str, Any]]

# ------------------------- Model Architecture (from training code) ------------------------- #
class ConversationAnomalyDetector(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int = 2,
        hidden_dropout: float = 0.3,
        attention_dropout: float = 0.1,
        use_attention_pooling: bool = True
    ):
        super().__init__()
        
        self.transformer = AutoModel.from_pretrained(model_name)
        self.config = self.transformer.config
        hidden_size = self.config.hidden_size
        
        self.dropout = nn.Dropout(hidden_dropout)
        
        self.use_attention_pooling = use_attention_pooling
        if use_attention_pooling:
            self.attention_weights = nn.Linear(hidden_size, 1)
            self.attention_dropout = nn.Dropout(attention_dropout)
        
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
    
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )
        
        sequence_output = outputs.last_hidden_state
        
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
        
        return logits, outputs.attentions

# ------------------------- FastAPI App ------------------------- #
app = FastAPI(title="Drug Chat Detection API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and tokenizer
model = None
tokenizer = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configuration
MODEL_PATH = "models_pure_lm/best_model.pt"  # Path to your trained model
MODEL_NAME = "klue/roberta-base"
WINDOW_SIZE = 2
STRIDE = 2  # Changed from 5 to 1 as per user request
MAX_LENGTH = 512

def load_model():
    """Load the trained model and tokenizer"""
    global model, tokenizer
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Add special tokens
    special_tokens = {
        'additional_special_tokens': [
            '[SEP]', '[SPEAKER_0]', '[SPEAKER_1]', '[MASK]'
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    
    print("Loading model...")
    model = ConversationAnomalyDetector(
        MODEL_NAME,
        num_labels=2,
        hidden_dropout=0.3,
        attention_dropout=0.1,
        use_attention_pooling=True
    )
    
    # Resize token embeddings
    model.transformer.resize_token_embeddings(len(tokenizer))
    
    # Load weights
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}. Using untrained model.")
    
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}")

def preprocess_messages(messages: List[Message], window_size: int = WINDOW_SIZE) -> str:
    """Convert messages to model input format"""
    conversation_parts = []
    
    for i, msg in enumerate(messages[:window_size]):
        speaker = f"[SPEAKER_{msg.speaker if msg.speaker is not None else i % 2}]"
        conversation_parts.append(f"{speaker} {msg.original_form}")
    
    return " [SEP] ".join(conversation_parts)

def build_padded_window(utterances: List[Dict[str, Any]], center_idx: int) -> List[Message]:
    """
    Take utterances list and a center index, return exactly WINDOW_SIZE Message objects,
    padding with empty messages before/after as needed.
    """
    half = WINDOW_SIZE // 2
    start = max(0, center_idx - half)
    end = min(len(utterances), start + WINDOW_SIZE)
    # ensure we have WINDOW_SIZE by padding
    msgs = []
    for i in range(start, end):
        utt = utterances[i]
        msgs.append(Message(original_form=utt.get('original_form', ''),
                            speaker=utt.get('speaker', i % 2)))
    # pad at end
    while len(msgs) < WINDOW_SIZE:
        msgs.append(Message(original_form="", speaker=len(msgs) % 2))
    return msgs

def predict_window(messages: List[Message]) -> Dict[str, Any]:
    """Predict anomaly for a window of messages"""
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
    
    # Predict
    with torch.no_grad():
        logits, _ = model(input_ids, attention_mask)
        probs = F.softmax(logits, dim=-1)
        pred = torch.argmax(logits, dim=-1).item()
        confidence = probs[0, pred].item()
        anomaly_prob = probs[0, 1].item()
    
    return {
        'prediction': pred,
        'confidence': confidence,
        'anomaly_probability': anomaly_prob
    }

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
    return {"message": "Drug Chat Detection API is running"}

@app.post("/api/analyze-window", response_model=WindowAnalysisResponse)
async def analyze_window(request: WindowAnalysisRequest):
    """Analyze a single window and return which messages are anomalous"""
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Predict for the window
    result = predict_window(request.messages)
    
    # Determine which messages in the window are likely anomalous
    anomaly_indices = []
    if result['prediction'] == 1 and request.message_indices:
        # Simple heuristic: mark non-empty messages in anomalous windows
        for i, (msg, idx) in enumerate(zip(request.messages, request.message_indices)):
            if msg.original_form.strip() and idx >= 0:
                anomaly_indices.append(idx)
    
    return WindowAnalysisResponse(
        anomaly_indices=anomaly_indices,
        **result
    )

@app.post("/api/analyze-chat", response_model=AnalysisResponse)
async def analyze_chat(request: ChatAnalysisRequest):
    """Analyze a chat window for anomalies"""
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(request.messages) < 3:
        raise HTTPException(status_code=400, detail="At least 3 messages required")
    
    # Predict for the window
    result = predict_window(request.messages)
    
    return AnalysisResponse(**result)

@app.post("/api/analyze-conversation", response_model=AnalysisResponse)
async def analyze_conversation(request: ConversationAnalysisRequest):
    """Analyze a full conversation using sliding windows"""
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    conversation = request.conversation
    utterances = conversation.get('utterances', [])
    
    if len(utterances) == 0:
        raise HTTPException(status_code=400, detail="At least 1 utterance required")
    
    # Convert to Message objects
    messages = [
        Message(
            original_form=utt.get('original_form', utt.get('text', '')),
            speaker=utt.get('speaker', i % 2)
        )
        for i, utt in enumerate(utterances)
    ]
    
    # Pad with empty messages if needed
    if len(messages) < WINDOW_SIZE:
        padding_needed = WINDOW_SIZE - len(messages)
        for i in range(padding_needed):
            messages.append(Message(
                original_form="",
                speaker=(len(messages) + i) % 2
            ))
    
    # Sliding window analysis
    windows = []
    anomaly_count = 0
    detailed_results = []
    message_anomaly_counts = {}  # Track how many times each message appears in anomalous windows
    
    for start_idx in range(0, len(messages) - WINDOW_SIZE + 1, STRIDE):
        end_idx = min(start_idx + WINDOW_SIZE, len(messages))
        window_messages = messages[start_idx:end_idx]
        
        # Track original message indices
        message_indices = []
        for i in range(start_idx, end_idx):
            if i < len(utterances):  # Only track real messages, not padding
                message_indices.append(i)
            else:
                message_indices.append(-1)
        
        result = predict_window(window_messages)
        
        window_pred = WindowPrediction(
            start=start_idx,
            end=end_idx,
            message_indices=message_indices,
            **result
        )
        windows.append(window_pred)
        
        # Track anomalies
        if result['prediction'] == 1:
            anomaly_count += 1
            # Count each real message in anomalous windows
            for idx in message_indices:
                if idx >= 0:
                    message_anomaly_counts[idx] = message_anomaly_counts.get(idx, 0) + 1
        
        detailed_results.append({
            'start': start_idx,
            'end': end_idx,
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'anomaly_probability': result['anomaly_probability'],
            'message_indices': message_indices
        })
    
    # Overall prediction (1 if any window is anomalous)
    overall_prediction = 1 if anomaly_count > 0 else 0
    avg_anomaly_prob = sum(w.anomaly_probability for w in windows) / len(windows) if windows else 0
    
    return AnalysisResponse(
        prediction=overall_prediction,
        confidence=max(w.confidence for w in windows) if windows else 0,
        anomaly_probability=avg_anomaly_prob,
        windows=windows,
        anomaly_windows=anomaly_count,
        normal_windows=len(windows) - anomaly_count,
        total_windows=len(windows),
        detailed_results=detailed_results
    )

# @app.post("/api/analyze-file", response_model=FileAnalysisResponse)
# async def analyze_file(file: UploadFile = File(...)):
#     """Analyze uploaded JSON file"""
#     if not model:
#         raise HTTPException(status_code=503, detail="Model not loaded")
    
#     if not file.filename.endswith('.json'):
#         raise HTTPException(status_code=400, detail="Only JSON files are supported")
    
#     try:
#         # Read and parse file
#         content = await file.read()
#         data = json.loads(content)
        
#         # Ensure it's a list
#         if not isinstance(data, list):
#             data = [data]
        
#         # Process each conversation
#         results = []
#         anomaly_conversations = 0
        
#         for conv in data:
#             # Generate ID if not present
#             if 'id' not in conv or conv['id'] is None:
#                 conv['id'] = f"CONV_{generate_conversation_id()}"
            
#             # Mask labels
#             conv = mask_labels(conv)
            
#             # Get utterances
#             utterances = conv.get('utterance', conv.get('utterances', []))
            
#             if len(utterances) < 3:
#                 continue
            
#             # Convert to messages
#             messages = [
#                 Message(
#                     original_form=utt.get('original_form', ''),
#                     speaker=i % 2  # Alternate speakers if not specified
#                 )
#                 for i, utt in enumerate(utterances)
#             ]
            
#             # Analyze conversation
#             anomaly_detected = False
#             analyzed_utterances = []
            
#             # Sliding window analysis
#             message_anomaly_counts = {}
#             for start_idx in range(0, len(messages) - WINDOW_SIZE + 1, STRIDE):
#                 end_idx = min(start_idx + WINDOW_SIZE, len(messages))
#                 window_messages = messages[start_idx:end_idx]
                
#                 result = predict_window(window_messages)
                
#                 if result['prediction'] == 1:
#                     anomaly_detected = True
#                     # Track which messages are in anomalous windows
#                     for i in range(start_idx, end_idx):
#                         if i < len(utterances):
#                             message_anomaly_counts[i] = message_anomaly_counts.get(i, 0) + 1
            
#             # Determine which messages are anomalous based on frequency in anomalous windows
#             anomaly_indices = []
#             threshold = max(1, len(messages) // WINDOW_SIZE // 2)  # Adaptive threshold
            
#             for idx, count in message_anomaly_counts.items():
#                 if count >= threshold:
#                     utterances[idx]['predicted_label'] = 1
#                     anomaly_indices.append(idx)
            
#             # Set predicted labels for all utterances
#             for i, utt in enumerate(utterances):
#                 if 'predicted_label' not in utt:
#                     utt['predicted_label'] = 0
#                 analyzed_utterances.append(utt)
            
#             if anomaly_detected:
#                 anomaly_conversations += 1
            
#             results.append({
#                 'id': conv['id'],
#                 'has_anomaly': anomaly_detected,
#                 'utterances': analyzed_utterances,
#                 'anomaly_indices': anomaly_indices
#             })
        
#         total_conversations = len(results)
#         detection_rate = (anomaly_conversations / total_conversations * 100) if total_conversations > 0 else 0
        
#         return FileAnalysisResponse(
#             total_conversations=total_conversations,
#             anomaly_conversations=anomaly_conversations,
#             normal_conversations=total_conversations - anomaly_conversations,
#             detection_rate=detection_rate,
#             conversations=results
#         )
        
#     except json.JSONDecodeError:
#         raise HTTPException(status_code=400, detail="Invalid JSON file")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/api/analyze-file", response_model=FileAnalysisResponse)
async def analyze_file(file: UploadFile = File(...)):
    """Analyze uploaded JSON file by running a model-window around each message."""
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Only JSON files are supported")

    try:
        content = await file.read()
        data = json.loads(content)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file")

    if not isinstance(data, list):
        data = [data]

    results = []
    anomaly_conversations = 0

    for conv in data:
        # ensure an ID
        conv_id = conv.get("id") or f"CONV_{generate_conversation_id()}"
        utterances = conv.get("utterance", conv.get("utterances", []))
        if len(utterances) == 0:
            continue

        # run model around each utterance
        flagged_indices = []
        for idx in range(len(utterances)):
            window_msgs = build_padded_window(utterances, idx)
            # prediction dict has keys: prediction, confidence, anomaly_probability
            pred = predict_window(window_msgs)
            if pred["prediction"] == 1 and utterances[idx].get("original_form", "").strip():
                flagged_indices.append(idx)

        has_anomaly = len(flagged_indices) > 0
        if has_anomaly:
            anomaly_conversations += 1

        # annotate each utterance with predicted_label
        for i, utt in enumerate(utterances):
            utt["predicted_label"] = 1 if i in flagged_indices else 0

        results.append({
            "id": conv_id,
            "has_anomaly": has_anomaly,
            "utterances": utterances,
            "anomaly_indices": flagged_indices
        })

    total = len(results)
    detection_rate = (anomaly_conversations / total * 100) if total else 0.0

    return FileAnalysisResponse(
        total_conversations=total,
        anomaly_conversations=anomaly_conversations,
        normal_conversations=total - anomaly_conversations,
        detection_rate=detection_rate,
        conversations=results
    )

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    }

# ------------------------- Run the server ------------------------- #
if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("models_pure_lm", exist_ok=True)
    
    # Run the server
    uvicorn.run(
        "demo_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )