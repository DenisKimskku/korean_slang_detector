import os
import time
import json
import random
from collections import Counter
from datetime import datetime

import numpy as np
import psutil

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel

# ------------------------- Config (benchmark with 10% of training windows) ------------------------- #
class BenchmarkConfig:
    MODEL_NAME = "klue/bert-base"
    WINDOW_SIZE = 10
    STRIDE = 5
    MAX_LENGTH = 512
    SAMPLE_FRACTION = 0.02  # 트레이닝 윈도우의 10%만 사용
    RANDOM_SEED = 42
    TRAIN_DATA_DIR = '/home/minseok/forensic/NIKL_MESSENGER_v2.0/modified'
    MODEL_PATH = os.path.join('models_pure_lm_attn', 'best_model.pt')
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    WARMUP_WINDOWS = 10

# ------------------------- Model Definition ------------------------- #
class ConversationAnomalyDetector(nn.Module):
    def __init__(
        self,
        model_name: str,
        msg_token_id: int,
        attn_supervision_weight: float = 0.5,
        hidden_dropout: float = 0.3,
        attention_dropout: float = 0.1,
        use_attention_pooling: bool = True
    ):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.config = self.transformer.config
        H = self.config.hidden_size

        self.msg_token_id = msg_token_id
        self.attn_supervision_weight = attn_supervision_weight
        self.use_attention_pooling = use_attention_pooling

        if use_attention_pooling:
            self.attn_scoring = nn.Linear(H, 1)
            self.attn_dropout = nn.Dropout(attention_dropout)

        self.dropout = nn.Dropout(hidden_dropout)
        self.classifier = nn.Sequential(
            nn.Linear(H, H),
            nn.LayerNorm(H),
            nn.ReLU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(H, H // 2),
            nn.ReLU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(H // 2, 2)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(
                    mean=0.0,
                    std=self.config.initializer_range
                )
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True
        )
        seq = outputs.last_hidden_state  # [B,L,H]

        scores = self.attn_scoring(seq).squeeze(-1)  # [B,L]
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)  # [B,L]
        attn_weights = self.attn_dropout(attn_weights)

        pooled = torch.bmm(attn_weights.unsqueeze(1), seq).squeeze(1)  # [B,H]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # [B,2]

        return logits, attn_weights

# ------------------------- Data Loading & Window Construction ------------------------- #
def load_conversations_and_build_windows(input_dir, window_size, stride):
    windows = []
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    for file in json_files:
        path = os.path.join(input_dir, file)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, list):
                continue
            for conv in data:
                utts = conv.get('utterance', [])
                if len(utts) < window_size:
                    continue
                for start in range(0, len(utts) - window_size + 1, stride):
                    end = start + window_size
                    msgs = [u['original_form'] for u in utts[start:end]]
                    parts = []
                    for i, m in enumerate(msgs):
                        speaker = f"[SPEAKER_{i%2}]"
                        parts.append(f"[MSG] {speaker} {m}")
                    conv_text = " [SEP] ".join(parts)
                    windows.append(conv_text)
        except Exception as e:
            print(f"Warning: failed to load {file}: {e}")
    return windows

# ------------------------- Benchmarking Utility ------------------------- #
def measure_throughput_and_memory(model, tokenizer, windows, device, max_length, warmup_windows=10):
    model.eval()
    model.to(device)

    # Ensure special tokens exist
    special_tokens = {
        'additional_special_tokens': [
            '[SEP]', '[SPEAKER_0]', '[SPEAKER_1]', '[MASK]', '[MSG]'
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    model.transformer.resize_token_embeddings(len(tokenizer))  # embedding resize if changed
    from tqdm import tqdm
    # Warm-up
    with torch.no_grad():
        for i in tqdm(range(min(warmup_windows, len(windows)))):
            enc = tokenizer(
                windows[i],
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )
            input_ids = enc['input_ids'].to(device)
            attention_mask = enc['attention_mask'].to(device)
            _ = model(input_ids, attention_mask)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    start_time = time.time()
    with torch.no_grad():
        for text in tqdm(windows):
            enc = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )
            input_ids = enc['input_ids'].to(device)
            attention_mask = enc['attention_mask'].to(device)
            _ = model(input_ids, attention_mask)
    elapsed = time.time() - start_time
    throughput = len(windows) / elapsed if elapsed > 0 else 0.0

    # Memory measurement
    if device.type == 'cuda':
        gpu_mem_used = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        gpu_mem_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        mem_info = {
            'gpu_memory_used_mib': gpu_mem_used,
            'gpu_memory_reserved_mib': gpu_mem_reserved
        }
    else:
        process = psutil.Process(os.getpid())
        rss = process.memory_info().rss / (1024 ** 3)
        mem_info = {
            'cpu_rss_gb': rss
        }

    return {
        'num_windows': len(windows),
        'elapsed_seconds': elapsed,
        'throughput_windows_per_sec': throughput,
        'memory': mem_info
    }

# ------------------------- Main ------------------------- #
def main():
    torch.manual_seed(BenchmarkConfig.RANDOM_SEED)
    random.seed(BenchmarkConfig.RANDOM_SEED)
    np.random.seed(BenchmarkConfig.RANDOM_SEED)

    print(f"[{datetime.now().isoformat()}] Benchmark start on {BenchmarkConfig.DEVICE}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BenchmarkConfig.MODEL_NAME)
    tokenizer.add_special_tokens({'additional_special_tokens': ['[SEP]', '[SPEAKER_0]', '[SPEAKER_1]', '[MASK]', '[MSG]']})

    # Build sliding windows from training data
    print("Loading training data and constructing windows...")
    windows = load_conversations_and_build_windows(
        BenchmarkConfig.TRAIN_DATA_DIR,
        window_size=BenchmarkConfig.WINDOW_SIZE,
        stride=BenchmarkConfig.STRIDE
    )
    if not windows:
        raise RuntimeError("슬라이딩 윈도우가 생성되지 않았습니다. 데이터 경로/형식을 확인하세요.")
    print(f"전체 생성된 윈도우 수: {len(windows)}")

    # 샘플링: 10%만 사용 (고정 시드로 셔플한 뒤)
    random.Random(BenchmarkConfig.RANDOM_SEED).shuffle(windows)
    subset_size = max(1, int(len(windows) * BenchmarkConfig.SAMPLE_FRACTION))
    windows = windows[:subset_size]
    print(f"샘플링된 윈도우 10% 사용: {len(windows)}개")

    # Initialize model and load checkpoint
    msg_tok_id = tokenizer.convert_tokens_to_ids("[MSG]")
    model = ConversationAnomalyDetector(
        model_name=BenchmarkConfig.MODEL_NAME,
        msg_token_id=msg_tok_id,
        attn_supervision_weight=0.5,
        hidden_dropout=0.3,
        attention_dropout=0.1,
        use_attention_pooling=True
    )
    model.transformer.resize_token_embeddings(len(tokenizer))

    if not os.path.isfile(BenchmarkConfig.MODEL_PATH):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {BenchmarkConfig.MODEL_PATH}")
    state = torch.load(BenchmarkConfig.MODEL_PATH, map_location=BenchmarkConfig.DEVICE)
    if isinstance(state, dict) and 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
    model.to(BenchmarkConfig.DEVICE)

    # # CPU 측정
    # print("\n=== CPU 측정 ===")
    # cpu_result = measure_throughput_and_memory(
    #     model=model,
    #     tokenizer=tokenizer,
    #     windows=windows,
    #     device=torch.device('cpu'),
    #     max_length=BenchmarkConfig.MAX_LENGTH,
    #     warmup_windows=BenchmarkConfig.WARMUP_WINDOWS
    # )
    # print(json.dumps({"cpu": cpu_result}, indent=2, ensure_ascii=False))

    # GPU 측정 (가능한 경우)
    gpu_result = None
    if torch.cuda.is_available():
        print("\n=== GPU 측정 ===")
        gpu_result = measure_throughput_and_memory(
            model=model,
            tokenizer=tokenizer,
            windows=windows,
            device=torch.device('cuda'),
            max_length=BenchmarkConfig.MAX_LENGTH,
            warmup_windows=BenchmarkConfig.WARMUP_WINDOWS
        )
        print(json.dumps({"gpu": gpu_result}, indent=2, ensure_ascii=False))
    else:
        print("\nCUDA 사용 불가: GPU 벤치마크 생략됨.")

    # 요약 환산: 평균 100개 메시지의 1,000개 대화 처리 예상 시간
    avg_msgs_per_conv = 100
    windows_per_conv = max(1, (avg_msgs_per_conv - BenchmarkConfig.WINDOW_SIZE) // BenchmarkConfig.STRIDE + 1)
    convs = 1000
    total_windows = convs * windows_per_conv
    summary = {}
    if cpu_result['throughput_windows_per_sec'] > 0:
        summary['estimated_minutes_cpu'] = total_windows / cpu_result['throughput_windows_per_sec'] / 60
    if gpu_result and gpu_result['throughput_windows_per_sec'] > 0:
        summary['estimated_minutes_gpu'] = total_windows / gpu_result['throughput_windows_per_sec'] / 60

    print("\n=== 요약 추정 ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # 리포트 저장
    report = {
        'timestamp': datetime.now().isoformat(),
        'cpu': cpu_result,
        'gpu': gpu_result,
        'summary_estimate': summary,
        'config': {
            'window_size': BenchmarkConfig.WINDOW_SIZE,
            'stride': BenchmarkConfig.STRIDE,
            'max_length': BenchmarkConfig.MAX_LENGTH,
            'sample_fraction': BenchmarkConfig.SAMPLE_FRACTION,
            'model_name': BenchmarkConfig.MODEL_NAME
        }
    }
    out_path = f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n리포트 저장됨: {out_path}")

if __name__ == "__main__":
    main()
