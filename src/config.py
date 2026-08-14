import os
from dataclasses import dataclass

@dataclass
class Config:
    # Dataset
    dataset_name: str = "dair-ai/emotion"
    labels: list = None
    max_length: int = 50
    
    # BiGRU
    bigru_vocab_size: int = 10000
    bigru_embed_dim: int = 300
    bigru_hidden_dim: int = 128
    
    # DistilBERT
    distilbert_model: str = "distilbert-base-uncased"
    num_labels: int = 6
    
    # LoRA
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: list = None
    
    # Training
    batch_size: int = 32
    eval_batch_size: int = 64
    num_epochs_bigru: int = 5
    num_epochs_distilbert: int = 3
    learning_rate: float = 2e-4
    
    # Paths
    bigru_path: str = "./models/bigru"
    distilbert_path: str = "./models/distilbert_lora"
    mlflow_experiment: str = "emotion_classification"
    
    def __post_init__(self):
        self.labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
        self.lora_target_modules = ["q_lin", "k_lin", "v_lin", "out_lin"]
        os.makedirs(self.bigru_path, exist_ok=True)
        os.makedirs(self.distilbert_path, exist_ok=True)
        os.makedirs("./logs", exist_ok=True)

config = Config()


