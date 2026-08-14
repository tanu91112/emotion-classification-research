import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from collections import Counter
from .config import config

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=50, vocab=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.vocab = vocab
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        tokens = self.tokenizer.tokenize(text)[:self.max_len]
        ids = [self.vocab.get(tok, 1) for tok in tokens]
        if len(ids) < self.max_len:
            ids += [0] * (self.max_len - len(ids))
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }

def build_vocab(texts, vocab_size=10000):
    counter = Counter()
    for text in texts:
        counter.update(text.lower().split())
    vocab = {'<pad>': 0, '<unk>': 1}
    for word, _ in counter.most_common(vocab_size - 2):
        vocab[word] = len(vocab)
    return vocab

def load_bigru_data():
    dataset = load_dataset(config.dataset_name)
    vocab = build_vocab(dataset["train"]["text"], config.bigru_vocab_size)
    
    class SimpleTokenizer:
        def __init__(self, vocab):
            self.vocab = vocab
        def tokenize(self, text):
            return text.lower().split()
    
    tokenizer = SimpleTokenizer(vocab)
    
    train_dataset = EmotionDataset(
        dataset["train"]["text"], dataset["train"]["label"],
        tokenizer, config.max_length, vocab
    )
    test_dataset = EmotionDataset(
        dataset["test"]["text"], dataset["test"]["label"],
        tokenizer, config.max_length, vocab
    )
    
    return {
        "train_loader": DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True),
        "test_loader": DataLoader(test_dataset, batch_size=config.eval_batch_size, shuffle=False),
        "vocab_size": len(vocab),
    }

def load_distilbert_data():
    dataset = load_dataset(config.dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(config.distilbert_model)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", 
                        truncation=True, max_length=config.max_length)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")
    
    return {"tokenized_dataset": tokenized_dataset, "tokenizer": tokenizer}


