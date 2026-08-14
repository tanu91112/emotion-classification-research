import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import mlflow
from .config import config
from .data_loader import load_bigru_data

class BiGRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bigru = nn.GRU(embed_dim, hidden_dim, num_layers=2, 
                           bidirectional=True, batch_first=True, dropout=0.5)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.bigru(embedded)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(self.dropout(hidden))

def train_bigru():
    print("=" * 60)
    print(" Training BiGRU Model")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_bigru_data()
    
    model = BiGRUClassifier(data["vocab_size"], config.bigru_embed_dim,
                           config.bigru_hidden_dim, config.num_labels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    print(f" Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
    
    mlflow.set_experiment(config.mlflow_experiment)
    with mlflow.start_run(run_name="bigru"):
        best_acc = 0
        for epoch in range(config.num_epochs_bigru):
            model.train()
            train_loss, correct, total = 0, 0, 0
            for batch in tqdm(data["train_loader"], desc=f"Epoch {epoch+1}"):
                input_ids = batch["input_ids"].to(device)
                labels = batch["label"].to(device)
                optimizer.zero_grad()
                outputs = model(input_ids)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            
            # Evaluate
            model.eval()
            test_correct, test_total = 0, 0
            with torch.no_grad():
                for batch in data["test_loader"]:
                    input_ids = batch["input_ids"].to(device)
                    labels = batch["label"].to(device)
                    outputs = model(input_ids)
                    _, preds = torch.max(outputs, 1)
                    test_correct += (preds == labels).sum().item()
                    test_total += labels.size(0)
            
            test_acc = test_correct / test_total
            print(f"Epoch {epoch+1}: Test Acc: {test_acc:.4f}")
            mlflow.log_metric("test_accuracy", test_acc, step=epoch)
            
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), f"{config.bigru_path}/best_model.pt")
                print(f" New best: {best_acc:.4f}")
        
        print(f" BiGRU Best Accuracy: {best_acc:.4f}")
        return model, best_acc

if __name__ == "__main__":
    train_bigru()


