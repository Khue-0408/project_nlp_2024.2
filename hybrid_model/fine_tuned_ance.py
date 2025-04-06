# import pandas as pd
# import torch
# from torch.utils.data import DataLoader, Dataset
# from transformers import AutoTokenizer, AutoModel
# from torch import nn, optim
# from tqdm import tqdm 

# class ANCEDataset(Dataset):
#     def __init__(self, data, tokenizer, max_length=128):
#         self.data = data
#         self.tokenizer = tokenizer
#         self.max_length = max_length
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         query, doc_text, label = self.data[idx]
#         inputs = self.tokenizer(query, doc_text, return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_length)
#         return inputs, torch.tensor(label)

# def load_training_data():
#     train_df = pd.read_csv('data/train.csv')
#     pairs = []
#     for _, row in train_df.iterrows():
#         query = row['question']
#         doc_text = row['context']
#         label = 1  # 1 for positive pair
#         pairs.append((query, doc_text, label))
#     return pairs

# def fine_tune_ance():
#     model_name = 'sentence-transformers/msmarco-bert-base-dot-v5'
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModel.from_pretrained(model_name)
    
#     training_data = load_training_data()
#     dataset = ANCEDataset(training_data, tokenizer)
#     dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    
#     model.train()
#     num_epochs = 3
#     for epoch in range(num_epochs):
#         total_loss = 0
#         print(f"Epoch {epoch+1}/{num_epochs}")
        
#         # Wrap the DataLoader in a tqdm progress bar
#         for batch in tqdm(dataloader, desc=f"Training Epoch {epoch+1}", leave=False):
#             inputs, labels = batch
#             optimizer.zero_grad()
            
#             # Forward pass
#             outputs = model(**{k: v.squeeze(1) for k, v in inputs.items()})
#             logits = torch.matmul(outputs.pooler_output, outputs.pooler_output.T)
#             labels = torch.arange(logits.size(0)).long()  # Labels are diagonal (self-prediction)
            
#             # Compute loss
#             loss = criterion(logits, labels)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
        
#         # Print average loss per epoch
#         avg_loss = total_loss / len(dataloader)
#         print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
    
#     # Save fine-tuned model and tokenizer
#     model.save_pretrained('fine_tuned_ance')
#     tokenizer.save_pretrained('fine_tuned_ance')
#     print("Fine-tuning complete. Model saved to 'fine_tuned_ance'.")

# if __name__ == "__main__":
#     fine_tune_ance()
