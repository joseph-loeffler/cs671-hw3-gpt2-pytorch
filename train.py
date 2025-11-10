import torch
import torch.nn.functional as F
from model import Transformer
from data import get_dataloaders
from tqdm import tqdm

# Hyperparameters
block_size = 64
batch_size = 12
embed_dim = 128
n_layer = 4
n_head = 4
learning_rate = 6e-4
max_epochs = 10000
eval_interval = 500

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load data
train_loader, test_loader, vocab_size, stoi, itos = get_dataloaders(
    "shakespeare.txt", block_size, batch_size, train_split=0.9
)

# Initialize model
model = Transformer(vocab_size=vocab_size, block_size=block_size, embed_dim=embed_dim, n_layer=n_layer)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1, betas=(0.9, 0.95))

# Training loop, one epoch only
model.train()
losses = []
for epoch_num, (x, y) in enumerate(tqdm(train_loader)):
  if epoch_num >= max_epochs:
      break

  # Forward pass
  x, y = x.to(device), y.to(device)
  logits = model(x)
  loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
  losses.append(loss.item())

  # Backward pass
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

torch.save(model, "model.pt")
import matplotlib.pyplot as plt

plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

