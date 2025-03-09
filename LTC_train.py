import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
import numpy as np
from ncps.torch import LTC

import pickle


# Import LTC from the ncps package


# Load dataset
data = np.load('dataset.npz')

X_train = torch.tensor(data['X_train']).float()
y_train = torch.tensor(data['y_train']).float()
X_test = torch.tensor(data['X_test']).float()
y_test = torch.tensor(data['y_test']).float()

target_scaler = MinMaxScaler()
target_scaler.min_ = data['target_scaler_min']
target_scaler.scale_ = data['target_scaler_scale']

# Define improved LTC model
class ImprovedLTC(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2):
        super(ImprovedLTC, self).__init__()
        self.num_layers = num_layers
        
        # Create the first LTC layer with input_size
        self.ltc_layers = nn.ModuleList()
        self.ltc_layers.append(LTC(input_size=input_size, units=hidden_size))
        
        # Create subsequent LTC layers with hidden_size as input
        for _ in range(1, num_layers):
            self.ltc_layers.append(LTC(input_size=hidden_size, units=hidden_size))
        
        # Optional dropout between layers
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Final fully connected block to produce output
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # x shape: (batch, seq_length, input_size)
        for layer in self.ltc_layers:
            x, _ = layer(x)
            if self.dropout is not None:
                x = self.dropout(x)
        # Use the output at the last time step from the final LTC layer
        out = self.fc(x[:, -1, :])
        return out

# Create model, loss function, optimizer, and scheduler
model = ImprovedLTC(dropout = .2, num_layers =2 ,input_size=X_train.shape[-1], hidden_size=64)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, verbose=True
)

# Training loop with early stopping
epochs = 100
batch_size = 64
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)

best_val_loss = float('inf')
patience = 10
counter = 0
early_stop = False

train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs.squeeze(), batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test)
        val_loss = criterion(val_outputs.squeeze(), y_test)
        val_losses.append(val_loss.item())
    
    scheduler.step(val_loss)
    
    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss.item():.4f}')
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_ltc_model.pth')
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            early_stop = True
            break

    if early_stop:
        break

# Load the best model
model.load_state_dict(torch.load('best_ltc_model.pth'))

# Evaluation
model.eval()
with torch.no_grad():
    preds = model(X_test)
    test_loss = criterion(preds.squeeze(), y_test)
    print(f"LTC Test MSE: {test_loss.item():.4f}")

# In this case, if your target is the first feature,
# we assume y_test and preds represent only that one value.
# Since the scaler was fitted on all features, if you need to inverse-transform,
# use a dummy array approach. Otherwise, if the target was scaled separately, simply:
y_actual = target_scaler.inverse_transform(y_test.numpy().reshape(-1, 1)).flatten()
y_pred = target_scaler.inverse_transform(preds.numpy().reshape(-1, 1)).flatten()

results = {
    'ltc_y_actual': y_actual,         
    'ltc_y_pred': y_pred,
    'ltc_train_losses': train_losses,   
    'ltc_val_losses': val_losses,
   
}

# Save to a file called "results.pkl"
with open('results_ltc.pkl', 'wb') as f:
    pickle.dump(results, f)