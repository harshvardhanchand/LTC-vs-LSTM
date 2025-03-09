import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle
data = np.load('dataset.npz')


X_train = torch.tensor(data['X_train']).float()
y_train = torch.tensor(data['y_train']).float()
X_test = torch.tensor(data['X_test']).float()
y_test = torch.tensor(data['y_test']).float()

# Reconstruct scalers if needed (here, we only need the target scaler for inverse transformation)
target_scaler = MinMaxScaler()
target_scaler.min_ = data['target_scaler_min']
target_scaler.scale_ = data['target_scaler_scale']

class ImprovedLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(ImprovedLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # Get the last output of LSTM sequence
        lstm_out, _ = self.lstm(x)
        # Use the output from the last time step
        out = self.fc(lstm_out[:, -1, :])
        return out


# Create model, loss function, and optimizer
model = ImprovedLSTM(input_size=X_train.shape[-1], hidden_size=64, num_layers = 2)

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

# Early stopping parameters
best_val_loss = float('inf')
patience = 10
counter = 0
early_stop = False

# Track losses for plotting
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
        # Add gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()
    
    # Calculate average training loss
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test)
        val_loss = criterion(val_outputs.squeeze(), y_test)
        val_losses.append(val_loss.item())
    
    # Update learning rate based on validation loss
    scheduler.step(val_loss)
    
    # Print progress
    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss.item():.4f}')
    
    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # Save the best model
        torch.save(model.state_dict(), 'best_lstm_model_1.pth')
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
model.load_state_dict(torch.load('best_lstm_model_1.pth'))

# Evaluation
model.eval()
with torch.no_grad():
    preds = model(X_test)
    mse = criterion(preds.squeeze(), y_test)
    print(f"LSTM Test MSE: {mse.item():.4f}")

# Inverse transform predictions and true values
y_actual = target_scaler.inverse_transform(y_test.numpy().reshape(-1, 1)).flatten()
y_pred = target_scaler.inverse_transform(preds.numpy().reshape(-1, 1)).flatten()

results = {
    'lstm_y_actual': y_actual,         
    'lstm_y_pred': y_pred,
    'lstm_train_losses': train_losses,   
    'lstm_val_losses': val_losses,
   
}

# Save to a file called "results.pkl"
with open('results_lstm.pkl', 'wb') as f:
    pickle.dump(results, f)