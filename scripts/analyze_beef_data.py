import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import joblib
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

# Update data path
DATA_PATH = 'data/beef_data.csv'

class MultiHeadAttention(nn.Module):
    def __init__(self, in_features, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = max(1, in_features // num_heads)

        self.query = nn.Linear(in_features, self.head_dim * num_heads)
        self.key = nn.Linear(in_features, self.head_dim * num_heads)
        self.value = nn.Linear(in_features, self.head_dim * num_heads)
        self.proj = nn.Linear(self.head_dim * num_heads, in_features)
        self.layer_norm = nn.LayerNorm(in_features)
        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        batch_size = x.size(0)

        q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim)
        k = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = self.dropout(torch.softmax(scores, dim=-1))

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.head_dim * self.num_heads)
        out = self.dropout(self.proj(out))

        return self.layer_norm(x + out.squeeze(-2))

class FeatureInteraction(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.W = nn.Parameter(torch.randn(in_features, in_features) / np.sqrt(in_features))
        self.V = nn.Parameter(torch.randn(in_features, 32) / np.sqrt(32))
        self.layer_norm = nn.LayerNorm(in_features)

    def forward(self, x):
        # Linear interactions
        linear = torch.matmul(x, self.W)

        # Factorization machine style interactions
        square_of_sum = torch.matmul(x, self.V).pow(2)
        sum_of_square = torch.matmul(x.pow(2), self.V.pow(2))
        pair_interactions = 0.5 * (square_of_sum - sum_of_square).sum(1, keepdim=True)

        return self.layer_norm(x + linear + pair_interactions)

class AdvancedResidualBlock(nn.Module):
    def __init__(self, in_features, expansion_factor=4):
        super().__init__()
        hidden_features = in_features * expansion_factor

        self.layer_norm1 = nn.LayerNorm(in_features)
        self.layer_norm2 = nn.LayerNorm(in_features)

        self.block = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_features, in_features),
            nn.Dropout(0.15)
        )

        self.se = nn.Sequential(
            nn.Linear(in_features, max(in_features // 4, 1)),
            nn.ReLU(),
            nn.Linear(max(in_features // 4, 1), in_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x
        x = self.layer_norm1(x)
        x = self.block(x)
        x = x * self.se(x)
        x = self.layer_norm2(x + residual)
        return x

class BeefPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.attention = MultiHeadAttention(input_size)
        self.feature_interaction = FeatureInteraction(input_size)

        self.input_proj = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15)
        )

        self.encoder = nn.ModuleList([
            AdvancedResidualBlock(256) for _ in range(3)
        ])

        self.decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.15),
            AdvancedResidualBlock(128),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.15)
        )

        self.output = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # Multi-head attention
        x = self.attention(x)

        # Feature interactions
        x = self.feature_interaction(x)

        # Main network
        x = self.input_proj(x)

        for encoder_layer in self.encoder:
            x = encoder_layer(x)

        x = self.decoder(x)
        return self.output(x)

class BeefDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

class EnsemblePredictor(nn.Module):
    def __init__(self, input_size, num_models=5):
        super().__init__()

        # Base models with different architectures
        self.base_models = nn.ModuleList([
            BeefPredictor(input_size) for _ in range(num_models)
        ])

        # Initialize with different dropout rates for diversity
        for i, model in enumerate(self.base_models):
            if i > 0:  # Keep first model as baseline
                for module in model.modules():
                    if isinstance(module, nn.Dropout):
                        module.p = 0.15 + (i * 0.05)  # Increasing dropout rates

        # Learnable weights
        self.weights = nn.Parameter(torch.ones(num_models) / num_models)

    def forward(self, x):
        # Get predictions from base models
        predictions = torch.stack([model(x) for model in self.base_models], dim=-1)
        # Weighted average of predictions
        weights = torch.softmax(self.weights, dim=0)
        return (predictions * weights).sum(-1)

def analyze_feature_importance(X, y):
    # Train a Random Forest to get feature importance
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # Get feature importance
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    # Plot feature importance
    plt.figure(figsize=(12, 6))
    plt.bar(importance['feature'][:15], importance['importance'][:15])
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 15 Most Important Features')
    plt.tight_layout()
    plt.savefig('outputs/feature_importance.png')
    plt.close()

    # Select top features
    top_features = importance['feature'][:20].tolist()  # Keep top 20 features
    return top_features

def load_and_preprocess_data():
    # Load the data
    data = pd.read_csv(DATA_PATH)
    print("Loaded data successfully!\n")

    # Separate features and target
    X = data.drop('tot_vol_sirloin', axis=1)
    y = data['tot_vol_sirloin']

    # Add polynomial features
    X_poly = X.copy()
    for col1 in X.columns:
        for col2 in X.columns:
            if col1 <= col2:  # Avoid duplicates
                X_poly[f'{col1}_{col2}'] = X[col1] * X[col2]

    # Analyze feature importance and select top features
    top_features = analyze_feature_importance(X_poly, y)
    X_poly = X_poly[top_features]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_poly, y, test_size=0.2, random_state=42
    )

    # Scale the features
    X_scaler = StandardScaler()
    X_train_scaled = X_scaler.fit_transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    # Scale the target
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))

    # Save scalers
    joblib.dump(X_scaler, 'outputs/X_scaler.joblib')
    joblib.dump(y_scaler, 'outputs/y_scaler.joblib')

    return (X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled,
            X_train, X_test, y_train, y_test, X_scaler, y_scaler)

def analyze_data(data):
    print("Data Analysis:")
    print("-" * 50 + "\n")

    print("Dataset Shape:", data.shape)
    print("\nFeature Statistics:")
    print(data.describe())
    print("\nCorrelation with Target (tot_vol_sirloin):")
    correlations = data.corr()['tot_vol_sirloin'].sort_values(ascending=False)
    print(correlations)

    # Create correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('outputs/correlation_heatmap.png')
    plt.close()

    print("\nData preprocessing completed!\n")

def train_model(X_train, X_test, y_train, y_test, X_scaler):
    train_dataset = BeefDataset(X_train, y_train)
    test_dataset = BeefDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = EnsemblePredictor(X_train.shape[1])
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-7
    )

    # Loss functions
    mse_criterion = nn.MSELoss()
    huber_criterion = nn.HuberLoss(delta=1.0)

    print("Training the model...")
    best_test_loss = float('inf')
    patience = 30
    patience_counter = 0
    best_model = None

    for epoch in range(400):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)

            # Combined loss
            loss = 0.5 * mse_criterion(output, batch_y) + 0.5 * huber_criterion(output, batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                output = model(batch_X)
                test_loss += mse_criterion(output, batch_y).item()

        train_loss /= len(train_loader)
        test_loss /= len(test_loader)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Test Loss = {test_loss:.6f}")

    torch.save(best_model, 'outputs/best_model.pth')
    return model

def evaluate_model(model, X_test, y_test, y_scaler):
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        predictions = model(X_test_tensor)

        # Transform predictions and actual values back to original scale
        predictions = y_scaler.inverse_transform(predictions.numpy())
        y_test_original = y_scaler.inverse_transform(y_test)

        # Calculate metrics
        mse = np.mean((predictions - y_test_original) ** 2)
        rmse = np.sqrt(mse)
        r2 = 1 - np.sum((y_test_original - predictions) ** 2) / np.sum((y_test_original - np.mean(y_test_original)) ** 2)

        print("\nModel Evaluation:")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"Root Mean Squared Error: {rmse:.2f}")
        print(f"R² Score: {r2:.4f}")

        # Plot actual vs predicted values
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test_original, predictions, alpha=0.5)
        plt.plot([y_test_original.min(), y_test_original.max()],
                [y_test_original.min(), y_test_original.max()],
                'r--', lw=2)
        plt.xlabel('Actual Sirloin Volume')
        plt.ylabel('Predicted Sirloin Volume')
        plt.title('Actual vs Predicted Sirloin Volume')
        plt.tight_layout()
        plt.savefig('prediction_plot.png')
        plt.close()

        # Example prediction
        example_idx = 0
        print(f"\nExample Prediction:")
        print(f"Actual Volume: {y_test_original[example_idx][0]:.2f}")
        print(f"Predicted Volume: {predictions[example_idx][0]:.2f}")

def predict_sirloin_volume(input_data):
    # Load the model and scalers
    model = EnsemblePredictor(input_data.shape[1])
    model.load_state_dict(torch.load('outputs/best_model.pth'))
    X_scaler = joblib.load('outputs/X_scaler.joblib')
    y_scaler = joblib.load('outputs/y_scaler.joblib')

    # Preprocess input data
    input_scaled = X_scaler.transform(input_data)

    # Make prediction
    model.eval()
    with torch.no_grad():
        input_tensor = torch.FloatTensor(input_scaled)
        prediction_scaled = model(input_tensor)
        prediction = y_scaler.inverse_transform(prediction_scaled.numpy())

    return prediction

def main():
    # Load and preprocess data
    data = pd.read_csv(DATA_PATH)
    analyze_data(data)

    (X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled,
     X_train, X_test, y_train, y_test, X_scaler, y_scaler) = load_and_preprocess_data()

    # Train the model
    model = train_model(X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, X_scaler)

    # Final evaluation
    evaluate_model(model, X_test_scaled, y_test_scaled, y_scaler)

    # Example prediction
    example_input = X_train.iloc[[0]]
    prediction = predict_sirloin_volume(example_input)
    print(f"\nPrediction for example input:")
    print(f"Input features:\n{example_input}")
    print(f"Predicted sirloin volume: {prediction[0][0]:.2f}")

if __name__ == "__main__":
    main()
