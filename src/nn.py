import os
import chessbridge
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import random

# ----------------------------
# ⚙️ Device Configuration
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🧠 Using device: {device}")

# ----------------------------
# 🧠 Value Network Definition
# ----------------------------
class ValueNetwork(nn.Module):
    def __init__(self, lr=1e-3, input_channels=12):
        super(ValueNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(8 * 8 * 256, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        self.criterion = nn.MSELoss(reduction='none')
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        if x.ndim == 4:
            x = x.permute(0, 3, 1, 2)  # (N, 12, 8, 8)
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)

    def train_model(self, X, y, epochs=1, batch_size=64, repetition_mask=False):
        self.train()

        # Convert inputs
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        # Build dataset
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                self.optimizer.zero_grad()

                out = self.forward(batch_x)
                raw_loss = self.criterion(out, batch_y)

                # Apply weight to entire batch based on repetition flag
                weight = 0.25 if repetition_mask else 2.50
                loss = (raw_loss * weight).mean()

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

    def save(self, path="model.pt"):
        torch.save(self.state_dict(), path)
        print(f"💾 Saved model weights to {path}")

# ----------------------------
# 헬 Helper Function
# ----------------------------
def apply_discounted_rewards(final_outcome, num_positions, gamma=0.98):
    """
    Calculates discounted rewards for each position in a game.
    """
    discounted_y = np.zeros(num_positions)
    for i in range(num_positions):
        moves_from_end = num_positions - 1 - i
        discounted_y[i] = final_outcome * (gamma ** moves_from_end)
    return discounted_y

# ----------------------------
# 🚀 Main Training Pipeline
# ----------------------------
if __name__ == "__main__":
    model_path = "model.pt"
    onnx_path = "value_net.onnx"
    net = ValueNetwork().to(device)

    # 1️⃣ Load existing model if available
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            net.load_state_dict(state_dict)
            print(f"✅ Loaded existing weights from {model_path}")
        except Exception as e:
            print(f"⚠️ Failed to load {model_path}: {e}")
            print("🧠 Starting with a new model.")
    else:
        print(f"⚠️ No {model_path} found — starting fresh model.")

    # 2️⃣ Export ONNX only if missing
    dummy_input = torch.randn(1, 8, 8, 12).to(device)
    if not os.path.exists(onnx_path):
        print("📦 ONNX model not found — exporting initial version...")
        torch.onnx.export(
            net,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=17,
            dynamic_axes={"input": {0: "batch_size"}}
        )
        print(f"✅ ONNX model created at {onnx_path}")
    else:
        print(f"✅ Found existing ONNX model at {onnx_path} — skipping export.")

    # 3️⃣ Self-play training loop
    total_games = 10000
    save_every = 1
    depth = 1
    start = time.time()
    checkmates = 0
    stalemates = 0
    repetitions = 0

    for game_num in range(1, total_games + 1):
        print(f"\n🎮 Game {game_num}/{total_games}")

        # 1. Run self-play game
        X, y_raw = chessbridge.run_selfplay_for_python(depth, False)
        X = np.array(X)
        game_outcome_code = y_raw[0]

        # 2. Track stats and determine final outcome
        if game_outcome_code == 1.0 or game_outcome_code == -1.0:
            checkmates += 1
            final_outcome = game_outcome_code
        elif game_outcome_code == 0.0:
            stalemates += 1
            final_outcome = 0.0
        elif game_outcome_code == 2.0:
            repetitions += 1
            final_outcome = 0.0

        repetition_mask = (game_outcome_code == 2.0)

        # 3. Apply discounted rewards to create the new labels
        y = apply_discounted_rewards(final_outcome, len(X))
        print(f"🧩 X shape: {X.shape}, y shape: {y.shape}, first label: {y[0]:.3f}, last label: {y[-1]:.3f}")

        # 4. Train model with new labels
        net.train_model(X, y, epochs=1, batch_size=32, repetition_mask=repetition_mask)

        # 5. Periodic save
        if game_num % save_every == 0:
            net.save(model_path)
            print("📦 Re-exporting updated ONNX model...")
            torch.onnx.export(
                net,
                dummy_input,
                onnx_path,
                input_names=["input"],
                output_names=["output"],
                opset_version=17,
                dynamic_axes={"input": {0: "batch_size"}}
            )
            print(f"✅ Updated ONNX model saved at {onnx_path}")

    elapsed = time.time() - start
    print(f"\n✅ Finished {total_games} games in {elapsed:.2f}s")
    print(f"🏁 Final ONNX model ready at {onnx_path}")
    print(f"📊 Game Outcomes — Checkmates: {checkmates}, Stalemates: {stalemates}, Repetitions: {repetitions}")