import os
import chessbridge
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import random
from collections import deque

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
        
        # 🧠 Replaced CNN layers with a single lightweight MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_channels * 8 * 8, 256), # Input is flattened 12x8x8 board
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        
        # --- The rest of your init remains the same ---
        self.criterion = nn.MSELoss(reduction='none')
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        # ♟️ Flatten the input tensor first, then pass to the MLP
        x = x.reshape(x.size(0), -1)
        return self.mlp(x)

    def train_model(self, X, y, epochs=1, batch_size=64):
        self.train()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                self.optimizer.zero_grad()
                out = self.forward(batch_x)
                loss = self.criterion(out, batch_y).mean()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"🧮 Epoch {epoch+1}, Loss: {total_loss:.6f}")
    
    def save(self, path="model.pt"):
        torch.save(self.state_dict(), path)
        print(f"💾 Saved model weights to {path}")

# ----------------------------
# 💾 Replay Buffer Management (Weighted)
# ----------------------------
REPLAY_FILE = "replay_buffer_v2.npz"
REPLAY_MAXLEN = 200_000

class ReplayBuffer:
    def __init__(self, maxlen=REPLAY_MAXLEN):
        self.buffer = deque(maxlen=maxlen)

    def add(self, X, y, outcome_code):
        """
        Adds positions and their labels to the buffer with a priority weight.
        """
        # Weight important positions more (checkmates and losses > draws)
        if outcome_code in (1.0, -1.0):   # checkmate
            weight = 2.0
        elif outcome_code == 2.0:         # repetition draw
            weight = 0.5
        else:                             # stalemate/neutral
            weight = 1.0

        for i in range(len(X)):
            self.buffer.append((X[i], y[i], weight))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None, None, None

        # Normalize weights for probability sampling
        weights = np.array([w for _, _, w in self.buffer], dtype=np.float32)
        probs = weights / np.sum(weights)
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        batch = [self.buffer[idx] for idx in indices]
        batch_X = np.stack([b[0] for b in batch])
        batch_y = np.array([b[1] for b in batch])
        batch_w = np.array([b[2] for b in batch])
        return batch_X, batch_y, batch_w

    def save(self):
        if len(self.buffer) == 0:
            return
        X_arr = np.stack([x for x, _, _ in self.buffer])
        y_arr = np.array([y for _, y, _ in self.buffer])
        w_arr = np.array([w for _, _, w in self.buffer])
        np.savez_compressed(REPLAY_FILE, X=X_arr, y=y_arr, w=w_arr)
        print(f"💾 Saved replay buffer ({len(self.buffer)} samples) → {REPLAY_FILE}")

    def load(self):
        if not os.path.exists(REPLAY_FILE):
            print("🆕 No replay buffer found — starting empty.")
            return
        data = np.load(REPLAY_FILE, allow_pickle=True)
        X, y, w = data["X"], data["y"], data["w"]
        for i in range(len(X)):
            self.buffer.append((X[i], y[i], w[i]))
        print(f"📂 Loaded replay buffer with {len(self.buffer)} samples from {REPLAY_FILE}")

# ----------------------------
# 🧩 Helper Function
# ----------------------------
def apply_discounted_rewards(final_outcome, num_positions, gamma=0.98):
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

    # 1️⃣ Load model
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            net.load_state_dict(state_dict)
            print(f"✅ Loaded existing weights from {model_path}")
        except Exception as e:
            print(f"⚠️ Failed to load {model_path}: {e}")
    else:
        print(f"⚠️ No {model_path} found — starting fresh model.")

    # 2️⃣ Load replay buffer
    replay_buffer = ReplayBuffer()
    replay_buffer.load()
    y_values = [y for _, y, _ in replay_buffer.buffer]
    #print(f"Mean label: {np.mean(y_values):.3f}, Std: {np.std(y_values):.3f}")
    #print(f"Label distribution: {np.unique(np.sign(y_values), return_counts=True)}")


    # 3️⃣ Export ONNX if missing
    dummy_input = torch.randn(1, 12, 8, 8).to(device)
    if not os.path.exists(onnx_path):
        print("📦 Exporting initial ONNX model...")
        torch.onnx.export(
            net, dummy_input, onnx_path,
            input_names=["input"], output_names=["output"],
            opset_version=17, dynamic_axes={"input": {0: "batch_size"}}
        )
        print(f"✅ ONNX model created at {onnx_path}")

    # 4️⃣ Self-play training loop
    total_games = 1160
    save_every = 5
    depth = 3
    batch_size = 1024
    start = time.time()

    wcheckmates = bcheckmates = stalemates = repetitions = 0

    for game_num in range(1, total_games + 1):
        print(f"\n🎮 Game {game_num}/{total_games}")
        X, y_raw = chessbridge.run_selfplay_for_python(depth, False)
        X = np.array(X)
        game_outcome_code = y_raw[0]
        print(f"🔚 Game outcome code: {game_outcome_code}")

        # Track stats
        if game_outcome_code == 1.0:
            wcheckmates += 1
            final_outcome = game_outcome_code
        elif game_outcome_code == -1.0:
            bcheckmates += 1
            final_outcome = game_outcome_code
        elif game_outcome_code == 0.0:
            stalemates += 1
            final_outcome = 0.0
        else:
            repetitions += 1
            final_outcome = 0.0

        # Compute discounted rewards
        y = apply_discounted_rewards(final_outcome, len(X))

        # ✅ Add to replay buffer
        replay_buffer.add(X, y, game_outcome_code)
        print(f"📈 Replay buffer size: {len(replay_buffer.buffer)} / {REPLAY_MAXLEN}")


        # 💾 Save periodically
        if game_num % save_every == 0:
            # 🧠 Train using weighted sampling
            batch_X, batch_y, batch_w = replay_buffer.sample(batch_size)
            if batch_X is not None:
                # optional: scale loss by sample weights
                net.train_model(batch_X, batch_y, epochs=1, batch_size=batch_size)
            net.save(model_path)
            replay_buffer.save()
            print("📦 Re-exporting updated ONNX model...")
            torch.onnx.export(
                net, dummy_input, onnx_path,
                input_names=["input"], output_names=["output"],
                opset_version=17, dynamic_axes={"input": {0: "batch_size"}}
            )
            print(f"✅ Updated ONNX model saved at {onnx_path}")

    elapsed = time.time() - start
    print(f"\n✅ Finished {total_games} games in {elapsed:.2f}s")
    print(f"📊 White Checkmates: {wcheckmates}, Black Checkmates: {bcheckmates}, Stalemates: {stalemates}, Repetitions: {repetitions}")

