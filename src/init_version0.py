import os
import subprocess
import shutil
import chess
import chess.engine

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ENGINE_PATH = os.path.join(PROJECT_ROOT, "stockfish")
NODCHIP_PATH = os.path.join(PROJECT_ROOT, "nstockfish")
NNUE_PYTORCH_DIR = os.path.join(PROJECT_ROOT, "nnue-pytorch")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
VERSION0_DIR = os.path.join(MODELS_DIR, "version0")

TRAINING_FILE = os.path.join(PROJECT_ROOT, "training_data.plain")
TRAINING_BINPACK = os.path.join(PROJECT_ROOT, "training_data.binpack")

# --- Step 1: Generate Data ---
def generate_genesis_data():
    # Optimization: If we already have the 1904 positions from your last run, 
    # we don't strictly need to regen them, but let's do it to be safe and clean.
    print("[Genesis] Playing 10 self-play games to ensure sufficient data...")
    
    if not os.path.exists(ENGINE_PATH):
        raise FileNotFoundError(f"Engine not found at {ENGINE_PATH}")

    all_game_blocks = []
    
    with chess.engine.SimpleEngine.popen_uci(ENGINE_PATH) as engine:
        for i in range(10):
            board = chess.Board()
            game_history = []
            while not board.is_game_over():
                info = engine.analyse(board, chess.engine.Limit(depth=6)) 
                score_white = info["score"].white().score(mate_score=10000)
                
                if "pv" in info and info["pv"]:
                    move = info["pv"][0]
                else:
                    move = list(board.legal_moves)[0]

                game_history.append({
                    "fen": board.fen(),
                    "move": move.uci(),
                    "score": score_white,
                    "ply": len(board.move_stack),
                })
                board.push(move)

            res_str = board.result()
            if res_str == "1-0": result = 1
            elif res_str == "0-1": result = -1
            else: result = 0

            for step in game_history:
                block = (
                    f"fen {step['fen']}\n"
                    f"move {step['move']}\n"
                    f"score {step['score']}\n"
                    f"ply {step['ply']}\n"
                    f"result {result}\n"
                    "e\n"
                )
                all_game_blocks.append(block)
            
            print(f"  > Game {i+1}/10 finished.")

    print(f"[Genesis] Writing {len(all_game_blocks)} positions to file...")
    with open(TRAINING_FILE, "w") as f:
        f.writelines(all_game_blocks)

# --- Step 2: Convert to Binpack ---
def convert_data():
    if not os.path.exists(NODCHIP_PATH):
        raise FileNotFoundError(f"nstockfish not found at {NODCHIP_PATH}")
        
    cmd = [NODCHIP_PATH, "convert", TRAINING_FILE, TRAINING_BINPACK]
    print("[Genesis] Converting to binpack...")
    subprocess.run(cmd, check=True)

# --- Step 3: Train Initial Model (Corrected) ---
def train_genesis_model():
    print("[Genesis] Training version0 model (Safe Mode)...")
    
    logs_dir = os.path.join(NNUE_PYTORCH_DIR, "logs")
    if os.path.exists(logs_dir):
        shutil.rmtree(logs_dir)

    cmd = [
        "python", "train.py",
        "../training_data.binpack", "../training_data.binpack",
        "--features", "HalfKAv2_hm",
        "--l1", "1536",
        "--max_epochs", "1",
        "--epoch-size", "100", 
        "--batch-size", "16",
        "--threads", "2",
        "--num-workers", "0",           # Main thread execution
        "--random-fen-skipping", "0",   # No skipping
        "--lr", "2.0e-4",
        "--gamma", "0.99",
        "--lambda", "0.1",
        "--save-last-network", "true",
    ]
    
    subprocess.run(cmd, check=True, cwd=NNUE_PYTORCH_DIR)

# --- Step 4: Serialize and Move ---
def finalize_files():
    print("[Genesis] Serializing and moving files...")
    os.makedirs(VERSION0_DIR, exist_ok=True)

    ckpt_path = os.path.join(NNUE_PYTORCH_DIR, "logs", "lightning_logs", "version_0", "checkpoints", "last.ckpt")
    nnue_out = os.path.join(NNUE_PYTORCH_DIR, "candidate.nnue")
    
    # Fallback search if version number incremented
    if not os.path.exists(ckpt_path):
        base_logs = os.path.join(NNUE_PYTORCH_DIR, "logs", "lightning_logs")
        if os.path.exists(base_logs):
            versions = sorted(os.listdir(base_logs))
            if versions:
                ckpt_path = os.path.join(base_logs, versions[-1], "checkpoints", "last.ckpt")

    serialize_cmd = [
        "python", "serialize.py",
        os.path.abspath(ckpt_path),
        os.path.abspath(nnue_out),
        "--features=HalfKAv2_hm",
        "--l1", "1536"
    ]
    subprocess.run(serialize_cmd, check=True, cwd=NNUE_PYTORCH_DIR)

    dest_ckpt = os.path.join(VERSION0_DIR, "checkpoint.ckpt")
    dest_nnue = os.path.join(VERSION0_DIR, "net.nnue")

    shutil.move(ckpt_path, dest_ckpt)
    shutil.move(nnue_out, dest_nnue)
    
    print(f"[Success] Version 0 created at: {VERSION0_DIR}")
    print(f"Files: \n - {dest_ckpt} \n - {dest_nnue}")

def main():
    try:
        generate_genesis_data()
        convert_data()
        train_genesis_model()
        finalize_files()
    except Exception as e:
        print(f"[Error] Failed to initialize version0: {e}")

if __name__ == "__main__":
    main()