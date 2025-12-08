import os
import subprocess
import random
import collections
import math
import time
import csv
import pickle
import shutil
import chess
import chess.engine

# --- Paths / Constants ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Ensure these point to valid executables on your system
ENGINE_PATH = os.path.join(PROJECT_ROOT, "stockfish")        
NODCHIP_PATH = os.path.join(PROJECT_ROOT, "nstockfish")      
STATE_FILE = os.path.join(PROJECT_ROOT, "training_state.pkl")
TRAINING_FILE = os.path.join(PROJECT_ROOT, "training_data.plain")
TRAINING_BINPACK = os.path.join(PROJECT_ROOT, "training_data.binpack")
ARCHIVE_FILE = os.path.join(PROJECT_ROOT, "archive_data.plain")
ARCHIVE_MAX_BYTES = 200 * 1024 * 1024  # 200 MB rotation threshold

NNUE_PYTORCH_DIR = os.path.join(PROJECT_ROOT, "nnue-pytorch")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
BASELINE_NET = os.path.join(MODELS_DIR, "version0", "net.nnue")    

CSV_LOG_PATH = os.path.join(PROJECT_ROOT, "training_log.csv")

# --- Self-play configuration ---
NNUE_DEPTH = 8
MAX_MOVES = 70
# TOGGLE THIS: True for Experiment (Material Win), False for Control (Draw)
use_material_adjudication = True 
MAX_MOVES_BENCHMARK = 250
NUM_GAMES_PER_ITER = 100          # used when SELFPLAY_MODE = "games"
SELFPLAY_TIME_PER_ITER_SEC = 600  # used when SELFPLAY_MODE = "time" (e.g. 10 minutes)

# SELFPLAY_MODE: "games" (original behavior) or "time"
SELFPLAY_MODE = "time"

REPLAY_BUFFER_SIZE = 8000000
OPENINGS_FILE = os.path.join(PROJECT_ROOT, "openings.txt")
MATE_SCORE = 10000

# --- Training loop configuration ---
NUM_ITERATIONS = 10  

# --- Benchmark configuration ---
BENCH_GAMES_VS_PREVIOUS = 200
BENCH_GAMES_VS_BASELINE = 200 

# --- Determinism / randomness configuration ---
DETERMINISTIC_MODE = False  
MASTER_SEED = 123456        
ENABLE_TRAINING_SEED_ARG = False


# ------------------ Utility helpers ------------------ #
def rebalance_training_data(raw_blocks, decisive_ratio=0.8):
    """Return a new list of blocks oversampling decisive results."""
    decisive = []
    neutral = []

    # Classify blocks
    for block in raw_blocks:
        if "result 1" in block or "result -1" in block:
            decisive.append(block)
        else:
            neutral.append(block)

    if not decisive:   # no decisive? just return original data
        return raw_blocks

    # Compute target counts
    total = len(raw_blocks)
    target_decisive = int(total * decisive_ratio)
    target_neutral = total - target_decisive

    # Sample with replacement if needed
    decisive_selected = [
        decisive[i % len(decisive)]
        for i in range(target_decisive)
    ]
    neutral_selected = [
        neutral[i % len(neutral)]
        for i in range(target_neutral)
    ]

    return decisive_selected + neutral_selected

def calculate_material_count(board: chess.Board) -> int:
    """
    Calculates the material balance score on the board.
    Returns the absolute score: (White Material - Black Material).
    Positive = White winning, Negative = Black winning.
    """
    # Standard piece values
    PIECE_VALUES = {
        chess.PAWN: 100,
        chess.KNIGHT: 300,
        chess.BISHOP: 300,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 0
    }

    white_material = 0
    black_material = 0

    # Iterate over all piece types and squares
    for piece_type, value in PIECE_VALUES.items():
        white_material += len(board.pieces(piece_type, chess.WHITE)) * value
        black_material += len(board.pieces(piece_type, chess.BLACK)) * value

    # Always return White - Black (Absolute perspective)
    return white_material - black_material

def ensure_dirs():
    os.makedirs(MODELS_DIR, exist_ok=True)

def clear_logs():
    logs_dir = os.path.join(NNUE_PYTORCH_DIR, "logs")
    if os.path.exists(logs_dir):
        print(f"[Train] Cleaning log directory to ensure output path is valid...")
        shutil.rmtree(logs_dir)

def get_candidate_paths():
    """Returns the expected paths of the newly generated files in the build dirs."""
    # 1. The Checkpoint (Brain) - strictly inside logs
    ckpt_path = os.path.join(
        NNUE_PYTORCH_DIR,
        "logs", "lightning_logs", "version_0", "checkpoints", "last.ckpt"
    )
    # 2. The NNUE (Body) - we will tell serialize.py to put it here
    nnue_path = os.path.join(NNUE_PYTORCH_DIR, "candidate.nnue")
    
    return nnue_path, ckpt_path

def save_new_version(version: int, source_nnue: str, source_ckpt: str):
    """Scenario A: New model is better. MOVE files to models/version{N}."""
    dest_dir = get_model_dir(version)
    os.makedirs(dest_dir, exist_ok=True)

    # Move Checkpoint
    dest_ckpt = os.path.join(dest_dir, "checkpoint.ckpt")
    shutil.move(source_ckpt, dest_ckpt)

    # Move NNUE
    dest_nnue = os.path.join(dest_dir, "net.nnue")
    shutil.move(source_nnue, dest_nnue)

    print(f"[Version Control] Created Version {version} (New Best).")
    return dest_nnue

def duplicate_previous_version(version: int):
    """Scenario B: New model failed. Copy Version{N-1} to Version{N}."""
    prev_dir = get_model_dir(version - 1)
    dest_dir = get_model_dir(version)
    
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
        
    print(f"[Version Control] New model failed. Cloning {prev_dir} -> {dest_dir}")
    shutil.copytree(prev_dir, dest_dir)
    
    return os.path.join(dest_dir, "net.nnue")

def rotate_archive_if_needed():
    """Simple rotation: if archive gets too big, rename it once and start fresh."""
    if os.path.exists(ARCHIVE_FILE):
        size = os.path.getsize(ARCHIVE_FILE)
        if size > ARCHIVE_MAX_BYTES:
            rotated = ARCHIVE_FILE + ".1"
            print(f"[Archive] {ARCHIVE_FILE} > {ARCHIVE_MAX_BYTES} bytes, rotating to {rotated}")
            if os.path.exists(rotated):
                os.remove(rotated)
            os.rename(ARCHIVE_FILE, rotated)

def get_model_dir(version: int) -> str:
    """Returns path to models/version{N}/"""
    return os.path.join(MODELS_DIR, f"version{version}")

def get_checkpoint_path(version: int) -> str:
    """Returns path to models/version{N}/checkpoint.ckpt"""
    return os.path.join(get_model_dir(version), "checkpoint.ckpt")

def get_net_path(version: int) -> str:
    """Returns path to models/version{N}/net.nnue"""
    return os.path.join(get_model_dir(version), "net.nnue")

def get_versioned_net_path(version: int) -> str:
    return os.path.join(MODELS_DIR, f"net_v{version:04d}.nnue")

def save_state(replay_buffer, iteration):
    """Saves the current buffer and iteration index to disk."""
    print(f"\n[State] Saving {len(replay_buffer)} positions to {STATE_FILE}...")
    try:
        with open(STATE_FILE, "wb") as f:
            pickle.dump({
                "buffer": replay_buffer,
                "iteration": iteration
            }, f)
        print("[State] Save complete.")
    except Exception as e:
        print(f"[State] Failed to save state: {e}")

def load_state(replay_buffer):
    """Loads buffer and iteration index if the file exists."""
    if not os.path.exists(STATE_FILE):
        print("[State] No saved state found. Starting fresh.")
        return 1  # Start at iteration 1

    print(f"[State] Loading saved state from {STATE_FILE}...")
    try:
        with open(STATE_FILE, "rb") as f:
            data = pickle.load(f)
            
        # Restore buffer
        saved_buffer = data["buffer"]
        replay_buffer.clear()
        replay_buffer.extend(saved_buffer)
        
        # Restore iteration
        last_iter = data["iteration"]
        print(f"[State] Resumed! Buffer size: {len(replay_buffer)} | Next Iteration: {last_iter + 1}")
        return last_iter + 1
        
    except Exception as e:
        print(f"[State] Error loading state (starting fresh): {e}")
        return 1


# ------------------ CSV logging ------------------ #

class ExperimentCSVLogger:
    HEADER = [
        "iteration",
        "games_played",
        "avg_game_length_plies",
        "positions_added",
        "iter_energy_uj",
        "training_duration_sec",
        "exported_net_path",
        "bench_prev_w",
        "bench_prev_l",
        "bench_prev_d",
        "bench_prev_elo",
        "bench_base_w",
        "bench_base_l",
        "bench_base_d",
        "bench_base_elo",
        "timed_out",
        "deterministic",
        "master_seed",
        "iteration_seed",
    ]

    def __init__(self, path: str):
        self.path = path
        self._ensure_header()

    def _ensure_header(self):
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.HEADER)

    def append_row(self, stats: dict):
        row = [
            stats.get("iteration"),
            stats.get("games_played"),
            stats.get("avg_game_length_plies"),
            stats.get("positions_added"),
            stats.get("iter_energy_uj"),
            stats.get("training_duration_sec"),
            stats.get("exported_net_path"),
            stats.get("bench_prev_w"),
            stats.get("bench_prev_l"),
            stats.get("bench_prev_d"),
            stats.get("bench_prev_elo"),
            stats.get("bench_base_w"),
            stats.get("bench_base_l"),
            stats.get("bench_base_d"),
            stats.get("bench_base_elo"),
            stats.get("timed_out"),
            stats.get("deterministic"),
            stats.get("master_seed"),
            stats.get("iteration_seed"),
        ]
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)


# ------------------ Determinism helpers ------------------ #

def setup_seeds(deterministic: bool, master_seed: int | None):
    if not deterministic:
        print("[Determinism] DETERMINISTIC_MODE = False -> stochastic behavior.")
        return None

    if master_seed is None:
        master_seed = 123456

    print(f"[Determinism] DETERMINISTIC_MODE = True, MASTER_SEED = {master_seed}")
    random.seed(master_seed)
    return master_seed


def get_iteration_seed(master_seed: int | None, iter_idx: int):
    if master_seed is None:
        return None
    return master_seed + iter_idx


def assess_determinism(deterministic_mode: bool) -> tuple[bool, str]:
    if not deterministic_mode:
        return False, "Stochastic: no global seeding, engine search + training nondeterministic."
    else:
        return False, (
            "Partially seeded but still effectively stochastic: engine search, "
            "thread scheduling, and training backend are not fully controlled."
        )


# ------------------ Self-play utilities ------------------ #

def pick_random_opening():
    if not os.path.exists(OPENINGS_FILE):
        return chess.Board()

    with open(OPENINGS_FILE, "r") as f:
        lines = f.readlines()
    if not lines:
        return chess.Board()

    fen_line = random.choice(lines).strip()
    if fen_line.startswith("pos "):
        fen_line = fen_line[4:]
    try:
        return chess.Board(fen_line)
    except ValueError:
        return chess.Board()


def run_selfplay_game(engine: chess.engine.SimpleEngine):
    """
    Run a single self-play game using an EXISTING engine instance.
    """
    board = pick_random_opening()
    game_history = []
    move_counter = 0

    while not board.is_game_over():

        info = engine.analyse(board, chess.engine.Limit(depth=NNUE_DEPTH))
        # Important: Get score from White's perspective for training data consistency
        score_white = info["score"].white().score(mate_score=MATE_SCORE)

        if "pv" in info and len(info["pv"]) > 0:
            best_move = info["pv"][0]
        else:
            best_move = list(board.legal_moves)[0]

        game_history.append({
            "fen": board.fen(),
            "move": best_move.uci(),
            "score": score_white,
            "ply": len(board.move_stack),
            "turn": board.turn,
        })

        board.push(best_move)
        move_counter += 1
        
        # --- TIMEOUT / ADJUDICATION LOGIC ---
        if move_counter >= MAX_MOVES:
            if use_material_adjudication:
                # Experiment Logic: Synthetic result based on material
                material_score = calculate_material_count(board)
                if material_score == 0:
                    score = 0
                elif material_score > 0:
                    score = 1
                else:
                    score = -1
                return game_history, score 
            else:
                # Control Logic: Standard Draw on timeout
                return game_history, 0

    result_str = board.result()
    if result_str == "1-0":
        global_result = 1
    elif result_str == "0-1":
        global_result = -1
    else:
        global_result = 0

    return game_history, global_result


def format_positions(game_history, global_result):
    blocks = []
    for step in game_history:
        result_pov = global_result

        block = (
            f"fen {step['fen']}\n"
            f"move {step['move']}\n"
            f"score {step['score']}\n"
            f"ply {step['ply']}\n"
            f"result {result_pov}\n"
            "e\n"
        )
        blocks.append(block)
    return blocks


# ------------------ Data management ------------------ #

def write_to_archive(blocks):
    rotate_archive_if_needed()
    with open(ARCHIVE_FILE, "a") as f_arch:
        for block in blocks:
            f_arch.write(block)


def convertDataToBinPack():
    if not os.path.exists(NODCHIP_PATH):
        raise FileNotFoundError(f"nstockfish not found at {NODCHIP_PATH}")
    cmd = [NODCHIP_PATH, "convert", TRAINING_FILE, TRAINING_BINPACK]
    print("[Convert] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ------------------ Training + export ------------------ #

def train_network(iter_idx: int, train_seed: int | None = None):
    """
    Trains the network for the current iteration.
    STRICTLY resumes from models/version{iter_idx - 1}/checkpoint.ckpt.
    """
    if not os.path.exists(TRAINING_BINPACK):
        raise FileNotFoundError(f"Binpack not found: {TRAINING_BINPACK}")

    # 1. Identify the checkpoint from the PREVIOUS iteration
    prev_version = iter_idx - 1
    resume_ckpt_path = os.path.join(MODELS_DIR, f"version{prev_version}", "checkpoint.ckpt")

    # 2. Clear temporary logs
    clear_logs()

    # 3. Build the training command
    cmd = [
        "python", "train.py",
        "../training_data.binpack", "../training_data.binpack", 
        "--features", "HalfKAv2_hm",
        "--l1", "1536",
        "--max_epochs", "1",
        "--epoch-size", "100000",
        "--batch-size", "2048",
        "--threads", "4",
        "--num-workers", "1",
        "--random-fen-skipping", "0",
        "--lr", "2.0e-4",
        "--gamma", "0.99",
        # CHANGED: Lambda 0.0 forces learning from GAME RESULT (material), ignoring random Eval
        "--lambda", "0.1", 
        "--pc-y1", "1.0",
        "--pc-y2", "1.0",
        "--pc-y3", "1.0",
        "--w1", "0.2",
        "--w2", "0.5",
        "--save-last-network", "true",
    ]

    # 4. Resume logic: FORCE resumption or crash
    if not os.path.exists(resume_ckpt_path):
        raise FileNotFoundError(f"CRITICAL: Cannot resume. Checkpoint missing at: {resume_ckpt_path}")

    print(f"[Train] Resuming from Version {prev_version} checkpoint: {resume_ckpt_path}")
    cmd.extend(["--resume-from-model", resume_ckpt_path])

    # 5. Determinism handling
    if ENABLE_TRAINING_SEED_ARG and train_seed is not None:
        cmd.extend(["--seed", str(train_seed)])

    print(f"[Train] Starting training for Iteration {iter_idx}...")
    subprocess.run(cmd, check=True, cwd=NNUE_PYTORCH_DIR)


def export_candidate_nnue() -> str:
    """
    Exports the latest training result to 'candidate.nnue' in the nnue-pytorch directory.
    """
    ckpt_path = os.path.join(
        NNUE_PYTORCH_DIR,
        "logs", "lightning_logs", "version_0", "checkpoints", "last.ckpt"
    )
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Output to the location main() expects: nnue-pytorch/candidate.nnue
    out_path = os.path.join(NNUE_PYTORCH_DIR, "candidate.nnue")
    
    cmd = [
        "python", "serialize.py",
        os.path.abspath(ckpt_path),
        os.path.abspath(out_path),
        "--features=HalfKAv2_hm",
        "--l1", "1536",
    ]
    print("[Export] Generating candidate network:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=NNUE_PYTORCH_DIR)

    if not os.path.exists(out_path):
        raise FileNotFoundError(f"Expected candidate NNUE not found: {out_path}")
    
    return out_path


# ------------------ Benchmarking ------------------ #

def run_engine_match(net_new: str, net_other: str, num_games: int) -> dict:
    if num_games <= 0:
        return {
            "wins_new": 0, "losses_new": 0, "draws": 0, "elo_diff": None,
        }

    if not os.path.exists(ENGINE_PATH):
        raise FileNotFoundError(f"Engine not found at {ENGINE_PATH}")
    if not os.path.exists(net_new):
        raise FileNotFoundError(f"NNUE net not found: {net_new}")
    if not os.path.exists(net_other):
        raise FileNotFoundError(f"NNUE net not found: {net_other}")

    wins_new = 0
    losses_new = 0
    draws = 0

    engine_new = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
    engine_other = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)

    try:
        engine_new.configure({"EvalFile": net_new})
        engine_other.configure({"EvalFile": net_other})

        for game_idx in range(num_games):
            board = pick_random_opening()
            move_counter = 0
            
            # Alternate engines by side
            new_plays_white = (game_idx % 2 == 0)
            engines = {
                chess.WHITE: engine_new if new_plays_white else engine_other,
                chess.BLACK: engine_other if new_plays_white else engine_new,
            }

            while not board.is_game_over() and move_counter < MAX_MOVES_BENCHMARK:
                engine = engines[board.turn]
                info = engine.analyse(board, chess.engine.Limit(depth=NNUE_DEPTH))
                if "pv" in info and len(info["pv"]) > 0:
                    best_move = info["pv"][0]
                else:
                    best_move = random.choice(list(board.legal_moves))
                board.push(best_move)
                move_counter += 1

            result_str = board.result()

            if result_str == "1-0":
                if new_plays_white: wins_new += 1
                else: losses_new += 1
            elif result_str == "0-1":
                if new_plays_white: losses_new += 1
                else: wins_new += 1
            else:
                draws += 1

            # Print match progress every game
            print(
                f"[Match] Game {game_idx+1}/{num_games} completed. "
                f"Score so far: +{wins_new} -{losses_new} ={draws}"
            )

    finally:
        engine_new.quit()
        engine_other.quit()

    total_games = wins_new + losses_new + draws
    if total_games == 0:
        elo_diff = None
    else:
        score_new = (wins_new + 0.5 * draws) / total_games
        score_new = max(0.01, min(0.99, score_new))
        elo_diff = 400.0 * math.log10(score_new / (1.0 - score_new))

    return {
        "wins_new": wins_new,
        "losses_new": losses_new,
        "draws": draws,
        "elo_diff": elo_diff,
    }



# ------------------ One self-play + train iteration ------------------ #

def selfplay_iteration(
    iter_idx: int,
    current_net: str,
    baseline_net: str,
    replay_buffer,
    master_seed: int | None,
    deterministic_mode: bool,
) -> tuple[str, dict]:
    print(f"\n=== Iteration {iter_idx} | Using net: {current_net} ===")

    iteration_seed = get_iteration_seed(master_seed, iter_idx)
    if deterministic_mode and iteration_seed is not None:
        print(f"[Determinism] Iter {iter_idx}: seeding Python RNG with {iteration_seed}")
        random.seed(iteration_seed)

    
    # Clear old training file from disk at start of iteration
    if os.path.exists(TRAINING_FILE):
        os.remove(TRAINING_FILE)

    games_played = 0
    total_plies = 0
    positions_added = 0
    timed_out = 0

    selfplay_start_time = time.perf_counter()

    if not os.path.exists(ENGINE_PATH):
        raise FileNotFoundError(f"Engine not found at {ENGINE_PATH}")
    
    with chess.engine.SimpleEngine.popen_uci(ENGINE_PATH) as engine:
        
        if not os.path.exists(current_net):
            raise FileNotFoundError(f"NNUE net not found: {current_net}")
        engine.configure({"EvalFile": current_net})

        # --- OPTIMIZED LOOP: No file I/O here ---
        while True:
            # Check constraints
            if SELFPLAY_MODE == "games":
                if games_played >= NUM_GAMES_PER_ITER:
                    break
            elif SELFPLAY_MODE == "time":
                elapsed = time.perf_counter() - selfplay_start_time
                if elapsed >= SELFPLAY_TIME_PER_ITER_SEC:
                    timed_out = 1
                    break

             # Print progress every 10 games
            if (games_played + 1) % 10 == 0:
                print(
                    f"[Self-Play] Iter {iter_idx} Game {games_played + 1} "
                    f"(mode={SELFPLAY_MODE})",
                    end="", flush=True,
                )

            history, result = run_selfplay_game(engine)
            
            new_blocks = format_positions(history, result)
            
            # Archive immediately (append mode is generally fast)
            write_to_archive(new_blocks)
            
            # Update RAM buffer only
            replay_buffer.extend(new_blocks)

            game_len = len(history)
            games_played += 1
            total_plies += game_len
            positions_added += len(new_blocks)

    print(f"[Data] Rebalancing sampling distribution for decisive outcomes...")
    filtered_data = rebalance_training_data(list(replay_buffer), decisive_ratio=0.8)

    print(f"[DEBUG] Replay buffer count before sampling: {len(replay_buffer)}")
    print(f"[DEBUG] Filtered (training) sample size: {len(filtered_data)}")

    print(f"[Data] Flushing {len(filtered_data)} sampled positions to {TRAINING_FILE}...")
    with open(TRAINING_FILE, "w") as f_train:
        f_train.writelines(filtered_data)


    convertDataToBinPack()

    train_start = time.perf_counter()
    train_network(iter_idx=iter_idx, train_seed=iteration_seed if deterministic_mode else None)
    new_net = export_candidate_nnue()
    train_duration = time.perf_counter() - train_start

    print(f"[Iter {iter_idx}] Exported new net: {new_net}")

    print(f"[Benchmark] Iter {iter_idx}: new vs previous ({BENCH_GAMES_VS_PREVIOUS} games)")
    bench_prev = run_engine_match(new_net, current_net, BENCH_GAMES_VS_PREVIOUS)

    print(f"[Benchmark] Iter {iter_idx}: new vs baseline ({BENCH_GAMES_VS_BASELINE} games)")
    bench_base = run_engine_match(new_net, baseline_net, BENCH_GAMES_VS_BASELINE)

    avg_game_length_plies = (total_plies / games_played) if games_played > 0 else 0.0

    stats = {
        "iteration": iter_idx,
        "games_played": games_played,
        "avg_game_length_plies": avg_game_length_plies,
        "positions_added": positions_added,
        "training_duration_sec": train_duration,
        "exported_net_path": os.path.abspath(new_net),
        "bench_prev_w": bench_prev["wins_new"],
        "bench_prev_l": bench_prev["losses_new"],
        "bench_prev_d": bench_prev["draws"],
        "bench_prev_elo": bench_prev["elo_diff"],
        "bench_base_w": bench_base["wins_new"],
        "bench_base_l": bench_base["losses_new"],
        "bench_base_d": bench_base["draws"],
        "bench_base_elo": bench_base["elo_diff"],
        "timed_out": timed_out,
        "deterministic": int(deterministic_mode),
        "master_seed": master_seed,
        "iteration_seed": iteration_seed,
    }

    return new_net, stats


# ------------------ Driver ------------------ #

def main():
    ensure_dirs()

    master_seed = setup_seeds(DETERMINISTIC_MODE, MASTER_SEED)
    overall_deterministic, det_msg = assess_determinism(DETERMINISTIC_MODE)
    print(f"[Determinism] Overall deterministic = {overall_deterministic}")
    print(f"[Determinism] Explanation: {det_msg}")

    csv_logger = ExperimentCSVLogger(CSV_LOG_PATH)

    

    # Initialize Buffer
    replay_buffer = collections.deque(maxlen=REPLAY_BUFFER_SIZE)
    
    # --- 1. LOAD STATE (Resume Logic) ---
    start_iter = load_state(replay_buffer)
    
    # UNIFIED LOGIC: Always load the network from the previous iteration index.
    # If start_iter == 1, this automatically looks for models/version0/net.nnue
    prev_net = get_net_path(start_iter - 1)
    
    if not os.path.exists(prev_net):
        raise FileNotFoundError(f"CRITICAL: Network file missing at {prev_net}. Cannot start/resume.")
    
    current_net = prev_net
    print(f"[State] Using network: {current_net}")

    # --- 2. MAIN LOOP (Wrapped in try/except for Ctrl+C) ---
    try:
        for iter_idx in range(start_iter, NUM_ITERATIONS + 1):
            new_net, stats = selfplay_iteration(
                iter_idx=iter_idx,
                current_net=current_net,
                baseline_net=BASELINE_NET,
                replay_buffer=replay_buffer,
                master_seed=master_seed,
                deterministic_mode=DETERMINISTIC_MODE,
            )
            # --- THE DECISION LOGIC ---
            elo_diff = stats["bench_prev_elo"]
            
            # Check if valid (not None) and improved (e.g. > 0 or > -10 depending on preference)
            is_better = (elo_diff is not None) and (elo_diff > -5)
            if(is_better):
                print(f"[Main] Success! Elo diff: {elo_diff:.2f}. Saving new version.")
                # Get paths of the temp files
                temp_nnue, temp_ckpt = get_candidate_paths()
                
                # Copy them to models/version{iter_idx}
                final_net_path = save_new_version(iter_idx, temp_nnue, temp_ckpt)
            else:
                print(f"[Main] Failure. Elo diff: {elo_diff}. Carrying over previous version.")
                
                # Copy models/version{iter_idx-1} -> models/version{iter_idx}
                final_net_path = duplicate_previous_version(iter_idx)
            current_net = final_net_path
            csv_logger.append_row(stats)
            
            # --- 3. SAVE STATE (After every iteration) ---
            save_state(replay_buffer, iter_idx)

    except KeyboardInterrupt:
        print("\n\n[State] PAUSE REQUESTED (KeyboardInterrupt).")
        print("[State] Your progress was saved at the end of the last completed iteration.")
        print("[State] Exiting safely.")
        return


    print("\n=== Training Complete ===")
    print(f"Final net: {current_net}")
    print("Load into Stockfish with:")
    print(f"  setoption name EvalFile value {os.path.abspath(current_net)}")

if __name__ == "__main__":
    main()