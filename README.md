# Optimizing Chess Engine Training Efficiency via Guided Curriculum Self-Play  

## Group Info  
- Vito Spatafora  
- Email: vitos@email.sc.edu 

## Overview  
This project implements a fully automated pipeline for training Neural Network Updatable Evaluations (NNUE) for chess engines. The system explores whether curriculum-style reinforcement learning can accelerate convergence and improve efficiency when compared to conventional unguided self-play.

The framework integrates three key components:

1. **Stockfish 16 (Player):** Generates self-play games using the current network.
2. **Nodchip Stockfish (Converter):** Converts plain text training data into `.binpack` format for NNUE training.
3. **NNUE-PyTorch (Trainer):** Trains the neural network on GPU using official Stockfish tooling.

The pipeline repeatedly plays games, collects training examples, updates the evaluation network, and benchmarks progress across iterations.

---

## Research Motivation  
Modern chess engines reach superhuman performance but require tremendous computational and energy expenditure. Early game phases during training provide limited learning signal, leading to wasted resources.

This project investigates whether guided curriculum self-play can:

* Improve learning efficiency of nnue training from scratch
* Maintain or improve resulting playing strength

The curriculum approach constrains game length during early iterations and bases outcomes on material balance, distilling simple strategic principles before exposing the model to full gameplay complexity.

---

## Problem Statement  

Current reinforcement learning strategies rely on millions of long self-play games. Early positions contain little information, making training slow and inefficient. The challenge addressed here is:

**Can a staged training curriculum accelerate early learning and improve efficiency without degrading final chess strength?**

Key difficulties include:

* Preventing model collapse during small-game learning
* Ensuring high-quality training data across search depths
* Avoiding deterministic repetition loops caused by the engine and network interaction

---

## Contribution  
### [`Novel contribution`]
This project provides:

* A complete NNUE reinforcement learning loop using Stockfish, data conversion tools, and GPU-accelerated training.
* A side-by-side evaluation of:
  * **Baseline Training:** Full game self-play from iteration zero
  * **Curriculum Training:** Artificial game truncation with adjudication by material evaluation
* A monitoring mechanism to compare convergence speed, Elo growth, and iteration
* Empirical insight into whether human-style instructional scaffolding improves machine learning dynamics in chess

---

## System Architecture  

### Core Components  
1. **Stockfish 16**  
   Self-play engine providing move selection, search, and NNUE inference.

2. **Nodchip Stockfish**  
   Required to convert `.plain` training files into `.binpack` format for NNUE-PyTorch.

3. **NNUE-PyTorch**  
   Training backend with GPU acceleration used to update the neural network each iteration.

---

## Installation and Setup  

Execute the following in the src directory:

### 1. Build Stockfish (Self-Play Engine)
```bash
git clone --branch sf_16 https://github.com/official-stockfish/Stockfish.git stockfish_src
cd stockfish_src/src
make -j4 build ARCH=x86-64
mv stockfish ../../stockfish
cd ../..
./stockfish
```

### 2. Build Nodchip Stockfish (Converter)
```bash
git clone https://github.com/nodchip/Stockfish.git nstockfish_src
cd nstockfish_src/src
make -j4 build ARCH=x86-64
mv stockfish ../../nstockfish
cd ../..
```

### 3. Setup NNUE-PyTorch Trainer
```bash
git clone https://github.com/official-stockfish/nnue-pytorch.git
cd nnue-pytorch
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install python-chess
chmod +x setup_script.sh
./setup_script.sh
pip install cupy-cuda12x
cd ..
```

---

## Instructions for Finished Project  

### Phase 1: Initialize Version 0 Network  
```bash
source nnue-pytorch/venv/bin/activate
python init_version0.py
```

### Phase 2: Begin Iterative Self-Play Training, Key hyper-parameters (Depth, Max Moves) and the experiment toggle (use_material_adjudication) can be modified directly at the top of selfplay_datagen.py.
```bash
python selfplay_datagen.py
```

The system then executes iterative reinforcement learning cycles consisting of:

1. Self-play data generation  
2. Conversion to binpack  
3. NNUE training  
4. Benchmarking versus previous and baseline networks  
5. Version logging and model selection  

---

## Dependencies  

* Python 3.12+
* PyTorch (GPU recommended)
* NumPy  
* CMake

---

## References  

[1] Bengio, Y. et al. (2009). Curriculum Learning. ICML.  
[2] Schaul, T. et al. (2015). Prioritized Experience Replay. arXiv.  
[3] Nasu, Y. (2018). Efficiently Updatable Neural-Network-based Evaluation Function.  
[4] Silver, D. et al. (2017). Mastering Chess and Shogi by Self-Play. arXiv.  
[5] Leela Chess Zero Project. https://lczero.org/  
[6] Stockfish Development Team. https://stockfishchess.org/  
[7] Henderson, P. et al. (2018). Deep Reinforcement Learning That Matters. AAAI.  
[8] Elo, A. (1978). The Rating of Chessplayers, Past and Present.
