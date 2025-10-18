
# Optimizing Chess Engine Training Efficiency via Guided Curriculum Self-Play  

## Group Info  
- Vito Spatafora  
- Email: vitos@email.sc.edu  

## Project Summary/Abstract  
### This project presents a hybrid chess engine and reinforcement learning framework designed to investigate training efficiency through guided curriculum self-play. The system integrates a C++ engine—responsible for move generation, rule enforcement, and alpha-beta search—with a Python-based training loop that manages self-play orchestration, neural network evaluation, and data collection. A PyTorch evaluation model is exported to ONNX for fast inference within the engine, enabling seamless interaction via PyBind11. The research introduces a novel curriculum-based training approach that incrementally increases game length to accelerate early learning and reduce energy consumption.

## Problem Description  
- Problem description: Modern chess engines like Stockfish and Leela Chess Zero have achieved superhuman strength, but this success comes with an enormous computational and energy cost. Training these engines from scratch requires playing millions of full games, where early moves are often random and uninformative—wasting computation while the model slowly discovers basic patterns.  

- Motivation  
  - As AI systems scale, the inefficiency of current training methods becomes increasingly unsustainable.  
  - Reducing the energy cost of training can accelerate AI innovation while lowering its environmental impact.  
  - This project proposes a curriculum-based strategy to improve early learning efficiency, reducing convergence time and energy consumption without sacrificing final playing strength.  

- Challenges  
  - The primary challenges include model collapse and insufficient training performance.  
  - The network cannot currently train at sufficiently high search depths due to computational inefficiencies, resulting in poor-quality training data.  
  - The deterministic behavior of the model combined with limited data quality leads to repeated game sequences, producing a high number of draws by repetition.  

## Contribution  
### [`Novel contribution`]
- Propose and implement a guided curriculum self-play framework for chess engine training, introducing a structured progression of training phases rather than relying on conventional unguided, full-game self-play.
- Develop two neural network evaluation functions within the same chess engine framework:  
  - **Baseline (Unguided):** Trained via full-game self-play from the beginning.  
  - **Guided (Curriculum):** Trained on progressively longer games (e.g., 10–20 moves per side).  
- Measure and compare the training efficiency of both approaches in terms of energy consumption (kWh) and convergence rate (Elo improvement over time).  
- Analyze the final Elo strength per unit of energy consumed to determine if the curriculum approach provides a more efficient training method.  


## References  
[1] Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). *Curriculum learning.* In Proceedings of the 26th Annual International Conference on Machine Learning (pp. 41–48).  

[2] Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2015). *Prioritized experience replay.* arXiv preprint arXiv:1511.05952.  

[3] Nasu, Y. (2018). *Efficiently Updatable Neural-Network-based Evaluation Function for computer Shogi.* Ziosoft Computer Shogi Club.  

[4] Silver, D. et al. (2017). *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm.* arXiv.  

[5] Linscott, J. et al. (n.d.). *Leela Chess Zero.* https://lczero.org/  

[6] Stockfish Development Team. (n.d.). *Stockfish Chess Engine.* https://stockfishchess.org/  

[7] Henderson, P. et al. (2018). *Deep Reinforcement Learning That Matters.* AAAI.  

[8] Elo, A. E. (1978). *The Rating of Chessplayers, Past and Present.* Arco Publishing.  

# Dependencies

Python 3.12+

PyTorch — for neural network training and inference

NumPy — for numerical operations and array management

PyBind11 — for C++/Python interoperability

ONNX Runtime — for running exported neural network models efficiently in C++

CMake — for building the C++ engine and linking dependencies

# Instructions for Milestone 1 Results:

[1] Download Repository

[2] Run uv sync

[3] run cd src

[4] run make clean

[5] run make

[6] open nn.py

[7] set the following variables:
total_games = 1000 save_every = 1 depth = 1

[8] run nn.py
