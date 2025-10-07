#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#ifdef _MSC_VER
  #include <BaseTsd.h>
  typedef SSIZE_T ssize_t;
#endif

#include "game_manager.hpp"
#include <vector>
#include <iostream>

namespace py = pybind11;
using namespace chess;

// Converts the game's final outcome to a numerical label.
int outcomeToLabel(const GameResult& result) {
    switch (result.outcome) {
        case Outcome::CHECKMATE:
            // If it's black's turn to move, it means white delivered the checkmate.
            return result.positions.back().whiteToMove ? -1 : +1; // -1 Black wins, +1 White wins
        case Outcome::STALEMATE:
            return 0;
        case Outcome::DRAW_FIFTY_MOVE:
            return 2;
        case Outcome::DRAW_THREEFOLD_REPETITION:
            return 2; // Draw
    }
}

/**
 * @brief Runs a self-play game and formats the results for Python.
 *
 * @param depth The search depth for the chess engine.
 * @param verbose Whether to print game progress to the console.
 * @return A Python tuple containing two NumPy arrays:
 * X: The training data, shape (N, 8, 8, 12)
 * y: The labels, shape (N,)
 */
py::tuple run_selfplay_for_python(int depth, bool verbose) {
    // Run one self-play game
    auto result = runSelfPlayGame(depth, verbose);

    // Convert final outcome into label (+1, 0, -1, 2)
    int label = outcomeToLabel(result);

    // Number of positions generated during this game
    size_t N = result.positions.size();

    // Define the shape for the input tensor
    std::vector<ssize_t> dims = {static_cast<ssize_t>(N), 8, 8, 12};

    // Create NumPy arrays
    py::array_t<float> X_np(dims);
    py::array_t<float> y_np(N);

    // Access mutable buffers
    auto X_ptr = X_np.mutable_unchecked<4>();
    auto y_ptr = y_np.mutable_unchecked<1>();

    // Populate tensors
    for (size_t i = 0; i < N; ++i) {
        const auto& pos = result.positions[i];

        // Assign same label for each position
        y_ptr(i) = static_cast<float>(label);

        // Fill board tensor
        for (int piece_type = 0; piece_type < 12; ++piece_type) {
            Bitboard b = pos.bb[piece_type];
            for (int r = 0; r < 8; ++r) {
                for (int c = 0; c < 8; ++c) {
                    int sq = r * 8 + c;
                    X_ptr(i, r, c, piece_type) = ((b >> sq) & 1ULL) ? 1.0f : 0.0f;
                }
            }
        }
    }

    // ✅ Return (X_np, y_np)
    return py::make_tuple(X_np, y_np);
}



PYBIND11_MODULE(chessbridge, m) {
    m.doc() = "C++ self-play generator for chess NN training";
    m.def("run_selfplay_for_python", &run_selfplay_for_python,
          py::arg("depth") = 3,
          py::arg("verbose") = false,
          "Run self-play and return (X, y) NumPy arrays");
}