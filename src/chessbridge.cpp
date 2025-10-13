#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>


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
        case Outcome::ONGOING:
        default:
            return 0;
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

    // Convert outcome to numeric label (+1, 0, -1, etc.)
    int label = outcomeToLabel(result);

    // Number of positions
    size_t N = result.positions.size();
    if (N == 0)
        throw std::runtime_error("No positions generated from self-play.");

    // 🧠 PyTorch expects NCHW: [N, 12, 8, 8]
    std::vector<ssize_t> dims = {static_cast<ssize_t>(N), 12, 8, 8};

    // Create NumPy arrays
    py::array_t<float> X_np(dims);
    py::array_t<float> y_np(N);

    // Access buffers
    auto X_ptr = X_np.mutable_unchecked<4>();
    auto y_ptr = y_np.mutable_unchecked<1>();

    // Fill arrays
    for (size_t i = 0; i < N; ++i) {
        const auto& pos = result.positions[i];
        y_ptr(i) = static_cast<float>(label);
        // Ensure tensor data exists
        if (pos.tensorData.empty()) {
            const_cast<Position&>(pos).toTensor(); // compute and cache
        }

        const std::vector<float>& tensor = pos.tensorData;
        if (tensor.size() != 12 * 8 * 8)
            throw std::runtime_error("Invalid tensor size for position " + std::to_string(i));

        // Copy flat tensor into 4D array (NCHW)
        size_t idx = 0;
        for (int ch = 0; ch < 12; ++ch)
            for (int r = 0; r < 8; ++r)
                for (int c = 0; c < 8; ++c)
                    X_ptr(i, ch, r, c) = tensor[idx++];
    }

    // ✅ Return (X_np, y_np)
    return py::make_tuple(X_np, y_np);
}




PYBIND11_MODULE(chessbridge, m) {
    m.doc() = "C++ self-play generator for chess NN training";

    // --------------------------------------------------------------------
    // 🧩 Existing bindings (for training & evaluation)
    // --------------------------------------------------------------------
    m.def("run_selfplay_for_python", &run_selfplay_for_python,
          py::arg("depth") = 3,
          py::arg("verbose") = false,
          "Run self-play and return (X, y) NumPy arrays");

    m.def("evaluate_fen", &chess::evaluateFEN,
          "Evaluate a position from a FEN string",
          py::arg("fen"),
          py::arg("depth") = 1);

    // --------------------------------------------------------------------
    // ♟️ Expose your C++ Piece enum (used by Flask mappings)
    // --------------------------------------------------------------------
    py::enum_<chess::Piece>(m, "Piece")
        .value("W_PAWN",   chess::Piece::W_PAWN)
        .value("W_KNIGHT", chess::Piece::W_KNIGHT)
        .value("W_BISHOP", chess::Piece::W_BISHOP)
        .value("W_ROOK",   chess::Piece::W_ROOK)
        .value("W_QUEEN",  chess::Piece::W_QUEEN)
        .value("W_KING",   chess::Piece::W_KING)
        .value("B_PAWN",   chess::Piece::B_PAWN)
        .value("B_KNIGHT", chess::Piece::B_KNIGHT)
        .value("B_BISHOP", chess::Piece::B_BISHOP)
        .value("B_ROOK",   chess::Piece::B_ROOK)
        .value("B_QUEEN",  chess::Piece::B_QUEEN)
        .value("B_KING",   chess::Piece::B_KING)
        .value("NO_PIECE", chess::Piece::NO_PIECE);

    // Also alias them directly for convenience:
    m.attr("W_PAWN")   = m.attr("Piece").attr("W_PAWN");
    m.attr("W_KNIGHT") = m.attr("Piece").attr("W_KNIGHT");
    m.attr("W_BISHOP") = m.attr("Piece").attr("W_BISHOP");
    m.attr("W_ROOK")   = m.attr("Piece").attr("W_ROOK");
    m.attr("W_QUEEN")  = m.attr("Piece").attr("W_QUEEN");
    m.attr("W_KING")   = m.attr("Piece").attr("W_KING");
    m.attr("B_PAWN")   = m.attr("Piece").attr("B_PAWN");
    m.attr("B_KNIGHT") = m.attr("Piece").attr("B_KNIGHT");
    m.attr("B_BISHOP") = m.attr("Piece").attr("B_BISHOP");
    m.attr("B_ROOK")   = m.attr("Piece").attr("B_ROOK");
    m.attr("B_QUEEN")  = m.attr("Piece").attr("B_QUEEN");
    m.attr("B_KING")   = m.attr("Piece").attr("B_KING");

    // --------------------------------------------------------------------
    // 🎯 PromoPieceType enum
    // --------------------------------------------------------------------
    py::enum_<chess::PromoPieceType>(m, "PromoPieceType")
        .value("PROMO_TYPE_NONE", chess::PromoPieceType::PROMO_TYPE_NONE)
        .value("PROMO_TYPE_N", chess::PromoPieceType::PROMO_TYPE_N)
        .value("PROMO_TYPE_B", chess::PromoPieceType::PROMO_TYPE_B)
        .value("PROMO_TYPE_R", chess::PromoPieceType::PROMO_TYPE_R)
        .value("PROMO_TYPE_Q", chess::PromoPieceType::PROMO_TYPE_Q);

    // --------------------------------------------------------------------
    // 🏁 Outcome enum
    // --------------------------------------------------------------------
    py::enum_<chess::Outcome>(m, "Outcome")
        .value("ONGOING", chess::Outcome::ONGOING)
        .value("CHECKMATE", chess::Outcome::CHECKMATE)
        .value("STALEMATE", chess::Outcome::STALEMATE)
        .value("DRAW_FIFTY_MOVE", chess::Outcome::DRAW_FIFTY_MOVE)
        .value("DRAW_THREEFOLD_REPETITION", chess::Outcome::DRAW_THREEFOLD_REPETITION);

    // --------------------------------------------------------------------
    // 🧠 Engine class
    // --------------------------------------------------------------------
    py::class_<chess::Engine>(m, "Engine")
        .def(py::init<int>(), py::arg("depth") = 3)
        .def("findBestMove", &chess::Engine::findBestMove,
             py::arg("position"), py::arg("history_hashes"))
        .def("getNodesVisited", &chess::Engine::getNodesVisited)
        .def("currentOutcome", &chess::Engine::currentOutcome,
             py::arg("position"), py::arg("history_hashes"))
        .def("perft", &chess::Engine::perft,
             py::arg("position"), py::arg("depth"));

    // --------------------------------------------------------------------
    // ⚙️ Position class
    // --------------------------------------------------------------------
    py::class_<chess::Position>(m, "Position")
        .def(py::init<>())
        .def("updateOccupancies", &chess::Position::updateOccupancies)
        .def("syncMailboxFromBitboards", &chess::Position::syncMailboxFromBitboards)
        .def("computeAndSetHash", &chess::Position::computeAndSetHash)
        .def_readwrite("whiteToMove", &chess::Position::whiteToMove)
        .def_readwrite("castlingRights", &chess::Position::castlingRights)
        .def_readwrite("epSquare", &chess::Position::epSquare)
        .def_readwrite("halfmoveClock", &chess::Position::halfmoveClock)
        .def_readwrite("fullmoveNumber", &chess::Position::fullmoveNumber)
        .def_readwrite("bb", &chess::Position::bb)
        .def_readwrite("currentHash", &chess::Position::currentHash);
}

