#include "game_manager.hpp"
#include "chess_engine.hpp"
#include <vector>
#include <iostream>
#include <sstream>


namespace chess {

/**
 * @brief Plays a full self-play game between two identical engines.
 *
 * Each side uses the same search depth. All positions (before each move)
 * are stored for later training, and the final result is recorded.
 */
GameResult runSelfPlayGame(int searchDepth, bool verbose) {
    Engine engine(searchDepth);

    Position pos;

    // --- Set up the standard chess starting position ---
    // White pieces
    pos.bb[W_PAWN]   = 0x000000000000FF00ULL;
    pos.bb[W_ROOK]   = 0x0000000000000081ULL;
    pos.bb[W_KNIGHT] = 0x0000000000000042ULL;
    pos.bb[W_BISHOP] = 0x0000000000000024ULL;
    pos.bb[W_QUEEN]  = 0x0000000000000008ULL;
    pos.bb[W_KING]   = 0x0000000000000010ULL;

    // Black pieces
    pos.bb[B_PAWN]   = 0x00FF000000000000ULL;
    pos.bb[B_ROOK]   = 0x8100000000000000ULL;
    pos.bb[B_KNIGHT] = 0x4200000000000000ULL;
    pos.bb[B_BISHOP] = 0x2400000000000000ULL;
    pos.bb[B_QUEEN]  = 0x0800000000000000ULL;
    pos.bb[B_KING]   = 0x1000000000000000ULL;

    pos.castlingRights = Position::WK_CASTLE_MASK | Position::WQ_CASTLE_MASK |
                         Position::BK_CASTLE_MASK | Position::BQ_CASTLE_MASK;
    pos.whiteToMove = true;
    pos.epSquare = -1;
    pos.halfmoveClock = 0;
    pos.fullmoveNumber = 1;
    pos.updateOccupancies();
    pos.syncMailboxFromBitboards();
    pos.computeAndSetHash();

    // --- Game history ---
    std::vector<Position> positions;
    std::vector<uint64_t> history_hashes;

    positions.push_back(pos);
    history_hashes.push_back(pos.currentHash);

    int moveCount = 0;



    while (true) {
        if (verbose) {
            std::cout << "Move #" << moveCount + 1
                      << " (" << (pos.whiteToMove ? "White" : "Black") << "):\n";
            std::cout << pos.pretty() << "\n";
        }
        std::vector<Move> moves;
        engine.generateMoves(pos, moves);
        Move bestMove = engine.findBestMove(pos, history_hashes);

        //just for testing restore later*****************************************************
        //Move bestMove = moves[0];
        if (bestMove == 0) {
            // No legal moves — get outcome and break
            std::cout << "No legal moves found. Breaking...\n";
            break;
        }

        engine.applyMove(pos, bestMove);
        pos.computeAndSetHash();

        positions.push_back(pos);
        history_hashes.push_back(pos.currentHash);

        moveCount++;

        // Optional safety stop
        if (moveCount > 400) {
            std::cout << "Game aborted (too long)." << std::endl;
            break;
        }

        // Check for terminal outcome
        Outcome out = engine.currentOutcome(pos, history_hashes);
        if (out != Outcome::ONGOING) {
            if (verbose) std::cout << "Game ended: " << static_cast<int>(out) << "\n";
            return {positions, out};
        }
    }

    Outcome finalOutcome = engine.currentOutcome(pos, history_hashes);
    return {positions, finalOutcome};
}


std::string outcomeToString(Outcome o) {
    switch (o) {
        case Outcome::ONGOING: return "Ongoing";
        case Outcome::CHECKMATE: return "Checkmate";
        case Outcome::STALEMATE: return "Stalemate";
        case Outcome::DRAW_FIFTY_MOVE: return "Draw (50-move rule)";
        case Outcome::DRAW_THREEFOLD_REPETITION: return "Draw (threefold repetition)";
        default: return "Unknown";
    }
}

int evaluateFEN(const std::string& fen, int depth) {
    Engine engine(depth);
    std::cout << "Evaluating FEN: " << fen << " at depth " << depth << std::endl;
    Position pos;

    // --- Parse FEN ---
    std::istringstream ss(fen);
    std::string boardPart, sideToMove, castling, enpassant;
    int halfmoveClock = 0, fullmoveNumber = 1;
    ss >> boardPart >> sideToMove >> castling >> enpassant >> halfmoveClock >> fullmoveNumber;

    // Reset bitboards
    for (int i = 0; i < 12; ++i) pos.bb[i] = 0ULL;

    int square = 56; // rank 8, file a
    for (char c : boardPart) {
        if (c == '/') {
            square -= 16;
        } else if (isdigit(c)) {
            square += (c - '0');
        } else {
            switch (c) {
                case 'P': pos.bb[W_PAWN]   |= 1ULL << square; break;
                case 'N': pos.bb[W_KNIGHT] |= 1ULL << square; break;
                case 'B': pos.bb[W_BISHOP] |= 1ULL << square; break;
                case 'R': pos.bb[W_ROOK]   |= 1ULL << square; break;
                case 'Q': pos.bb[W_QUEEN]  |= 1ULL << square; break;
                case 'K': pos.bb[W_KING]   |= 1ULL << square; break;
                case 'p': pos.bb[B_PAWN]   |= 1ULL << square; break;
                case 'n': pos.bb[B_KNIGHT] |= 1ULL << square; break;
                case 'b': pos.bb[B_BISHOP] |= 1ULL << square; break;
                case 'r': pos.bb[B_ROOK]   |= 1ULL << square; break;
                case 'q': pos.bb[B_QUEEN]  |= 1ULL << square; break;
                case 'k': pos.bb[B_KING]   |= 1ULL << square; break;
                default: break;
            }
            square++;
        }
    }

    // --- Side to move ---
    pos.whiteToMove = (sideToMove == "w");

    // --- Castling rights ---
    pos.castlingRights = 0;
    if (castling.find('K') != std::string::npos) pos.castlingRights |= Position::WK_CASTLE_MASK;
    if (castling.find('Q') != std::string::npos) pos.castlingRights |= Position::WQ_CASTLE_MASK;
    if (castling.find('k') != std::string::npos) pos.castlingRights |= Position::BK_CASTLE_MASK;
    if (castling.find('q') != std::string::npos) pos.castlingRights |= Position::BQ_CASTLE_MASK;

    // --- En passant square ---
    if (enpassant != "-" && enpassant.size() == 2) {
        int file = enpassant[0] - 'a';
        int rank = enpassant[1] - '1';
        pos.epSquare = rank * 8 + file;
    } else {
        pos.epSquare = -1;
    }

    pos.halfmoveClock = halfmoveClock;
    pos.fullmoveNumber = fullmoveNumber;
    pos.updateOccupancies();
    pos.syncMailboxFromBitboards();
    pos.computeAndSetHash();

    // --- Prepare search bookkeeping ---
    std::vector<uint64_t> search_path_history;
    std::vector<uint64_t> game_history_hashes;

    // --- Perform search using negamax ---
    int alpha = -300000;
    int beta =  300000;
    int score = engine.negamax(pos, depth, alpha, beta, search_path_history, game_history_hashes, true);
    //int score = engine.findBestMove(pos, game_history_hashes);


    // --- Debug Output ---
    std::cout << "FEN: " << fen << "\n";
    std::cout << "White to move: " << pos.whiteToMove << "\n";
    std::cout << "Depth: " << depth << "\n";
    std::cout << "Eval score: " << score << " (positive = White better, negative = Black better)\n";

    return score;
}



}


