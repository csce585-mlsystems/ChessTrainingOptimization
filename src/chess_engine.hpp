// chess_engine.hpp
#pragma once

#include <cstdint>
#include <vector>
#include <array>
#include <string>
#include <unordered_map>
#include <onnxruntime_cxx_api.h>
#include <memory>

namespace chess {

// ───────────────────────── Basic Types ─────────────────────────
using Bitboard = uint64_t;
using Move     = uint32_t;

// ───────────────────────── Piece constants ──────────────────────────
enum Piece {
    W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
    B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING,
    NO_PIECE
};

extern const std::array<int, 12> PIECE_VALUE;
extern const std::array<int, 6> POSITIVE_PIECE_VALUE;
extern const std::array<char, 13> PIECE_CHAR_REPR;

// ───────────────────────── Board Helpers ─────────────────────────
int file_of(int sq);
int rank_of(int sq);
int square(int r, int f);
bool on_board_rf(int r, int f);
bool on_board(int sq);

// ───────────────────────── Attack Generation ─────────────────────────
namespace attacks {
    extern std::array<Bitboard, 64> knight_attacks_table;
    extern std::array<Bitboard, 64> king_attacks_table;
    extern std::array<std::array<Bitboard, 64>, 2> pawn_attacks_table;

    extern std::array<Bitboard, 64> ray_n, ray_s, ray_e, ray_w;
    extern std::array<Bitboard, 64> ray_ne, ray_nw, ray_se, ray_sw;

    extern std::array<std::array<Bitboard, 64>, 64> line_between_bb;

    Bitboard get_ray_between(int sq1, int sq2);
    Bitboard get_line_through(int sq1, int sq2);
    void init();
    Bitboard get_rook_attacks(int sq, Bitboard blockers);
    Bitboard get_bishop_attacks(int sq, Bitboard blockers);
}

// ───────────────────────── Move Encoding ─────────────────────────
enum PromoPieceType { PROMO_TYPE_NONE, PROMO_TYPE_N, PROMO_TYPE_B, PROMO_TYPE_R, PROMO_TYPE_Q };

constexpr int EP_FLAG  = 1 << 0;
constexpr int DPP_FLAG = 1 << 1;
constexpr int KSC_FLAG = 1 << 2;
constexpr int QSC_FLAG = 1 << 3;

Move encodeMove(int f, int t, int promo_val = PROMO_TYPE_NONE, int flags = 0);
int fromSquare(Move m);
int toSquare(Move m);
int promotion(Move m);
int moveFlags(Move m);
std::string squareToAlgebraic(int sq);
std::string moveToString(Move m);

// ───────────────────────── Zobrist Hashing ───────────────────
struct ZobristKeys {
    std::array<std::array<uint64_t, 64>, 12> piece_square_keys;
    uint64_t black_to_move_key;
    std::array<uint64_t, 16> castling_keys;
    std::array<uint64_t, 8> ep_file_keys;
    ZobristKeys();
};
const ZobristKeys& getZobristKeys();

// ───────────────────────── Position ────────────────────────────────
struct Position {
    std::array<Bitboard, 12> bb{};
    std::array<Piece, 64> mailbox{};
    Bitboard occWhite = 0, occBlack = 0, occ = 0;
    uint8_t castlingRights = 0;
    static const uint8_t WK_CASTLE_MASK = 0b0001;
    static const uint8_t WQ_CASTLE_MASK = 0b0010;
    static const uint8_t BK_CASTLE_MASK = 0b0100;
    static const uint8_t BQ_CASTLE_MASK = 0b1000;
    int epSquare = -1;
    bool whiteToMove = true;
    int halfmoveClock = 0;
    int fullmoveNumber = 1;
    uint64_t currentHash = 0;
    mutable std::vector<float> tensorData;

    Position();
    void updateOccupancies();
    void syncMailboxFromBitboards();
    Piece piece_at(int sq) const;
    void computeAndSetHash();
    std::string pretty() const;
    std::vector<float> toTensor() const;
};

// ───────────────────────── Game Outcome ────────────────────────────
enum class Outcome {
    ONGOING,
    CHECKMATE,
    STALEMATE,
    DRAW_FIFTY_MOVE,
    DRAW_THREEFOLD_REPETITION
};

// ───────────────────────── Transposition Table ───────────────
enum class TTEntryType { NONE, EXACT, LOWER_BOUND, UPPER_BOUND };

struct TranspositionTableEntry {
    uint64_t zobristHash = 0;
    int score = 0;
    int depth = -1;
    Move bestMove = 0;
    TTEntryType type = TTEntryType::NONE;
};

// ───────────────────────── Engine ─────────────────────────────
class Engine {
public:
    explicit Engine(int depth);

    // Core
    Move findBestMove(Position& pos, const std::vector<uint64_t>& game_history_hashes);
    void applyMove(Position& pos, Move move) const;
    Outcome currentOutcome(const Position& pos, const std::vector<uint64_t>& game_history_hashes) const;
    void generateMoves(const Position& pos, std::vector<Move>& moves) const;

    // Utilities
    uint64_t getHashForPosition(const Position& pos);
    uint64_t perft(Position& pos, int depth);
    long getNodesVisited() const;
    float evaluateWithModel(const Position& p);
    int negamax(Position& p, int depth, int alpha, int beta,
                std::vector<uint64_t>& search_path_history,
                const std::vector<uint64_t>& game_history_hashes,
                bool isRootNode);

private:
    // Helpers
    void add_pawn_moves(const Position& p, int from_sq, std::vector<Move>& moves, Bitboard target_mask) const;
    void add_knight_moves(const Position& p, int from_sq, std::vector<Move>& moves, Bitboard target_mask) const;
    void add_sliding_moves(const Position& p, int from_sq, bool is_bishop, bool is_rook, std::vector<Move>& moves, Bitboard target_mask) const;
    void add_king_moves(const Position& p, int from_sq, std::vector<Move>& moves) const;

    bool isSquareAttacked(const Position& p, int sq, bool by_white) const;
    bool isSquareAttacked(const Position& p, int sq, bool by_white, Bitboard blockers) const;

    int evaluateMaterial(const Position& p) const;
    int scoreMove(const Position& p, Move m) const;
    int get_piece_type_value_index(Piece p) const;
    uint64_t perft_recursive(Position& p, int depth);
    Bitboard get_attackers(const Position& p, int sq, bool by_white) const;


    bool loadModel(const std::string& path);
    std::unique_ptr<Ort::Session> onnx_session;

    // Members
    int searchDepth;
    long nodes_visited_search = 0;
    std::unordered_map<uint64_t, TranspositionTableEntry> transposition_table;
};

} // namespace chess
