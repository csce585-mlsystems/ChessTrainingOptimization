#pragma once
#include <string>
#include <vector>
#include "chess_engine.hpp"

namespace chess {

struct GameResult {
    std::vector<Position> positions;
    Outcome outcome;
};

// Plays a self-play game and returns all positions + final outcome
GameResult runSelfPlayGame(int searchDepth, bool verbose);

// Converts outcome enum to string
std::string outcomeToString(Outcome o);

int evaluateFEN(const std::string& fen, int depth);

} // namespace chess
