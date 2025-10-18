from flask import Flask, request, jsonify
from flask_cors import CORS
import chessbridge # This is your compiled C++ module
import os
import traceback # For detailed error logging
import time # For timing
import logging
app = Flask(__name__)
CORS(app)

PIECE_CHAR_TO_CPP_ENUM_INDEX = {
    'P': chessbridge.W_PAWN, 'N': chessbridge.W_KNIGHT,
    'B': chessbridge.W_BISHOP, 'R': chessbridge.W_ROOK,
    'Q': chessbridge.W_QUEEN, 'K': chessbridge.W_KING,
    'p': chessbridge.B_PAWN, 'n': chessbridge.B_KNIGHT,
    'b': chessbridge.B_BISHOP, 'r': chessbridge.B_ROOK,
    'q': chessbridge.B_QUEEN, 'k': chessbridge.B_KING,
}

CPP_PROMO_TYPE_TO_CHAR = {
    getattr(chessbridge.PromoPieceType, "PROMO_TYPE_N").value: 'N',
    getattr(chessbridge.PromoPieceType, "PROMO_TYPE_B").value: 'B',
    getattr(chessbridge.PromoPieceType, "PROMO_TYPE_R").value: 'R',
    getattr(chessbridge.PromoPieceType, "PROMO_TYPE_Q").value: 'Q'
}

# Constants for C++ move flags (as per chess_engine.hpp, after being shifted down)
# These correspond to the bits in the 4-bit flags field (m >> 16) & 0xF
EP_FLAG_CPP  = 1<<0 # En Passant
DPP_FLAG_CPP = 1<<1 # Double Pawn Push
KSC_FLAG_CPP = 1<<2 # King Side Castle
QSC_FLAG_CPP = 1<<3 # Queen Side Castle


def _fen_char_to_piece_enum(char_piece):
    return PIECE_CHAR_TO_CPP_ENUM_INDEX.get(char_piece)

def _parse_fen_to_cpp_position(fen_string):
    """
    Parses a FEN string and populates a chessbridge.Position object.
    Returns the Position object or None if FEN is invalid.
    """
    try:
        pos = chessbridge.Position()
        parts = fen_string.split(" ")
        if len(parts) < 4:
            print(f"[PYTHON ERROR] Invalid FEN (not enough parts): {fen_string}")
            return None

        # 1. Piece Placement
        piece_placement = parts[0]
        temp_bb_list = [0] * 12 # Initialize with 12 zeroes
        fen_rank_iter = 0
        fen_file_iter = 0

        for char_code in piece_placement:
            if char_code == '/':
                fen_rank_iter += 1
                fen_file_iter = 0
            elif '1' <= char_code <= '8':
                fen_file_iter += int(char_code)
            else:
                cpp_piece_enum = _fen_char_to_piece_enum(char_code)
                if cpp_piece_enum is not None and fen_rank_iter < 8 and fen_file_iter < 8:
                    cpp_internal_rank = 7 - fen_rank_iter
                    cpp_sq = cpp_internal_rank * 8 + fen_file_iter
                    temp_bb_list[cpp_piece_enum.value] |= (1 << cpp_sq)
                fen_file_iter += 1
        pos.bb = temp_bb_list


        # 2. Side to move
        pos.whiteToMove = (parts[1] == 'w')

        # 3. Castling availability
        js_castling_rights_fen_str = parts[2]
        cpp_castling = 0
        if hasattr(chessbridge.Position, "WK_CASTLE_MASK"):
            if 'K' in js_castling_rights_fen_str: cpp_castling |= chessbridge.Position.WK_CASTLE_MASK
            if 'Q' in js_castling_rights_fen_str: cpp_castling |= chessbridge.Position.WQ_CASTLE_MASK
            if 'k' in js_castling_rights_fen_str: cpp_castling |= chessbridge.Position.BK_CASTLE_MASK
            if 'q' in js_castling_rights_fen_str: cpp_castling |= chessbridge.Position.BQ_CASTLE_MASK
        else:
            if 'K' in js_castling_rights_fen_str: cpp_castling |= 0b0001
            if 'Q' in js_castling_rights_fen_str: cpp_castling |= 0b0010
            if 'k' in js_castling_rights_fen_str: cpp_castling |= 0b0100
            if 'q' in js_castling_rights_fen_str: cpp_castling |= 0b1000
        pos.castlingRights = cpp_castling


        # 4. En passant target square
        js_ep_square_algebraic = parts[3]
        if js_ep_square_algebraic and js_ep_square_algebraic != '-':
            if len(js_ep_square_algebraic) == 2:
                file_char = js_ep_square_algebraic[0].lower()
                rank_char = js_ep_square_algebraic[1]
                cpp_ep_file = ord(file_char) - ord('a')
                cpp_ep_rank = int(rank_char) - 1
                if 0 <= cpp_ep_file < 8 and 0 <= cpp_ep_rank < 8:
                    pos.epSquare = cpp_ep_rank * 8 + cpp_ep_file
                else: pos.epSquare = -1
            else: pos.epSquare = -1
        else: pos.epSquare = -1

        # 5. Halfmove clock
        pos.halfmoveClock = int(parts[4]) if len(parts) > 4 else 0
        # 6. Fullmove number
        pos.fullmoveNumber = int(parts[5]) if len(parts) > 5 else 1

        pos.updateOccupancies()
        pos.syncMailboxFromBitboards() # *** NEW: Sync mailbox after setting up board ***
        pos.computeAndSetHash()
        return pos
    except Exception as e:
        print(f"[PYTHON ERROR] Failed to parse FEN string '{fen_string}': {e}")
        traceback.print_exc()
        return None


def board_js_to_cpp_position(js_board_state, js_player_color, js_castling_rights_fen_str, js_ep_square_algebraic, js_halfmove_clock, js_fullmove_number):
    pos = chessbridge.Position()
    temp_bb_list = [0] * 12
    for r_js in range(8):
        for c_js in range(8):
            char_piece = js_board_state[r_js][c_js]
            if char_piece:
                cpp_piece_enum = PIECE_CHAR_TO_CPP_ENUM_INDEX.get(char_piece)
                if cpp_piece_enum is not None:
                    cpp_internal_rank = 7 - r_js
                    cpp_sq = cpp_internal_rank * 8 + c_js
                    try:
                        temp_bb_list[cpp_piece_enum.value] |= (1 << cpp_sq)
                    except IndexError:
                        print(f"[PYTHON ERROR] IndexError in board_js_to_cpp_position for piece enum value: {cpp_piece_enum.value}")
    pos.bb = temp_bb_list
    pos.whiteToMove = (js_player_color == 'white')

    cpp_castling = 0
    if hasattr(chessbridge.Position, "WK_CASTLE_MASK"):
        if 'K' in js_castling_rights_fen_str: cpp_castling |= chessbridge.Position.WK_CASTLE_MASK
        if 'Q' in js_castling_rights_fen_str: cpp_castling |= chessbridge.Position.WQ_CASTLE_MASK
        if 'k' in js_castling_rights_fen_str: cpp_castling |= chessbridge.Position.BK_CASTLE_MASK
        if 'q' in js_castling_rights_fen_str: cpp_castling |= chessbridge.Position.BQ_CASTLE_MASK
    else:
        if 'K' in js_castling_rights_fen_str: cpp_castling |= 0b0001
        if 'Q' in js_castling_rights_fen_str: cpp_castling |= 0b0010
        if 'k' in js_castling_rights_fen_str: cpp_castling |= 0b0100
        if 'q' in js_castling_rights_fen_str: cpp_castling |= 0b1000
    pos.castlingRights = cpp_castling


    if js_ep_square_algebraic and js_ep_square_algebraic != '-':
        if len(js_ep_square_algebraic) == 2:
            file_char = js_ep_square_algebraic[0].lower()
            rank_char = js_ep_square_algebraic[1]
            cpp_ep_file = ord(file_char) - ord('a')
            cpp_ep_rank = int(rank_char) - 1
            if 0 <= cpp_ep_file < 8 and 0 <= cpp_ep_rank < 8:
                pos.epSquare = cpp_ep_rank * 8 + cpp_ep_file
            else: pos.epSquare = -1
        else: pos.epSquare = -1
    else: pos.epSquare = -1

    pos.halfmoveClock = int(js_halfmove_clock)
    pos.fullmoveNumber = int(js_fullmove_number)
    pos.updateOccupancies()
    pos.syncMailboxFromBitboards() # *** NEW: Sync mailbox after setting up board ***
    pos.computeAndSetHash()
    return pos

def decode_cpp_move_to_js(cpp_move_int, player_who_moved_color):
    """
    Decodes a C++ move integer into a dictionary suitable for JSON response to JavaScript.
    Includes decoding of en passant and castling flags.
    """
    if cpp_move_int == 0:
        return None

    # Extract parts of the move integer based on C++ encoding
    # fromSquare: bits 0-5
    # toSquare:   bits 6-11
    # promotion:  bits 12-15 (Note: C++ uses 4 bits, 0-4 for promo type, so (m>>12)&0xF is okay)
    # moveFlags:  bits 16-19 (Note: C++ uses 4 bits, so (m>>16)&0xF is correct)
    from_sq_cpp = cpp_move_int & 0x3F
    to_sq_cpp    = (cpp_move_int >> 6) & 0x3F
    promo_type_cpp_val = (cpp_move_int >> 12) & 0xF # 4 bits for promotion type
    move_flags_cpp_val = (cpp_move_int >> 16) & 0xF # 4 bits for flags

    # Convert C++ square indices (0-63, a1=0) to JS visual/actual board indices (0-7 for row/col)
    # C++ internal rank 0-7 (a1-h1 to a8-h8)
    # JS visual row 0-7 (rank 8 down to 1 if not flipped, or actualBoardRow)
    from_r_cpp = from_sq_cpp // 8
    from_c_cpp = from_sq_cpp % 8
    to_r_cpp   = to_sq_cpp // 8
    to_c_cpp   = to_sq_cpp % 8

    # Convert C++ internal rank to JS actualBoardRow (0=rank8, 7=rank1)
    from_r_js = 7 - from_r_cpp
    to_r_js   = 7 - to_r_cpp
    # File (column) is the same for C++ internal and JS actualBoardCol (0=a, 7=h)

    promo_char_js = None
    # Accessing PromoPieceType.PROMO_TYPE_NONE.value correctly
    promo_type_none_val = getattr(chessbridge.PromoPieceType, "PROMO_TYPE_NONE").value
    if promo_type_cpp_val != promo_type_none_val and promo_type_cpp_val != 0: # 0 might be an uninitialized value
        base_promo_char = CPP_PROMO_TYPE_TO_CHAR.get(promo_type_cpp_val)
        if base_promo_char:
            # Promotion piece character should match the color of the player who moved
            promo_char_js = base_promo_char.upper() if player_who_moved_color == 'white' else base_promo_char.lower()

    # Decode flags
    is_en_passant = bool(move_flags_cpp_val & EP_FLAG_CPP)
    # is_double_pawn_push = bool(move_flags_cpp_val & DPP_FLAG_CPP) # Can be added if needed by JS

    castling_type_js = None
    if move_flags_cpp_val & KSC_FLAG_CPP:
        castling_type_js = 'kingSide'
    elif move_flags_cpp_val & QSC_FLAG_CPP:
        castling_type_js = 'queenSide'

    return {
        'fromSq': {'row': from_r_js, 'col': from_c_cpp}, # These are actualBoardRow/Col for JS
        'toSq':   {'row': to_r_js,   'col': to_c_cpp},   # These are actualBoardRow/Col for JS
        'promotionPiece': promo_char_js,
        'isEnPassant': is_en_passant,                   # NEW: Pass en passant flag
        'castling': castling_type_js                     # NEW: Pass castling type
        # 'isDoublePawnPush': is_double_pawn_push       # Optional
    }

# Store engine and its depth globally
ENGINE_SEARCH_DEPTH = 4 # Example depth
engine = None
try:
    engine = chessbridge.Engine(ENGINE_SEARCH_DEPTH)
    print(f"C++ Chess Engine initialized successfully for web app with depth: {ENGINE_SEARCH_DEPTH}.")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to initialize chessbridge.Engine: {e}")
    traceback.print_exc()


@app.route('/get-ai-move', methods=['POST'])
def get_ai_move_route():
    if not engine:
        return jsonify({'error': 'AI Engine not initialized. Check server logs.'}), 500
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data received'}), 400

        js_board_state = data.get('board_state')
        ai_player_color = data.get('player_color') # This is the AI's color (whose turn it is)
        js_castling_rights_fen_str = data.get('castling_rights')
        js_ep_square_algebraic = data.get('en_passant_target')
        js_halfmove_clock = data.get('halfmove_clock', 0)
        js_fullmove_number = data.get('fullmove_number', 1)
        js_fen_history_strings = data.get('fen_history', [])

        missing_fields = []
        if js_board_state is None: missing_fields.append("board_state")
        if ai_player_color is None: missing_fields.append("player_color")
        if js_castling_rights_fen_str is None: missing_fields.append("castling_rights")
        if js_ep_square_algebraic is None: missing_fields.append("en_passant_target")
        if missing_fields:
            return jsonify({'error': f'Missing fields in request: {", ".join(missing_fields)}'}), 400

        cpp_current_position = board_js_to_cpp_position(
            js_board_state, ai_player_color, # Pass AI's color as current player to C++
            js_castling_rights_fen_str, js_ep_square_algebraic,
            js_halfmove_clock, js_fullmove_number
        )
        if not cpp_current_position:
             print("[PYTHON ERROR] Failed to create C++ position from JS board state in /get-ai-move.")
             return jsonify({'error': 'Failed to create C++ position from current board state.'}), 500

        cpp_history_hashes = []
        for fen_str in js_fen_history_strings:
            hist_pos = _parse_fen_to_cpp_position(fen_str)
            if hist_pos:
                cpp_history_hashes.append(hist_pos.currentHash)

        start_time = time.perf_counter()
        # findBestMove expects the position for the player whose turn it is.
        best_move_cpp_int = engine.findBestMove(cpp_current_position, cpp_history_hashes)
        end_time = time.perf_counter()
        duration_s = end_time - start_time

        nodes_evaluated = 0
        try:
            nodes_evaluated = engine.getNodesVisited()
        except AttributeError:
            print("[PYTHON WARNING] engine.getNodesVisited() not available. Ensure C++ bindings are updated.")
            nodes_evaluated = "N/A"

        print(f"Depth: {ENGINE_SEARCH_DEPTH}")
        print(f"Positions Evaluated: {nodes_evaluated}")
        print(f"Time: {duration_s:.4f} seconds")


        if best_move_cpp_int == 0: # No legal moves
            outcome = engine.currentOutcome(cpp_current_position, cpp_history_hashes)
            outcome_map = {
                chessbridge.Outcome.CHECKMATE: f'AI ({ai_player_color}) is Checkmated.',
                chessbridge.Outcome.STALEMATE: f'AI ({ai_player_color}) is Stalemated.',
                chessbridge.Outcome.DRAW_THREEFOLD_REPETITION: f'Draw by Threefold Repetition for AI ({ai_player_color}).',
                chessbridge.Outcome.DRAW_FIFTY_MOVE: f'Draw by Fifty Move Rule for AI ({ai_player_color}).',
                chessbridge.Outcome.ONGOING: f'AI ({ai_player_color}) has no legal moves (Ongoing but no moves?).' # Should ideally be caught by checkmate/stalemate
            }
            message = outcome_map.get(outcome, f'AI ({ai_player_color}) has no legal moves or unspecified outcome.')
            return jsonify({'message': message}), 200

        # Pass ai_player_color to decode_cpp_move_to_js for correct promotion piece casing
        ai_move_js_format = decode_cpp_move_to_js(best_move_cpp_int, ai_player_color)

        if ai_move_js_format:
            return jsonify(ai_move_js_format), 200
        else:
            return jsonify({'error': 'Failed to decode AI move from C++'}), 500

    except Exception as e:
        print(f"Error in /get-ai-move route: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Unhandled error in AI move calculation: {str(e)}'}), 500

if __name__ == '__main__':
    werkzeug_logger = logging.getLogger('werkzeug')
    werkzeug_logger.setLevel(logging.ERROR)

    if not engine:
        print("Flask app will run, but AI functionality (get-ai-move) will be broken due to chessbridge.Engine initialization failure.")

    run_perft_test = False # Set to True to run perft on startup
    fen_to_test = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"

    if run_perft_test:
        print("\n--- Running Perft Test ---")
        perft_engine = chessbridge.Engine(1) # Depth for engine doesn't matter for perft method itself
        perft_depth_to_run = 5 # Actual perft depth

        print(f"Testing FEN: {fen_to_test}")
        print(f"Perft Depth: {perft_depth_to_run}")

        cpp_pos_for_perft = _parse_fen_to_cpp_position(fen_to_test)

        if cpp_pos_for_perft:
            start_time_perft = time.perf_counter()
            try:
                node_count = perft_engine.perft(cpp_pos_for_perft, perft_depth_to_run)
                end_time_perft = time.perf_counter()
                duration_perft = end_time_perft - start_time_perft
                print(f"Perft({perft_depth_to_run}) Node Count: {node_count}")
                print(f"Time taken: {duration_perft:.4f} seconds")
                if duration_perft > 0:
                    print(f"Nodes per second: {node_count / duration_perft:.2f}")

            except AttributeError as ae:
                print(f"AttributeError during perft: {ae}. Is 'perft' method bound correctly in chessbridge for the Engine class?")
                traceback.print_exc()
            except Exception as e:
                print(f"Error during perft test: {e}")
                traceback.print_exc()
        else:
            print(f"Failed to parse FEN for perft test: {fen_to_test}")
        print("--- Perft Test Finished ---")

    # Ensure Flask server runs even if perft is enabled, but not when Flask auto-reloads
    if not run_perft_test or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        print("Starting Flask web server...")
        app.run(debug=True, port=5000)