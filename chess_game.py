import pygame
import sys
from copy import deepcopy


WIDTH, HEIGHT = 640, 640
ROWS, COLS = 8, 8
SQUARE_SIZE = WIDTH // COLS

FPS = 30

# Piece notation (simplified):
#   Uppercase = White
#   Lowercase = Black
#   r/R = Rook, n/N = Knight, b/B = Bishop, q/Q = Queen,
#   k/K = King, p/P = Pawn
# Initial board in FEN-like arrangement for demonstration:
STARTING_BOARD = [
    ["r", "n", "b", "q", "k", "b", "n", "r"],
    ["p", "p", "p", "p", "p", "p", "p", "p"],
    [" ", " ", " ", " ", " ", " ", " ", " "],
    [" ", " ", " ", " ", " ", " ", " ", " "],
    [" ", " ", " ", " ", " ", " ", " ", " "],
    [" ", " ", " ", " ", " ", " ", " ", " "],
    ["P", "P", "P", "P", "P", "P", "P", "P"],
    ["R", "N", "B", "Q", "K", "B", "N", "R"]
]

# Color definitions
LIGHT_COLOR = (240, 217, 181)
DARK_COLOR = (181, 136, 99)
HIGHLIGHT_COLOR = (186, 202, 68)

# --------------------------------------------------------------------------------
# Chess Mechanics (Simplified)
# --------------------------------------------------------------------------------

def is_white(piece):
    return piece.isupper()

def is_black(piece):
    return piece.islower()

def in_bounds(r, c):
    return 0 <= r < ROWS and 0 <= c < COLS

def get_piece_moves(board, r, c):
    """
    Return all possible (r2, c2) moves for piece at board[r][c].
    This is a simplified version: it does not fully check for pinned pieces,
    does not handle castling, etc. 
    """
    moves = []
    piece = board[r][c]
    if piece == " ":
        return moves  # No piece here

    # Directions for each piece type
    if piece.upper() == "P":
        # Pawns move differently for white vs black
        direction = -1 if is_white(piece) else 1
        start_row = 6 if is_white(piece) else 1
        # Single step
        forward_r = r + direction
        if in_bounds(forward_r, c) and board[forward_r][c] == " ":
            moves.append((forward_r, c))
            # Double step
            if r == start_row:
                double_r = r + 2 * direction
                if board[double_r][c] == " ":
                    moves.append((double_r, c))
        # Captures
        for dc in [-1, 1]:
            cap_r, cap_c = r + direction, c + dc
            if in_bounds(cap_r, cap_c) and board[cap_r][cap_c] != " ":
                if is_white(piece) and is_black(board[cap_r][cap_c]):
                    moves.append((cap_r, cap_c))
                elif is_black(piece) and is_white(board[cap_r][cap_c]):
                    moves.append((cap_r, cap_c))

    elif piece.upper() == "R":
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            while in_bounds(nr, nc):
                if board[nr][nc] == " ":
                    moves.append((nr, nc))
                else:
                    if (is_white(piece) and is_black(board[nr][nc])) \
                       or (is_black(piece) and is_white(board[nr][nc])):
                        moves.append((nr, nc))
                    break
                nr += dr
                nc += dc

    elif piece.upper() == "N":
        knight_moves = [
            (r+2, c+1), (r+2, c-1), (r-2, c+1), (r-2, c-1),
            (r+1, c+2), (r+1, c-2), (r-1, c+2), (r-1, c-2)
        ]
        for (nr, nc) in knight_moves:
            if in_bounds(nr, nc):
                if board[nr][nc] == " " or (is_white(piece) and is_black(board[nr][nc])) \
                   or (is_black(piece) and is_white(board[nr][nc])):
                    moves.append((nr, nc))

    elif piece.upper() == "B":
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            while in_bounds(nr, nc):
                if board[nr][nc] == " ":
                    moves.append((nr, nc))
                else:
                    if (is_white(piece) and is_black(board[nr][nc])) \
                       or (is_black(piece) and is_white(board[nr][nc])):
                        moves.append((nr, nc))
                    break
                nr += dr
                nc += dc

    elif piece.upper() == "Q":
        # Queen = Rook + Bishop moves
        moves += get_piece_moves(board, r, c, )  # We'll do a trick: just combine.
        # But let's do it explicitly here for clarity:
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1), 
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            while in_bounds(nr, nc):
                if board[nr][nc] == " ":
                    moves.append((nr, nc))
                else:
                    if (is_white(piece) and is_black(board[nr][nc])) \
                       or (is_black(piece) and is_white(board[nr][nc])):
                        moves.append((nr, nc))
                    break
                nr += dr
                nc += dc

    elif piece.upper() == "K":
        king_moves = [
            (r+1, c), (r-1, c), (r, c+1), (r, c-1),
            (r+1, c+1), (r+1, c-1), (r-1, c+1), (r-1, c-1)
        ]
        for (nr, nc) in king_moves:
            if in_bounds(nr, nc):
                if board[nr][nc] == " " or (is_white(piece) and is_black(board[nr][nc])) \
                   or (is_black(piece) and is_white(board[nr][nc])):
                    moves.append((nr, nc))

    return moves

def make_move(board, r1, c1, r2, c2):
    """
    Move piece from (r1, c1) to (r2, c2). Return a *new* board (copy) so we
    can safely use it in minimax without mutating the original.
    """
    new_board = deepcopy(board)
    piece = new_board[r1][c1]
    new_board[r1][c1] = " "
    new_board[r2][c2] = piece
    return new_board

def evaluate_board(board):
    """
    Basic material evaluation:
    Positive if advantage White, negative if advantage Black.
    """
    values = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 100}
    score = 0
    for row in board:
        for piece in row:
            if piece == " ":
                continue
            val = values.get(piece.upper(), 0)
            if is_white(piece):
                score += val
            else:
                score -= val
    return score

def get_all_moves(board, white_to_move):
    """
    Get all possible moves for the current side (white or black).
    Returns list of (r1, c1, r2, c2).
    """
    moves = []
    for r in range(ROWS):
        for c in range(COLS):
            piece = board[r][c]
            if piece == " ":
                continue
            if white_to_move and is_white(piece):
                for (r2, c2) in get_piece_moves(board, r, c):
                    moves.append((r, c, r2, c2))
            elif not white_to_move and is_black(piece):
                for (r2, c2) in get_piece_moves(board, r, c):
                    moves.append((r, c, r2, c2))
    return moves

def minimax(board, depth, alpha, beta, white_to_move):
    """
    Minimax with alpha-beta pruning.
    """
    if depth == 0:
        return evaluate_board(board), None

    moves = get_all_moves(board, white_to_move)
    if not moves:
        # No moves available: check for checkmate or stalemate
        return evaluate_board(board), None

    if white_to_move:
        max_eval = float('-inf')
        best_move = None
        for move in moves:
            (r1, c1, r2, c2) = move
            new_board = make_move(board, r1, c1, r2, c2)
            evaluation, _ = minimax(new_board, depth-1, alpha, beta, False)
            if evaluation > max_eval:
                max_eval = evaluation
                best_move = move
            alpha = max(alpha, evaluation)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        best_move = None
        for move in moves:
            (r1, c1, r2, c2) = move
            new_board = make_move(board, r1, c1, r2, c2)
            evaluation, _ = minimax(new_board, depth-1, alpha, beta, True)
            if evaluation < min_eval:
                min_eval = evaluation
                best_move = move
            beta = min(beta, evaluation)
            if beta <= alpha:
                break
        return min_eval, best_move

# --------------------------------------------------------------------------------
# Pygame Rendering and Main Loop
# --------------------------------------------------------------------------------

def draw_board(screen, board, selected_square):
    for r in range(ROWS):
        for c in range(COLS):
            color = LIGHT_COLOR if (r + c) % 2 == 0 else DARK_COLOR
            pygame.draw.rect(screen, color, (c*SQUARE_SIZE, r*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
    
    # Highlight selected square
    if selected_square is not None:
        (sr, sc) = selected_square
        pygame.draw.rect(
            screen, HIGHLIGHT_COLOR,
            (sc*SQUARE_SIZE, sr*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
        )

    # Draw pieces (very basic text)
    font = pygame.font.SysFont(None, 36)
    for r in range(ROWS):
        for c in range(COLS):
            piece = board[r][c]
            if piece != " ":
                text_surface = font.render(piece, True, (0, 0, 0))
                screen.blit(text_surface, (c*SQUARE_SIZE + 10, r*SQUARE_SIZE + 10))

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    pygame.display.set_caption("AI Chess Demo")

    board = deepcopy(STARTING_BOARD)
    white_to_move = True

    # Mode selection: 
    #   - user_is_white=True, user_is_black=False => Human vs. AI
    #   - user_is_white=False => AI vs. AI or White is AI
    user_is_white = True  
    user_is_black = False  
    ai_depth = 3  # Increase for stronger but slower AI

    selected_square = None
    running = True

    while running:
        clock.tick(FPS)
        screen.fill((0, 0, 0))

        # --- Check for events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if (white_to_move and user_is_white) or (not white_to_move and user_is_black):
                    x, y = event.pos
                    r, c = y // SQUARE_SIZE, x // SQUARE_SIZE
                    if selected_square is None:
                        # Select a piece
                        if board[r][c] != " ":
                            # Check correct side's piece
                            if (white_to_move and is_white(board[r][c])) or (not white_to_move and is_black(board[r][c])):
                                selected_square = (r, c)
                    else:
                        # Move attempt
                        (sr, sc) = selected_square
                        moves = get_piece_moves(board, sr, sc)
                        if (r, c) in moves:
                            board = make_move(board, sr, sc, r, c)
                            white_to_move = not white_to_move
                        selected_square = None

        # --- If it's AI's turn, let AI move ---
        if ((white_to_move and not user_is_white) or 
            (not white_to_move and not user_is_black)):
            _, best_move = minimax(board, ai_depth, float('-inf'), float('inf'), white_to_move)
            if best_move is not None:
                (r1, c1, r2, c2) = best_move
                board = make_move(board, r1, c1, r2, c2)
            white_to_move = not white_to_move

        # --- If we are in AI-vs-AI mode, let the other side move as well ---
        if not user_is_white and not user_is_black:
            # AI vs AI
            _, best_move = minimax(board, ai_depth, float('-inf'), float('inf'), white_to_move)
            if best_move is not None:
                (r1, c1, r2, c2) = best_move
                board = make_move(board, r1, c1, r2, c2)
            white_to_move = not white_to_move

        # --- Draw everything ---
        draw_board(screen, board, selected_square)
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
