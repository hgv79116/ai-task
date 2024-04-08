# COMP30024 Artificial Intelligence, Semester 1 2024
# Project Part A: Single Player Tetress

import json
from .core import PlayerColor, Coord, PlaceAction, Direction, BOARD_N
from .utils import render_board
from collections import deque, defaultdict
from queue import PriorityQueue

from typing import Dict, List

from dataclasses import dataclass

BOARD_N = 11

def generateMoves():
    ROOT_MOVES = [
        PlaceAction(Coord(0, 0), Coord(0, 1), Coord(0, 2), Coord(0, 3)),
        PlaceAction(Coord(0, 0), Coord(0, 1), Coord(0, 2), Coord(1, 0)),
        PlaceAction(Coord(0, 0), Coord(0, 1), Coord(0, 2), Coord(1, 1)),
        PlaceAction(Coord(0, 0), Coord(0, 1), Coord(0, 2), Coord(1, 2)),
        PlaceAction(Coord(0, 0), Coord(0, 1), Coord(0, 2), Coord(1, 2)),
        PlaceAction(Coord(0, 0), Coord(0, 1), Coord(1, 1), Coord(1, 2)),
        PlaceAction(Coord(0, 0), Coord(0, 1), Coord(1, 0), Coord(1, 1))
    ]

    def rotate(move: PlaceAction) -> PlaceAction:
        return PlaceAction(
            *[Coord(coord.c, BOARD_N - 1 - coord.r) for coord in move.coords]
        )

    def translate(move: PlaceAction, x_translation: int, y_translation: int) -> PlaceAction:
        return PlaceAction(
            *[coord + Direction.Down * x_translation + \
                Direction.Right * y_translation for coord in move.coords]
        )

    move_strings = set()
    moves = []
    for root_move in ROOT_MOVES:
        untranslated_move = root_move
        for rotation_iter in range(4):
            untranslated_move = rotate(untranslated_move)
            for x_translation in range(BOARD_N):
                for y_translation in range(BOARD_N):
                    move = translate(untranslated_move, x_translation, y_translation)

                    move_string = str(move)

                    if move_string not in move_strings:
                        move_strings.add(move_string)
                        moves.append(move)

    return moves

MOVES = generateMoves()

def getColor(
    board: dict[Coord, PlayerColor],
    cell: Coord
) -> PlayerColor:
    if cell not in board:
        return None

    return board[cell]

def getClearedBoard(
    uncleared_board: dict[Coord, PlayerColor]
):
    row_count = defaultdict(int)
    col_count = defaultdict(int)

    for cell in uncleared_board:
        row_count[cell.r] += 1
        col_count[cell.c] += 1

    returned_board = dict(uncleared_board)

    for cell in returned_board:
        if row_count[cell.r] == BOARD_N or col_count[cell.c] == BOARD_N:
            del returned_board[cell]

    return returned_board

def getAdjCells(
    cell: Coord
):
    return [
        cell + Direction.Up,
        cell + Direction.Down,
        cell + Direction.Left,
        cell + Direction.Right
    ]

def apply(
    input_board: dict[Coord, PlayerColor],
    place_action: PlaceAction,
    first_action: bool = False
) -> dict[Coord, PlayerColor] | None:
    for cell in place_action.coords:
        if cell in input_board:
            return None

    adj_to_atleast_one = False
    for cell in place_action.coords:
        for adj_cell in getAdjCells(cell):
            if getColor(input_board, adj_cell) == PlayerColor.RED:
                adj_to_atleast_one = True

    if not adj_to_atleast_one and not first_action:
        return None

    returned_board = dict(input_board)
    for cell in place_action.coords:
        returned_board[cell] = PlayerColor.RED

    print(input_board, returned_board)

    return returned_board

def toInt(board: dict[Coord, PlayerColor]):
    flattened = ['0'] * (BOARD_N * BOARD_N)

    for (r, c), color in board.items():
        flattened[r * BOARD_N + c] = '1' if color == PlayerColor.RED else '2'

    return ''.join(flattened)

def intToBoard(board_int: int):
    board = {}

    index = 0
    for r in range(BOARD_N):
        for c in range(BOARD_N):
            color_str = board_int[r * BOARD_N + c]
            if color_str == '1':
                board[Coord(r, c)] = PlayerColor.RED
            elif color_str == '2':
                board[Coord(r, c)] = PlayerColor.BLUE

            index += 1

    return board


def getDistanceToTarget(
    input_board: dict[Coord, PlayerColor],
    target: Coord
):
    INF = 1e20

    dist = dict()
    board = input_board

    queue = deque()
    queue.append(target)
    dist[target] = 0

    while queue:
        current_cell = queue.pop()

        adj_cells = [
            current_cell + Direction.Up,
            current_cell + Direction.Down,
            current_cell + Direction.Left,
            current_cell + Direction.Right
        ]

        for adj_cell in adj_cells:
            visisted = adj_cell in dist
            is_blue = adj_cell in board and board[adj_cell] == PlayerColor.BLUE

            if visisted or is_blue:
                continue

            dist[adj_cell] = dist[current_cell] + 1
            queue.append(adj_cell)

    min_dist = INF
    for cell in board:
        for adj_cell in getAdjCells(cell):
            if getColor(board, adj_cell) == PlayerColor.RED:
                min_dist = min(min_dist, dist[adj_cell])

    return min_dist

def isTargetCleared(
    input_board: dict[Coord, PlayerColor],
    target: Coord
):
    return target not in input_board

def search(
    input_board: dict[Coord, PlayerColor],
    target: Coord
) -> list[PlaceAction] | None:
    """
    This is the entry point for your submission. You should modify this
    function to solve the search problem discussed in the Part A specification.
    See `core.py` for information on the types being used here.

    Parameters:
        `board`: a dictionary representing the initial board state, mapping
            coordinates to "player colours". The keys are `Coord` instances,
            and the values are `PlayerColor` instances.
        `target`: the target BLUE coordinate to remove from the board.

    Returns:
        A list of "place actions" as PlaceAction instances, or `None` if no
        solution is possible.
    """

    # The render_board() function is handy for debugging. It will print out a
    # board state in a human-readable format. If your terminal supports ANSI
    # codes, set the `ansi` flag to True to print a colour-coded version!

    starting_board = input_board
    starting_board_int = toInt(starting_board)
    to_be_expanded = PriorityQueue()

    actions: Dict[Dict[Coord, PlayerColor], List[PlaceAction]] = dict()
    dist = dict()

    dist[starting_board_int] = 0
    actions[starting_board_int] = []
    to_be_expanded.put((0, starting_board_int))

    while not to_be_expanded.empty():
        (estimated_solution, current_board_int) = to_be_expanded.get_nowait()
        current_board = intToBoard(current_board_int)

        print(render_board(current_board, target, ansi=False))


        for move in MOVES:
            next_board = apply(current_board, move)

            if next_board is None:
                continue

            print (render_board(next_board, target, ansi=False))

            if isTargetCleared(next_board, target):
                return actions[current_board]

            next_board_int = toInt(next_board)

            if next_board_int in dist:
                continue

            dist[next_board_int] = dist[current_board_int] + 1
            actions[next_board_int] = actions[current_board_int] + [move]
            estimated_next_board_solution = dist[next_board_int] + getDistanceToTarget(next_board, target) // 4

            to_be_expanded.put_nowait((estimated_next_board_solution, next_board_int))

    return None

    # Do some impressive AI stuff here to find the solution...
    # ...
    # ... (your solution goes here!)
    # ...

    # Here we're returning "hardcoded" actions as an example of the expected
    # output format. Of course, you should instead return the result of your
    # search algorithm. Remember: if no solution is possible for a given input,
    # return `None` instead of a list.
