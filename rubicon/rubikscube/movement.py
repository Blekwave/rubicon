import rubikscube.rubikscube as rc

from collections import namedtuple, OrderedDict
from functools import partial

import numpy as np


def face_side_indices(side):
    """Lists the 2D indices for each side in a face of a certain side.

    The index tuples contained by the list can be used as indices for
    a 3D cube's face in order to return the list of facelets in that
    side of the face.

    e.g.:
    >>> a = np.array(range(9)).reshape((3,3))
    >>> a[(slice(None), 2)]  # right side
    array([2, 5, 8])

    Parameters:
    - side: side of the faces in the cube"""
    return [
        (0, slice(None)),  # up
        (slice(None), side - 1),  # right
        (side - 1, slice(None)),  # down
        (slice(None), 0)  # left
    ]


def rotate(cube, face, k=1):
    """Rotate a Rubik's Cube's face clockwise.

    Parameters:
    - cube: Rubik's Cube np.ndarray (in 3D form, not flat)
    - face: index of the face to be rotated
    - k: number of subsequent rotations to perform
         (1: clockwise, 2: 180o, 3: counterclockwise)"""

    assert cube.shape[0] == rc.FACES
    assert cube.shape[1] == cube.shape[2]

    side = cube.shape[1]

    # Rotate facelets of the face itself
    cube[face] = np.rot90(cube[face], 3)  # rot90 is ccw

    # Rotate vectors adjacent to the face
    adj_vector_indices = rc.adjacent_vector_indices()[face]

    prev_vector = cube[adj_vector_indices[-1]].copy()
    for vector_index in adj_vector_indices:
        prev_vector, cube[vector_index] = cube[vector_index].copy(), prev_vector

    if k > 1:
        rotate(cube, face, k - 1)


Move = namedtuple("Move", ("name", "function"))


def freeze_move(f):
    """Transform a 3D cube transformation function into a constant-time,
    equivalent function for a flat array-based Rubik's Cube.

    Parameters:
    - f: cube transformation function

    Returns: flat array transformation function equivalent to f"""
    cube_before = rc.gen_3d_cube()
    cube_after = cube_before.copy()

    f(cube_after)

    changed_indices = cube_after != cube_before

    before_indices = cube_before[changed_indices]
    after_indices = cube_after[changed_indices]

    def frozen(cube):
        cube[before_indices] = cube[after_indices]

    return frozen


def gen_moves():
    """Generate all valid moves for a cube.

    Returns an OrderedDict of move functions, indexed by their
    mnemonic move names."""
    rotations = [
        Move(name=name_fmt.format(face_name),
             function=partial(rotate, face=face, k=k))
        for face, face_name in enumerate(["U", "L", "F", "R", "B", "D"])
        for k, name_fmt in enumerate(["{}", "{}2", "{}'"], start=1)
    ]

    moves = OrderedDict()

    for move in rotations:
        f = freeze_move(move.function)
        f.__name__ = move.name
        moves[move.name] = f

    return moves


moves = gen_moves()
move_list = list(moves.values())

_move_str_to_id = {
    move_str: move_id
    for move_id, move_str in enumerate(moves)
}

def move_names_to_ids(move_names):
    """Transforms a list of move names to a list of integer ids.

    Parameters:
    - move_names: list or space-separated string of move names

    Returns a list of move ids (integers), corresponding to values in
    the move_list."""
    if issubclass(move_names, str):
        move_names = move_names.split()
    return [_move_str_to_id[move_name] for move_name in move_names]


def apply_moves(cube, move_ids):
    """Perform a series of moves onto a cube.

    Parameters:
    - cube: initial state of the cube
    - move_ids: identifiers of each move

    Returns a copy of the cube onto which the moves have been
    performed."""
    cube = cube.copy()
    for move_id in move_ids:
        move_list[move_id](cube)
    return cube
