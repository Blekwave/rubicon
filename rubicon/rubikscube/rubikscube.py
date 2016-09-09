import sys
import numpy as np
from itertools import chain
from math import sqrt

"""
USEFUL READING

Common Rubik's Cube notation usually regards clockwise as the
default orientation. This implementation respects this convention.

The cube's facelets are stored in a simple array of length 54. It is
considered solved if these facelets, from 0 to 53,  are in ascending
order.

There are six essential moves for solving the 3x3 Rubik's Cube, which
are the clockwise rotation of each of the six faces. Other moves
often used can be made from these basic moves: counter-clockwise
rotation, 180-degree rotation, rotation of the middle row of a face
etc.

This representation also maintains an adjacency map which maps faces
to other faces which are adjacent, in clockwise order. This map is
stored as a simple 6-long vector of 4-long integer vectors, which is
indexed by face and side of adjacency. For example, adj[3][0] and
adj[3][2] refer to the sides which are to the top and bottom of side
#3.
"""

FACES = 6
SQR_SIDES = 4
SIDE = 3


def gen_cube():
    """Generate a solved, flat array Rubik's Cube.

    The facelets in the cube are indexed from 0 to 6 * side ** 2."""
    return np.array(range(FACES * SIDE ** 2))


def gen_3d_cube():
    """Generate a solved, 3D array Rubik's Cube.

    The facelets in the cube are indexed from 0 to 6 * side ** 2."""
    return gen_cube().reshape((FACES, SIDE, SIDE))


def print_3d_cube(cube, file=sys.stdout):
    """Prints a 3D cube to the terminal, in color."""
    color_fmt = "\033[{style};{color};{bgcolor}m[{string}]"
    colors = (color_fmt.format(style=1, color=color, bgcolor=40, string=string)
              for string, color in
              zip(("U", "L", "F", "R", "B", "D"),
                  (37, 35, 32, 31, 34, 33)))

    dims = len(cube.shape)
    if dims == 1:
        face_area = len(cube) / FACES
        side = int(sqrt(face_area))
        cube = cube.reshape(FACES, side, side)
    elif dims == 3:
        side = cube.shape[1]
    else:
        raise RuntimeError("Cube isn't 1D (flat) or 3D "
                           "(dims = {})".format(dims))

    fclts_per_side = side ** 2
    fclt_colors = {}
    for i, color in enumerate(colors):
        for facelet in range(i * fclts_per_side, (i + 1) * fclts_per_side):
            fclt_colors[facelet] = color

    def print_line(line, padding=0):
        print(" " * padding, end="", file=file)
        for facelet in line:
            string = fclt_colors[facelet]
            print(string, end="", file=file)
        print(file=file)

    up_face = cube[0]
    for row in up_face:
        print_line(row, padding=9)

    middle_faces = cube[1:5]
    for first_row, *other_rows in zip(*middle_faces):
        row = np.append(first_row, other_rows)
        print_line(row)

    down_face = cube[5]
    for row in down_face:
        print_line(row, padding=9)


def adjacent_faces():
    """Generates a list of lists of adjacent faces.

    The list of lists is indexed by face. The lists of adjacent faces
    are indexed by side of the face, in clockwise order."""

    up = 0
    left = 1
    front = 2
    right = 3
    back = 4
    down = 5
    return np.array([
        [back, right, front, left],  # up
        [up, front, down, back], # left
        [up, right, down, left], # front
        [up, back, down, front], # right
        [up, right, down, left], # back
        [front, right, back, left]  # down
    ])


def adjacency_sides():
    """Generates a list lists of adjacency sides.

    Given a face A and a side S of that face, to which face B is
    adjacent to face A, the adjacency side of face A's side S is the
    side of B which is adjacent to A.

    The list of lists is indexed by face. The lists of adjacency sides
    are indexed by side of the face clockwise.
    """
    adj_faces = adjacent_faces()
    adj_sides = np.zeros((FACES, SQR_SIDES), dtype=int)
    for face_a in range(FACES):
        for a_to_b, face_b in enumerate(adj_faces[face_a]):
            b_to_a = list(adj_faces[face_b]).index(face_a)
            adj_sides[face_b][b_to_a] = a_to_b
    return adj_sides


def adjacent_vector_indices():
    # Faces
    up = 0
    left = 1
    front = 2
    right = 3
    back = 4
    down = 5

    pos = slice(None)
    neg = slice(None, None, -1)

    return [
        [ # up
            (back, 0, pos),
            (right, 0, pos),
            (front, 0, pos),
            (left, 0, pos)
        ],
        [ # left
            (up, pos, 0),
            (front, pos, 0),
            (down, pos, 0),
            (back, 0, neg)
        ],
        [ # front
            (up, 2, pos),
            (right, pos, 0),
            (down, 0, neg),
            (left, neg, 2)
        ],
        [ # right
            (up, neg, 2),
            (back, pos, 0),
            (down, neg, 2),
            (front, neg, 2)
        ],
        [ # back
            (up, 0, neg),
            (right, pos, 0),
            (down, 2, pos),
            (left, neg, 2)
        ],
        [ # down
            (front, 2, pos),
            (right, 2, pos),
            (back, 2, pos),
            (left, 2, pos)
        ]
    ]


ADJACENT = adjacent_faces()
ADJACENCY = adjacency_sides()


CORNERS = [
    ((0, 0, 0), (1, 0, 0), (4, 0, 2)), # up left back
    ((0, 0, 2), (3, 0, 2), (4, 0, 0)), # up right back
    ((0, 2, 0), (1, 0, 2), (2, 0, 0)), # up left front
    ((0, 2, 2), (2, 0, 2), (3, 0, 0)), # up right front
    ((1, 2, 2), (2, 2, 0), (5, 0, 0)), # left front down
    ((2, 2, 2), (3, 2, 0), (5, 0, 2)), # front right down
    ((1, 2, 0), (5, 2, 0), (4, 2, 2)), # left down back
    ((3, 2, 2), (4, 2, 0), (5, 2, 2))  # right back down
]


EDGES = [
    ((0, 0, 1), (4, 0, 1)), # up back
    ((0, 1, 0), (1, 0, 1)), # up left
    ((0, 1, 2), (3, 0, 1)), # up right
    ((0, 2, 1), (2, 0, 1)), # up front
    ((1, 1, 2), (2, 1, 0)), # left front
    ((2, 1, 2), (3, 1, 0)), # front right
    ((3, 1, 2), (4, 1, 0)), # right back
    ((4, 1, 2), (1, 1, 0)), # back left
    ((5, 0, 1), (2, 2, 1)), # down front
    ((5, 1, 0), (1, 2, 1)), # down left
    ((5, 1, 2), (3, 2, 1)), # down right
    ((5, 2, 1), (4, 2, 1))  # down back
]


CENTERS = [
    ((0, 1, 1),),
    ((1, 1, 1),),
    ((2, 1, 1),),
    ((3, 1, 1),),
    ((4, 1, 1),),
    ((5, 1, 1),)
]


def piece_iter():
    """Iterator for every piece in a Rubik's Cube.

    Yields multi-indices for a numpy 3D array."""
    for vector_of_indices in chain(CORNERS, EDGES, CENTERS):
        yield tuple(zip(*vector_of_indices))


def piece_facelets():
    """Generates a dict of color vector to facelet vector."""
    facelet_cube = gen_3d_cube()
    color_cube = gen_3d_cube()
    for face in range(FACES):
        color_cube[face].fill(face)

    color_to_facelet = {}

    for index in piece_iter():
        sort = sorted(zip(color_cube[index], facelet_cube[index]))
        color_vector, facelet_vector = zip(*sort)
        color_to_facelet[color_vector] = facelet_vector

    return color_to_facelet
