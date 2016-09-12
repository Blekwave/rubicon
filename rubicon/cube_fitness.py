from rubikscube import piece_iter, gen_3d_cube


SIDE = 3
FACES = 6


def facelet_to_color(cube):
    """Convert a facelet cube into a color-coded cube.

    Parameters:
    - cube: flat array Rubik's Cube with integers from 0 to 53

    Returns an array with integers from 0 to 5, in reference to each
    of the six colors/faces."""
    return cube // (SIDE * SIDE)


def wrong_color_facelets(cube):
    """Count the number of facelets with the wrong color in each face.

    Parameters:
    - cube: flat array Rubik's Cube

    Returns the number of wrong color facelets in the cube."""
    color_cube = facelet_to_color(cube)
    total = 0
    for right_color, face in enumerate(color_cube.reshape(FACES, SIDE * SIDE)):
        total += sum(face != right_color)
    return total

_perfect_cube = gen_3d_cube()


def wrong_cubelets(cube):
    """Count the number of wrongly positioned cubelets in the cube.

    Parameters:
    - cube: flat array Rubik's Cube

    Returns the number of wrongly positioned cubelets in the cube."""
    cube = cube.reshape(FACES, SIDE, SIDE)
    total = 0
    for index in piece_iter():
        if not sorted(cube[index]) == sorted(_perfect_cube[index]):
            total += 1
    return total


