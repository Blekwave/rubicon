import rubikscube.rubikscube as rc
import numpy as np

_colors = ["O", "G", "B", "R", "Y", "W"]
_color_num = {color: num
              for num, color in enumerate(_colors)}

_face_names = ["Up", "Left", "Front", "Right", "Back", "Down"]
_face_num = {face: num
             for num, face in enumerate(_face_names)}


def read_cube(f):
    """Read a color-coded Rubik's Cube from a file.

    Parameters:
    - f: open file object from which the cube should be read

    Returns a color-coded cube, with colors from 0 to 5 in no
    particular order."""
    side = int(next(f)[:-1])
    cube = rc.gen_3d_cube()

    for _ in range(rc.FACES):
        face_name = next(f)[:-1]
        k = _face_num[face_name]

        for i in range(side):
            row = next(f)[:-1]
            for j, facelet in enumerate(row.split()):
                cube[k, i, j] = _color_num[facelet]

    return cube


def center_colors(cube):
    """List the center colors of a cube's faces, in order.

    Parameters:
    - cube: color-coded 3D Rubik's Cube

    Returns a numpy array with the color code of the central piece of
    each face, indexed by face number."""
    return np.array([cube[k, 1, 1] for k in range(rc.FACES)])


def remap_colors(cube, before_colors, after_colors):
    """Remap a cube's colors.

    Parameters:
    - cube: color-coded 3D Rubik's Cube
    - before_colors: colors before the translation
    - after_colors: color after the translation, indexed in parallel
                    with before_colors

    Returns a cube with remapped colors."""
    color_dict = {before: after
                  for before, after in zip(before_colors, after_colors)}

    color_map = np.vectorize(lambda before: color_dict[before])
    return color_map(cube)


def color_to_facelet(color_cube):
    """Transforms a color-coded cube into a facelet cube.

    Parameters:
    - cube: color-coded 3D Rubik's Cube

    Returns a facelet Rubik's Cube, with values from 0 to 53."""

    side = color_cube.shape[1]
    facelet_cube = color_cube.copy()

    piece_ids = rc.piece_facelets()

    for index in rc.piece_iter():
        color_vector = color_cube[index]
        color_sort = sorted(zip(color_vector, range(len(color_vector))))
        sorted_colors, order = zip(*color_sort)

        sorted_ids = piece_ids[tuple(sorted_colors)]
        id_sort = sorted(zip(order, sorted_ids))
        _, ids = zip(*id_sort)
        facelet_cube[index] = ids

    return facelet_cube


def from_file(path, flatten=False):
    """Read a facelet Rubik's Cube from a color-coded file.

    Parameters:
    - path: path to the file containing the color-coded cube
    - flatten: whether to flatten the cube

    Returns a facelet Rubik's Cube, with values from 0 to 53."""
    with open(path) as f:
        cube = read_cube(f)

    before_colors = center_colors(cube)
    after_colors = np.array(range(rc.FACES))
    remapped_cube = remap_colors(cube, before_colors, after_colors)
    cube = color_to_facelet(remapped_cube)
    if flatten:
        cube = cube.flatten()
    return cube
