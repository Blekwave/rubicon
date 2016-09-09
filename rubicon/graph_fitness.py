import rubikscube as rc
import numpy as np


N = 54  # facelets in a 3x3 cube


def move_transition_adjmatrix(move):
    """Generate the adjacency matrix for a cube move's transitions

    This adjacency matrix has edges between the position of each of
    a cube's facelets before and after a move (self loops are not
    recorded)

    Parameters:
    - move: flat Rubik's Cube move function

    Returns the aforementioned adjacency matrix"""
    cube = rc.gen_cube()
    move(cube)

    adjmatrix = np.zeros((N, N), dtype=int)

    for before, after in enumerate(cube):
        if before != after:
            adjmatrix[before, after] = 1

    return adjmatrix


def all_moves_adj_matrix():
    """Combine move transition graphs for all possible moves.

    Returns a single adjacency matrix which is the union of the move
    graphs for all possible moves for a cube."""
    adjmatrix = np.zeros((N, N), dtype=int)

    for move in rc.move_list:
        adjmatrix |= move_transition_adjmatrix(move)

    return adjmatrix


def bfs_all(adjmatrix, source):
    """Find the BFS distance to all vertices from a source vertex.

    Parameters:
    - adjmatrix: graph, in the form of an adjacency matrix
    - source: source vertex index

    Returns a list of distances, indexed by vertex index."""
    n = adjmatrix.shape[0]
    assert source >= 0 and source < n

    visited = np.zeros(n, dtype=bool)
    distance = np.zeros(n, dtype=int)
    distance.fill(-999)

    queue = [(source, 0)]
    visited[source] = True

    while queue:
        vertex, dist = queue.pop(0)
        distance[vertex] = dist

        for neighbor, is_adjacent in enumerate(adjmatrix[vertex]):
            if is_adjacent and not visited[neighbor]:
                queue.append((neighbor, dist + 1))
                visited[neighbor] = True

    return distance


def facelet_distances():
    """Compute the pairwise distance matrix between all facelets.

    The distance between two facelets i and j is the minimum amount
    of moves required to move facelet i to facelet j's position.

    Returns a 54x54 matrix D where D[i, j] is the distance between
    facelets i and j."""
    adjmatrix = all_moves_adj_matrix()
    distances = np.zeros((N, N), dtype=int)

    for i in range(N):
        distances[i, :] = bfs_all(adjmatrix, i)

    return distances

_distances = facelet_distances()


def solution_distance(cube):
    """Compute the minimum solution distance of a cube.

    The minimum solution distance of a cube is a metric given by the
    sum of the distance between a facelet and its position in the
    solved cube, for all facelets.

    Parameters:
    - cube: cube for which the distance should be calculated"""
    distances = [_distances[pos, facelet] for pos, facelet in enumerate(cube)]
    return sum(distances)


def graph_fitness(ind, initial_cube):
    """Compute the fitness based on the BFS solution distance.

    Parameters:
    - ind: individual to be evaluated
    - initial_cube: initial state of the cube

    Returns the solution distance for the cube obtained by evaluating
    the individual."""
    cube = rc.apply_moves(initial_cube, ind)
    score = solution_distance(cube)
    return score
