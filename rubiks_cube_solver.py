import copy
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import heapq


colors = ['W', 'Y', 'G', 'B', 'O', 'R']
faces = ['U', 'D', 'F', 'B', 'L', 'R']


move_map = {
    'U': [('B', 0), ('R', 0), ('F', 0), ('L', 0)],
    'D': [('F', 2), ('R', 2), ('B', 2), ('L', 2)],
    'F': [('U', 2), ('R', 'col0'), ('D', 0), ('L', 'col2')],
    'B': [('U', 0), ('L', 'col0'), ('D', 2), ('R', 'col2')],
    'L': [('U', 'col0'), ('F', 'col0'), ('D', 'col0'), ('B', 'col2')],
    'R': [('U', 'col2'), ('B', 'col0'), ('D', 'col2'), ('F', 'col2')]
}


def create_solved_cube():
    return {face: [[color]*3 for _ in range(3)] for face, color in zip(faces, colors)}


def rotate_face_cw(face):
    return [list(row) for row in zip(*face[::-1])]


def rotate_face_ccw(face):
    return [list(row) for row in zip(*face)][::-1]


def get_row(face, idx):
    return face[idx][:]


def set_row(face, idx, row):
    face[idx] = row[:]


def get_col(face, idx):
    return [row[idx] for row in face]


def set_col(face, idx, col):
    for i in range(3):
        face[i][idx] = col[i]


def apply_move(cube, move):
    face = move[0]
    direction = 1 if len(move) == 1 else -1
    cube[face] = rotate_face_cw(
        cube[face]) if direction == 1 else rotate_face_ccw(cube[face])
    adj = move_map[face]
    if direction == -1:
        adj = adj[::-1]
    edges = []
    for side, idx in adj:
        if isinstance(idx, int):
            edges.append(get_row(cube[side], idx))
        else:
            col_idx = int(idx[-1])
            edges.append(get_col(cube[side], col_idx))
    edges = [edges[-1]] + edges[:-1]
    for (side, idx), edge in zip(adj, edges):
        if isinstance(idx, int):
            set_row(cube[side], idx, edge)
        else:
            col_idx = int(idx[-1])
            set_col(cube[side], col_idx, edge)


def random_scramble(cube, moves=10):
    move_list = ['U', "U'", 'D', "D'", 'F',
                 "F'", 'B', "B'", 'L', "L'", 'R', "R'"]
    scramble_seq = random.choices(move_list, k=moves)
    states = [copy.deepcopy(cube)]
    for move in scramble_seq:
        new_state = copy.deepcopy(states[-1])
        apply_move(new_state, move)
        states.append(new_state)
    return states, scramble_seq


def build_state_graph(states):
    G = nx.DiGraph()
    for i in range(len(states) - 1):
        G.add_node(i, state=states[i])
        G.add_edge(i, i + 1, move='')
    G.add_node(len(states) - 1, state=states[-1])
    return G


def visualize_cube_state(cube, title='Cube State', ax=None):
    face_coords = {
        'U': (3, 6), 'L': (0, 3), 'F': (3, 3),
        'R': (6, 3), 'B': (9, 3), 'D': (3, 0)
    }
    color_lookup = {
        'W': 'lightgray',  # Changed from white to lightgray
        'Y': 'yellow',
        'G': 'green',
        'B': 'blue',
        'O': 'orange',
        'R': 'red'
    }
    if ax is None:
        fig, ax = plt.subplots()
    ax.clear()
    for face, (x_offset, y_offset) in face_coords.items():
        face_grid = cube[face]
        for i in range(3):
            for j in range(3):
                color = color_lookup[face_grid[i][j]]
                ax.add_patch(plt.Rectangle(
                    (x_offset + j, y_offset + 2 - i), 1, 1, facecolor=color))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.axis('off')


def animate_solution(states, title='Solving Cube'):
    fig, ax = plt.subplots()

    def update(frame):
        visualize_cube_state(
            states[frame], title=f'{title}: Step {frame}', ax=ax)
    ani = animation.FuncAnimation(
        fig, update, frames=len(states), repeat=False, interval=500)
    plt.show()


def is_solved(cube):
    for face in cube:
        center = cube[face][1][1]
        for row in cube[face]:
            for color in row:
                if color != center:
                    return False
    return True


# def heuristic(cube):
#     score = 0
#     for face in cube:
#         center = cube[face][1][1]
#         for row in cube[face]:
#             for color in row:
#                 if color != center:
#                     score += 1
#     return score

def heuristic(cube):
    score = 0
    incorrect_faces = {}

    for face in cube:
        center = cube[face][1][1]
        face_grid = cube[face]
        display_grid = []
        incorrect_count = 0

        for i in range(3):
            row_display = []
            for j in range(3):
                color = face_grid[i][j]
                if color != center:
                    row_display.append(color.lower())  # Mark incorrect
                    incorrect_count += 1
                    score += 1
                else:
                    row_display.append(color)
            display_grid.append(row_display)

        if incorrect_count > 0:
            incorrect_faces[face] = (display_grid, incorrect_count)

    # Print visual display of incorrect faces
    if incorrect_faces:
        print("\nIncorrect faces (lowercase = incorrect):")
        for face, (grid, count) in incorrect_faces.items():
            print(
                f"\nFace {face} (center: {cube[face][1][1]}): {count}/9 incorrect")
            for row in grid:
                print(" ".join(row))

    return score


def a_star_solver(start_cube):
    move_list = ['U', "U'", 'D', "D'", 'F',
                 "F'", 'B', "B'", 'L', "L'", 'R', "R'"]
    visited = set()
    queue = []
    counter = 0
    heapq.heappush(queue, (0, 0, counter, start_cube, []))

    while queue:
        _, cost, _, current, path = heapq.heappop(queue)
        cube_id = str(current)
        if cube_id in visited:
            continue
        visited.add(cube_id)
        if is_solved(current):
            return path
        for move in move_list:
            new_cube = copy.deepcopy(current)
            apply_move(new_cube, move)
            if str(new_cube) not in visited:
                new_path = path + [move]
                est_cost = cost + 1 + heuristic(new_cube)
                counter += 1
                heapq.heappush(queue, (est_cost, cost + 1,
                               counter, new_cube, new_path))
    return []


def apply_sequence(cube, sequence):
    states = [copy.deepcopy(cube)]
    for move in sequence:
        apply_move(cube, move)
        states.append(copy.deepcopy(cube))
    return states


if __name__ == '__main__':
    cube = create_solved_cube()
    states, scramble_seq = random_scramble(cube, moves=5)
    scrambled = copy.deepcopy(states[-1])

    print("Scramble sequence:", scramble_seq)
    visualize_cube_state(states[0], title='Solved State')
    visualize_cube_state(scrambled, title='Scrambled State')

    solution_seq = a_star_solver(copy.deepcopy(scrambled))
    print("Solution sequence:", solution_seq)
    solve_states = apply_sequence(scrambled, solution_seq)
    animate_solution(solve_states, title='Solving with A* Search')
