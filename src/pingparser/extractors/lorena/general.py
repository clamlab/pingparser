import numpy as np

def euclidean_dist(df, xy1_col, xy2_col):
    # calculate euclidean distance between two pairs of columns in df
    # each pair containing x, and y, coordinate
    # xy1_col: list of colnames e.g. ['Cue_x', 'Cue_y']

    xy1 = np.array(df[[xy1_col[0], xy1_col[1]]])
    xy2 = np.array(df[[xy2_col[0], xy2_col[1]]])

    differences = xy1 - xy2
    squared_differences = differences ** 2
    sum_of_squares = np.sum(squared_differences, axis=1)

    dist = np.sqrt(sum_of_squares)
    return dist
