import itertools
def get_grid_iterable():
    param_grid = {'learning_rate': [float(v) for v in np.arange(0.01, 0.25, 0.01)],
                  'subsample': [float(v) for v in np.arange(0.5, 1.01, 0.1)],
                  'reg_alpha': [float(v) for v in np.arange(0.01, 0.5, 0.05)],
                  'max_depth': [int(v) for v in np.arange(3, 14, 1)],
                  'gamma': [int(v) for v in np.arange(0, 10, 2)]
                  }
    grid_iter = []
    length = 1
    for k in param_grid:
        grid_iter.append(param_grid[k])
        length *= len(param_grid[k])

    return itertools.product(*grid_iter), list(param_grid.keys()), length-1

grid=get_grid_iterable()
print(grid)