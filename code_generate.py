code = """
    acc += tileMatWUnroll[ty][{}] * tileMatXUnroll[{}][tx];
    __syncthreads();
"""
TILE_WIDTH = 16
for i in range(TILE_WIDTH):
    print(code.format(i, i), end='')