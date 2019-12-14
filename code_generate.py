code = """
    if (c >= currM && c < nextM && (h + {}) >= currH && (h + {}) < nextH && (w + {}) >= currW && (w + {}) < nextW)
        y[bmhw] += subTile[c - currM][t2 + {}][t3 + {}] * K4d(m, c, {}, {});
    else
        y[bmhw] += x4d(b, c, (h + {}), (w + {})) * K4d(m, c, {}, {});
"""

for i in range(5):
    for j in range(5):
        print(code.format(i, i, j, j, i, j, i, j, i, j, i ,j), end='')