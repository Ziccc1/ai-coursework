# 直接保留给定的地图、TreeNode 定义及共享 max2 建树方式。
from .tree import TreeNode

MAZE = [['S', '.', '.', 'X', '.'], ['.', 'X', '.', '.', '.'],
        ['.', 'X', '.', 'X', '.'], ['.', '.', '.', 'X', 'E'], ['X', 'X', '.', '.', '.']]
WEIGHTED = [['S', 1, 1, 8, 1], [1, 8, 3, 1, 1], [1, 8, 1, 8, 1],
            [1, 1, 1, 8, 1], [8, 8, 1, 1, 'E']]


def teacher_tree():
    leaf1, leaf2 = TreeNode(value=3), TreeNode(value=5)
    leaf3, leaf4 = TreeNode(value=6), TreeNode(value=9)
    leaf5, leaf6 = TreeNode(value=1), TreeNode(value=2)
    max1 = TreeNode(children=[leaf1, leaf2])
    max2 = TreeNode(children=[leaf3, leaf4])
    max3 = TreeNode(children=[leaf5, leaf6])
    min1 = TreeNode(children=[max1, max2])
    min2 = TreeNode(children=[max2, max3])
    return TreeNode(children=[min1, min2])
