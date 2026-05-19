"""Hungarian algorithm — mirrors `lib/core/utils/hungarian_algorithm.dart`.

Finds minimum-cost assignment in a bipartite graph via the Kuhn-Munkres
algorithm. Costs are converted to profits internally.
"""

from __future__ import annotations

from collections import deque
from typing import List


class HungarianAlgorithm:
    def __init__(self, cost_matrix: List[List[int]]):
        self._n = len(cost_matrix)
        # Deep copy and negate to convert min-cost into max-profit.
        self._cost = [[-cost_matrix[i][j] for j in range(self._n)] for i in range(self._n)]

        self._xy = [-1] * self._n
        self._yx = [-1] * self._n
        self._lx = [0] * self._n
        self._ly = [0] * self._n
        self._slack = [0] * self._n
        self._slack_x = [0] * self._n
        self._prev = [0] * self._n
        self._in_tree_x = [False] * self._n
        self._in_tree_y = [False] * self._n

        self._match_count = 0

    def _label_it(self) -> None:
        for i in range(self._n):
            for j in range(self._n):
                if self._cost[i][j] > self._lx[i]:
                    self._lx[i] = self._cost[i][j]

    def _add_tree(self, x: int, prev_x: int) -> None:
        self._in_tree_x[x] = True
        self._prev[x] = prev_x
        for y in range(self._n):
            current_slack = self._lx[x] + self._ly[y] - self._cost[x][y]
            if current_slack < self._slack[y]:
                self._slack[y] = current_slack
                self._slack_x[y] = x

    def _update_labels(self) -> None:
        delta = 999_999_999
        for y in range(self._n):
            if not self._in_tree_y[y]:
                if self._slack[y] < delta:
                    delta = self._slack[y]

        for x in range(self._n):
            if self._in_tree_x[x]:
                self._lx[x] -= delta
        for y in range(self._n):
            if self._in_tree_y[y]:
                self._ly[y] += delta
        for y in range(self._n):
            if not self._in_tree_y[y]:
                self._slack[y] -= delta

    def _augment(self) -> None:
        if self._match_count == self._n:
            return

        y = -1
        root = -1
        q: deque[int] = deque()

        for i in range(self._n):
            if self._xy[i] == -1:
                root = i
                q.append(root)
                self._prev[i] = -2
                self._in_tree_x[i] = True
                break

        if root == -1:
            return

        for i in range(self._n):
            self._slack[i] = self._lx[root] + self._ly[i] - self._cost[root][i]
            self._slack_x[i] = root

        x = root
        while True:
            while q:
                x = q.popleft()
                found = False
                for y_inner in range(self._n):
                    if (self._lx[x] + self._ly[y_inner] - self._cost[x][y_inner] == 0) and (
                        not self._in_tree_y[y_inner]
                    ):
                        if self._yx[y_inner] == -1:
                            y = y_inner
                            found = True
                            break
                        else:
                            self._in_tree_y[y_inner] = True
                            q.append(self._yx[y_inner])
                            self._add_tree(self._yx[y_inner], x)
                if found:
                    break
            if y != -1 and self._yx[y] == -1:
                break

            self._update_labels()

            found_after = False
            for y_inner in range(self._n):
                if (not self._in_tree_y[y_inner]) and self._slack[y_inner] == 0:
                    if self._yx[y_inner] == -1:
                        x = self._slack_x[y_inner]
                        y = y_inner
                        found_after = True
                        break
                    else:
                        self._in_tree_y[y_inner] = True
                        if not self._in_tree_x[self._yx[y_inner]]:
                            q.append(self._yx[y_inner])
                            self._add_tree(self._yx[y_inner], self._slack_x[y_inner])
            if found_after:
                break

        if y != -1 and self._yx[y] == -1:
            self._match_count += 1
            cx = self._slack_x[y]
            cy = y
            while cx != -2:
                ty = self._xy[cx]
                self._xy[cx] = cy
                self._yx[cy] = cx
                cy = ty
                cx = self._prev[cx]
            self._in_tree_x = [False] * self._n
            self._in_tree_y = [False] * self._n
            self._augment()

    def get_assignment(self) -> List[int]:
        self._label_it()
        while self._match_count < self._n:
            self._in_tree_x = [False] * self._n
            self._in_tree_y = [False] * self._n
            self._prev = [0] * self._n
            self._slack = [999_999_999] * self._n
            self._slack_x = [0] * self._n
            initial = self._match_count
            self._augment()
            if self._match_count == initial and self._match_count < self._n:
                break
        return self._xy

    getAssignment = get_assignment
