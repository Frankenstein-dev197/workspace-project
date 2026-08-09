"""Algorithm pattern library: curated algorithmic knowledge from LeetCode.

Integrates problem patterns, data structure summaries, and algorithmic
techniques inspired by azl397985856/leetcode. Provides agents with
ready-to-use algorithmic knowledge for coding tasks.
"""

from __future__ import annotations

from typing import Any

from daemon_engine.knowledge.knowledge_base import (
    KnowledgeBase,
    KnowledgeEntry,
    KnowledgeSource,
)


ALGORITHM_PATTERNS: list[dict[str, Any]] = [
    {
        "title": "Two Pointers Technique",
        "content": (
            "Two pointers is a technique where two pointers traverse a data structure "
            "(usually an array or linked list) simultaneously. Common patterns:\n"
            "1. Opposite direction: one at start, one at end, move towards center (palindrome, two-sum sorted)\n"
            "2. Same direction: slow and fast pointers (cycle detection, remove duplicates)\n"
            "3. Sliding window: maintain a window of elements, expand/contract based on condition\n\n"
            "Use when: array/string problems with sorted input, pair-finding, or window-based queries.\n"
            "Time: O(n), Space: O(1)"
        ),
        "category": "technique",
        "tags": ["array", "string", "two-pointers", "sliding-window"],
    },
    {
        "title": "Binary Search",
        "content": (
            "Binary search finds an element in a sorted array in O(log n) time.\n"
            "Pattern: maintain left/right boundaries, check middle, narrow search space.\n\n"
            "Variants:\n"
            "1. Standard: find exact element\n"
            "2. Lower bound: first element >= target\n"
            "3. Upper bound: first element > target\n"
            "4. Rotated array: modified binary search with pivot check\n"
            "5. Binary search on answer: search over possible answer space\n\n"
            "Template:\n"
            "  left, right = 0, len(arr) - 1\n"
            "  while left <= right:\n"
            "      mid = left + (right - left) // 2\n"
            "      if arr[mid] == target: return mid\n"
            "      elif arr[mid] < target: left = mid + 1\n"
            "      else: right = mid - 1"
        ),
        "category": "technique",
        "tags": ["binary-search", "array", "divide-conquer"],
    },
    {
        "title": "Dynamic Programming",
        "content": (
            "Dynamic Programming (DP) solves problems by breaking them into overlapping "
            "subproblems and storing results to avoid recomputation.\n\n"
            "Two approaches:\n"
            "1. Top-down (memoization): recursive with cache\n"
            "2. Bottom-up (tabulation): iterative, fill table\n\n"
            "Steps to solve DP:\n"
            "1. Define state: what does dp[i] represent?\n"
            "2. Base case: smallest subproblem solution\n"
            "3. Recurrence: how to compute dp[i] from previous states\n"
            "4. Order: compute states in dependency order\n"
            "5. Answer: which state(s) give the final answer\n\n"
            "Common patterns:\n"
            "- 1D DP: fibonacci, climbing stairs, house robber\n"
            "- 2D DP: grid paths, edit distance, LCS\n"
            "- Interval DP: matrix chain multiplication, burst balloons\n"
            "- Knapsack: 0/1 and unbounded\n"
            "- State machine DP: stock trading problems"
        ),
        "category": "technique",
        "tags": ["dynamic-programming", "optimization", "memoization"],
    },
    {
        "title": "Breadth-First Search (BFS)",
        "content": (
            "BFS explores a graph/tree level by level using a queue.\n"
            "Use for: shortest path in unweighted graph, level-order traversal, "
            "minimum steps to reach target.\n\n"
            "Template:\n"
            "  from collections import deque\n"
            "  queue = deque([start])\n"
            "  visited = {start}\n"
            "  while queue:\n"
            "      node = queue.popleft()\n"
            "      for neighbor in graph[node]:\n"
            "          if neighbor not in visited:\n"
            "              visited.add(neighbor)\n"
            "              queue.append(neighbor)\n\n"
            "Level-order: track level size, process all nodes at current level before next.\n"
            "Time: O(V+E), Space: O(V)"
        ),
        "category": "graph",
        "tags": ["bfs", "graph", "shortest-path", "queue"],
    },
    {
        "title": "Depth-First Search (DFS)",
        "content": (
            "DFS explores as deep as possible before backtracking, using recursion or a stack.\n"
            "Use for: connected components, topological sort, cycle detection, path finding.\n\n"
            "Template (recursive):\n"
            "  def dfs(node, visited):\n"
            "      visited.add(node)\n"
            "      for neighbor in graph[node]:\n"
            "          if neighbor not in visited:\n"
            "              dfs(neighbor, visited)\n\n"
            "Template (iterative with stack):\n"
            "  stack = [start]\n"
            "  visited = set()\n"
            "  while stack:\n"
            "      node = stack.pop()\n"
            "      if node in visited: continue\n"
            "      visited.add(node)\n"
            "      stack.extend(graph[node])\n\n"
            "Time: O(V+E), Space: O(V) for recursion stack"
        ),
        "category": "graph",
        "tags": ["dfs", "graph", "recursion", "backtracking"],
    },
    {
        "title": "Hash Map / Hash Table",
        "content": (
            "Hash maps provide O(1) average insert, delete, and lookup.\n"
            "Use for: frequency counting, caching, two-sum, deduplication, grouping.\n\n"
            "Python dict operations:\n"
            "  d[key] = value      # O(1) insert/update\n"
            "  d.get(key, default) # O(1) lookup with default\n"
            "  del d[key]          # O(1) delete\n"
            "  key in d            # O(1) membership\n\n"
            "Counter for frequency:\n"
            "  from collections import Counter\n"
            "  freq = Counter(iterable)\n\n"
            "Defaultdict for grouping:\n"
            "  from collections import defaultdict\n"
            "  graph = defaultdict(list)"
        ),
        "category": "data-structure",
        "tags": ["hash-map", "dict", "lookup", "counter"],
    },
    {
        "title": "Heap / Priority Queue",
        "content": (
            "A heap provides O(log n) insert and O(1) peek at min/max element.\n"
            "Use for: top-k problems, merge k sorted lists, scheduling, median stream.\n\n"
            "Python heapq (min-heap):\n"
            "  import heapq\n"
            "  heap = []\n"
            "  heapq.heappush(heap, item)    # O(log n)\n"
            "  min_val = heapq.heappop(heap) # O(log n)\n"
            "  heapq.heapify(list)           # O(n)\n\n"
            "Max-heap: negate values or use custom comparator\n"
            "Top-k: maintain heap of size k, push/pop as needed\n"
            "Time: O(n log k) for top-k, O(n log n) for full sort"
        ),
        "category": "data-structure",
        "tags": ["heap", "priority-queue", "top-k", "heapq"],
    },
    {
        "title": "Union-Find / Disjoint Set",
        "content": (
            "Union-Find tracks connected components with near-O(1) operations "
            "using path compression and union by rank.\n\n"
            "Use for: connected components, cycle detection in undirected graph, "
            "kruskal's MST, dynamic connectivity.\n\n"
            "Implementation:\n"
            "  class UnionFind:\n"
            "      def __init__(self, n):\n"
            "          self.parent = list(range(n))\n"
            "          self.rank = [0] * n\n"
            "      def find(self, x):\n"
            "          if self.parent[x] != x:\n"
            "              self.parent[x] = self.find(self.parent[x])\n"
            "          return self.parent[x]\n"
            "      def union(self, x, y):\n"
            "          px, py = self.find(x), self.find(y)\n"
            "          if px == py: return False\n"
            "          if self.rank[px] < self.rank[py]: px, py = py, px\n"
            "          self.parent[py] = px\n"
            "          if self.rank[px] == self.rank[py]: self.rank[px] += 1\n"
            "          return True\n\n"
            "Time: O(alpha(n)) ~ O(1) amortized"
        ),
        "category": "data-structure",
        "tags": ["union-find", "disjoint-set", "components", "kruskal"],
    },
    {
        "title": "Backtracking",
        "content": (
            "Backtracking explores all possible solutions by building candidates "
            "incrementally and abandoning a candidate (backtracking) as soon as "
            "it cannot lead to a valid solution.\n\n"
            "Template:\n"
            "  def backtrack(path, choices):\n"
            "      if is_solution(path):\n"
            "          results.append(path[:])\n"
            "          return\n"
            "      for choice in choices:\n"
            "          if is_valid(choice):\n"
            "              path.append(choice)\n"
            "              backtrack(path, updated_choices)\n"
            "              path.pop()  # backtrack\n\n"
            "Use for: permutations, combinations, subsets, N-Queens, sudoku.\n"
            "Optimization: prune early to reduce search space."
        ),
        "category": "technique",
        "tags": ["backtracking", "recursion", "permutation", "combination"],
    },
    {
        "title": "Sliding Window",
        "content": (
            "Sliding window maintains a window [left, right] over data, expanding "
            "right and shrinking left based on conditions.\n\n"
            "Two patterns:\n"
            "1. Fixed window: window of size k, slide by 1\n"
            "2. Variable window: expand right, shrink left until valid\n\n"
            "Template (variable window):\n"
            "  left = 0\n"
            "  for right in range(len(arr)):\n"
            "      add arr[right] to window\n"
            "      while window_invalid():\n"
            "          remove arr[left] from window\n"
            "          left += 1\n"
            "      update_result()\n\n"
            "Use for: max sum subarray, longest substring without repeats, "
            "min window substring, fruit into baskets."
        ),
        "category": "technique",
        "tags": ["sliding-window", "array", "string", "two-pointers"],
    },
]


class AlgorithmPatternLibrary:
    """Pre-loaded algorithmic knowledge base inspired by LeetCode patterns."""

    def __init__(self, knowledge_base: KnowledgeBase | None = None) -> None:
        self.kb = knowledge_base or KnowledgeBase()
        self._loaded = False

    def load_patterns(self) -> int:
        if self._loaded:
            return len(self.kb.get_by_source(KnowledgeSource.LEETCODE))
        count = 0
        for pattern in ALGORITHM_PATTERNS:
            self.kb.add_entry(
                title=pattern["title"],
                content=pattern["content"],
                source=KnowledgeSource.LEETCODE,
                category=pattern["category"],
                tags=pattern["tags"],
                metadata={"type": "algorithm_pattern"},
            )
            count += 1
        self._loaded = True
        return count

    def search_patterns(self, query: str, limit: int = 5) -> list[KnowledgeEntry]:
        if not self._loaded:
            self.load_patterns()
        results = self.kb.search(query, limit=limit, source=KnowledgeSource.LEETCODE)
        return [entry for _, entry in results]

    def get_pattern(self, name: str) -> KnowledgeEntry | None:
        if not self._loaded:
            self.load_patterns()
        for entry in self.kb.get_by_source(KnowledgeSource.LEETCODE):
            if name.lower() in entry.title.lower():
                return entry
        return None

    def list_patterns(self) -> list[str]:
        if not self._loaded:
            self.load_patterns()
        return [e.title for e in self.kb.get_by_source(KnowledgeSource.LEETCODE)]
