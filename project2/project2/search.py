class Searcher:
    def __init__(self, graph, algorithm, start, goals, expansions, heuristic,
                 verbose):
        self.graph = graph
        self.algorithm = algorithm

        # Turn these from labels into actual Node objects when we put them on
        # the Searcher object
        self.start = graph.nodes[start]
        self.goals = {}
        for goal in goals:
            self.goals.add(graph.nodes[goal])

        self.expansions = expansions
        self.heuristic = heuristic
        self.verbose = verbose

        self.open = set([start])
        self.visited = set([start])
        self.expansions_done = 0

    def __call__(self):
        algorithms = {
            'bfs': self.BFS,
            'dfs': self.DFS,
            'best': self.BEST,
            'ids': self.IDS,
            'a*': self.AStar
        }

        algorithms[self.algorithm]()

    def BFS(self):
        pass

    def DFS(self):
        pass

    def BEST(self):
        pass

    def IDS(self):
        pass

    def AStar(self):
        pass
