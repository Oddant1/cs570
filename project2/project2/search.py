class Searcher:
    def __init__(self, graph, algorithm, start, goals, expansions, heuristic,
                 verbose):
        self.graph = graph
        self.algorithm = algorithm

        self.start = SearchNode(start)
        self.goals = goals

        self.expansions = expansions
        self.expansions_taken = 0
        self.heuristic = heuristic
        self.verbose = verbose

        # Currently open nodes
        self.frontier = [self.start]
        self.current_node = None

        # Stats
        self.max_open = 0
        self.avg_open = 0

        self.max_depth = 0
        self.avg_depth = 0

        self.avg_branching_factor = 0

    def search(self):
        while self.expansions == 0 or self.expansions_taken < self.expansions:
            self.current_node = self.frontier.pop(0)

            if self.current_node in self.goals:
                return self.current_node
            # TODO: Generate stats not related to branching factor here

            children = self._expand_node()
            legal_children = self._find_legal_children(children)
            self._add_legal_children(legal_children)

            if self.verbose:
                print(f'VERBOSE:\nEXPANDED: {self.current_node}\nCHILDREN:'
                      f' {children}\nLEGAL CHILDREN:'
                      f' {legal_children}\nFRONTIER: {self.frontier}\n')

            self.expansions_taken += 1

        return self.current_node

    def _expand_node(self):
        """ Gets all children of the current Node
        """
        children = []
        for edge in self.graph.edges[self.current_node.label]:
            child = SearchNode(edge.end, path=self.current_node.path,
                               cost=self.current_node.cost + edge.length)
            children.append(child)

        if self.algorithm == 'BEST':
            children.sort(key=lambda node: node.cost)
        elif self.algorithm == 'A*':
            if self.heuristic == 1:
                pass
            else:
                pass
        else:
            children.sort(key=lambda node: node.label)

        # TODO: Generate stats related to branching factor here
        return children

    def _find_legal_children(self, children):
        """ Determines which of the children in the node are legal
        """
        legal_children = []

        for child in children:
            # Don't go in circles
            if child in self.current_node.path:
                continue

            if child in self.frontier:
                # For BFS we keep the old one
                if self.algorithm == 'BFS':
                    pass
                # For all other algorithms we may keep th new one
                else:
                    legal_children.append(child)
            else:
                legal_children.append(child)

        return legal_children

    def _add_legal_children(self, legal_children):
        """ Add legal children to the frontier and visited list
        """
        if self.algorithm == 'BFS':
            self.frontier.extend(legal_children)
        elif self.algorithm == 'DFS':
            # if not legal_children:
            #     last_node = self.current_node.path[-2]
            #     if last_node in self.frontier:
            #         self.frontier.remove(last_node)
            # else:
            legal_children.reverse()
            for child in legal_children:
                if child in self.frontier:
                    self.frontier.remove(child)
                self.frontier.insert(0, child)
        elif self.algorithm == 'IDS':
            pass
        elif self.algorithm == 'BEST':
            pass
        else:
            pass


class SearchNode:
    """ This represents a node identified in the search. It holds the node's
    label, the cost to reach it, and the path taken
    """

    def __init__(self, label, path=[], cost=0):
        self.label = label

        # The path we took to this node
        self.path = path.copy()
        self.path.append(self)

        # The cost to reach this node from the start
        self.cost = cost

    def __eq__(self, other):
        """ We only really care if the objects are referring to the same node
        on the graph not the same path to the node
        """
        if isinstance(other, SearchNode):
            return self.label == other.label
        else:
            return self.label == other

    def __repr__(self):
        return self.label
