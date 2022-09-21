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
        # All currently and previously opened nodes in order accessed
        self.visited = []

        # Stats
        self.max_open = 0
        self.avg_open = 0

        self.max_depth = 0
        self.avg_depth = 0

        self.avg_branching_factor = 0

    def search(self):
        while self.expansions == 0 or self.expansions_taken < self.expansions:
            next_node = self.frontier.pop(0)
            self.visited.append(next_node)

            if next_node in self.goals:
                # A node is implicitly at the end of our own path, append it
                # before returning for display purposes
                next_node.path.append(next_node.label)
                return next_node
            # TODO: Generate stats not related to branching factor here

            children = self._expand_node(next_node)
            legal_children = self._find_legal_children(children)
            self._add_legal_children(legal_children)

            if self.verbose:
                print(f'VERBOSE:\nEXPANDED: {next_node}\nCHILDREN: '
                      f'{legal_children}\nFRONTIER: {self.frontier}\nVISITED: {self.visited}\n')

            self.expansions_taken += 1

        return next_node

    def _expand_node(self, node):
        """ Gets all children of the given node
        """
        children = []
        for edge in self.graph.edges[node.label]:
            new_path = node.path.copy()
            new_path.append(edge.start)
            new_cost = node.cost + edge.length

            child = SearchNode(edge.end, path=new_path, cost=new_cost)
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
            if child in self.frontier:
                existing_idx = self.frontier.index(child)
                existing_copy = self.frontier[existing_idx]

                if (self.algorithm == 'BEST'
                        and child.cost < existing_copy.cost) \
                        or self.algorithm == 'DFS' \
                        or self.algorithm == 'IDS':
                    legal_children.append(child)
                elif self.algorithm == 'A*':
                    pass
            else:
                legal_children.append(child)

        return legal_children

    def _add_legal_children(self, legal_children):
        """ Add legal children to the frontier and visited list
        """
        if self.algorithm == 'BFS':
            self.frontier.extend(legal_children)
        elif self.algorithm == 'DFS':
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
        self.path = path
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
