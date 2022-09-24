from math import sqrt, inf, acos
from collections import namedtuple


Vec2 = namedtuple('Vec2', 'x y')


class Searcher:
    def __init__(self, graph, algorithm, start, goals, expansions, heuristic,
                 verbose):
        self.graph = graph
        self.algorithm = algorithm

        self.start = SearchNode(start, self.graph.nodes[start])
        self.goals = \
            [SearchNode(goal, self.graph.nodes[goal]) for goal in goals]

        self.expansions = expansions
        self.expansions_taken = 0
        self.heuristic = heuristic
        self.verbose = verbose

        # Currently open nodes
        self.frontier = [self.start]
        self.current_node = None

        self.avg_open = 0
        self.max_open = 0

        self.avg_depth = 0
        self.max_depth = 0

        self.avg_branching = 0

    def search(self):
        while self.expansions == 0 or self.expansions_taken < self.expansions:
            self.expansions_taken += 1

            self.current_node = self.frontier.pop(0)
            self._update_open_depth()

            # Stop if we are at a goal
            if self.current_node in self.goals:
                break

            # Get our children and add them to the frontier appropriately
            children = self._expand_node()
            self._update_branching(children)
            legal_children = self._find_legal_children(children)
            added_children = self._add_children(legal_children)

            # Print the extra verbose stats
            if self.verbose:
                print(f'VERBOSE:\nEXPANDED: {self.current_node}\nCHILDREN:'
                      f' {children}\nLEGAL_CHILDREN: {legal_children}\n'
                      f'ADDED CHILDREN: {added_children}\nFRONTIER:'
                      f' {self.frontier}\n')

        # Divide to get our averagess
        self.avg_open /= self.expansions_taken
        self.avg_depth /= self.expansions_taken
        self.avg_branching /= self.expansions_taken

        return self.current_node

    def _expand_node(self):
        """ Gets all children of the current node
        """
        children = []
        for edge in self.graph.edges[self.current_node.label]:
            child = SearchNode(edge.end, self.graph.nodes[edge.end],
                               path=self.current_node.path,
                               cost=self.current_node.cost + edge.length)
            children.append(child)

        return children

    def _find_legal_children(self, children):
        """ Determines which of the children of the node might actually be
        added to the frontier
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
                # For all other algorithms we may keep the new one
                else:
                    legal_children.append(child)
            else:
                legal_children.append(child)

        return legal_children

    def _add_children(self, legal_children):
        """ Add children to the frontier and remove from frontier as necessary
        to prevent duplicates. Return the children that were actually added
        """
        added_children = []

        # In DFS we append children alphabetically only if they aren't in the
        # frontier yet
        if self.algorithm == 'BFS':
            added_children = legal_children.copy()
            self.frontier.extend(legal_children)
        # In DFS we prepend children alphabetically and remove old copies of
        # them from the frontier if they were already there
        elif self.algorithm == 'DFS':
            # Children is alphabetical, so we want to prepend them in reverse
            # so they will be in alphabetical order in the frontier in the end
            for child in reversed(legal_children):
                if child in self.frontier:
                    self.frontier.remove(child)

                self.frontier.insert(0, child)
                added_children.insert(0, child)
        # In BEST remove the worst copy of a node from the frontier and sort
        # the frontier based on distance to goal
        elif self.algorithm == 'BEST':
            for child in legal_children:
                child.heuristic_cost = self._SLD(child)

                if child in self.frontier:
                    existing = self.frontier[self.frontier.index(child)]
                    if child.cost < existing.cost:
                        self.frontier.remove(existing)
                        self.frontier.append(child)
                else:
                    self.frontier.append(child)

            self.frontier.sort(key=lambda node: node.heuristic_cost)
        elif self.algorithm == 'IDS':
            pass
        else:
            for child in legal_children:
                if self.heuristic == 'SLD':
                    child.heuristic_cost = self._H(child, self._SLD) + child.cost
                else:
                    child.heuristic_cost = self._H(child, self._DIR)

                if child in self.frontier:
                    existing = self.frontier[self.frontier.index(child)]
                    if self.heuristic == 'SLD' and child.heuristic_cost < existing.heuristic_cost:
                        self.frontier.remove(existing)
                        self.frontier.append(child)
                    elif child.cost < existing.cost:
                        self.frontier.remove(existing)
                        self.frontier.append(child)
                else:
                    self.frontier.append(child)
            self.frontier.sort(key=lambda node: node.heuristic_cost)

        return added_children

    def _H(self, node, func):
        """ Takes ina heuristic and finds the min value from current node to a
        goal for the given heuristic
        """
        smallest = inf

        for goal in self.goals:
            new = func(node, goal)
            smallest = min(smallest, new)

        return smallest

    def _SLD(self, start, end):
        """ Returns the Euclidian distance between start and end
        """
        return sqrt((start.x - end.x)**2 + (start.y - end.y)**2)

    def _DIR(self, node, goal):
        """ Returns the difference in heading between a straight path from the
        current node to the goal and a straight path from the current node to
        the given node
        """
        current_to_goal = goal - self.current_node
        current_to_node = node - self.current_node

        dot_product = self._dot(current_to_goal, current_to_node)

        goal_dist = self._SLD(self.current_node, goal)
        node_dist = self._SLD(self.current_node, node)

        return acos(dot_product / (goal_dist * node_dist))

    def _dot(self, vec1, vec2):
        """ Calculates the dot product between two vectors
        """
        return vec1.x * vec2.x + vec1.y * vec2.y

    def _update_open_depth(self):
        self.avg_open += len(self.frontier)
        if len(self.frontier) > self.max_open:
            self.max_open = len(self.frontier)

        self.avg_depth += self.current_node.depth
        if self.current_node.depth > self.max_depth:
            self.max_depth = self.current_node.depth

    def _update_branching(self, children):
        self.avg_branching += len(children)


class SearchNode:
    """ This represents a node identified in the search. It holds the node's
    label, the cost to reach it, and the path taken
    """

    def __init__(self, label, coords, path=[], cost=0):
        self.label = label

        self.x = coords.x
        self.y = coords.y

        # The path we took to this node
        self.path = path.copy()
        self.path.append(self)

        # The cost to reach this node from the start
        self.cost = cost
        # The chost assigned to this node under a given heuristic (SLD or DIR)
        self.heuristic_cost = 0

    @property
    def depth(self):
        return len(self.path)

    def __eq__(self, other):
        """ We only really care if the objects are referring to the same node
        on the graph not the same path to the node b/c we are using this to
        dedup nodes in the frontier
        """
        if isinstance(other, SearchNode):
            return self.label == other.label
        else:
            return self.label == other

    def __repr__(self):
        return self.label

    def __sub__(self, other):
        """ Return a vector between this node and another node
        """
        return Vec2(self.x - other.x, self.y - other.y)
