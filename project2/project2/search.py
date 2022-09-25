from math import sqrt, inf, acos
from collections import namedtuple

import matplotlib.pyplot as plt

from .graph import Point


Vec2 = namedtuple('Vec2', 'x y')


class Searcher:
    def __init__(self, graph, algorithm, start, goals, expansions=0,
                 heuristic=None, verbose=False, graph_viz=None):
        self.graph = graph
        self.algorithm = algorithm

        self.start = SearchNode(start, self.graph.nodes[start])
        self.goals = \
            [SearchNode(goal, self.graph.nodes[goal]) for goal in goals]

        self.expansions = expansions
        self.expansions_taken = 0
        self.heuristic = heuristic

        self.IDS_limit = 3
        self.IDS_step = 3

        self.verbose = verbose

        # Currently open nodes
        self.frontier = [self.start]
        self.current_node = None

        self.avg_open = 0
        self.max_open = 0

        self.avg_depth = 0
        self.max_depth = 0

        self.avg_branching = 0

        # I wanted these prints in main, but the submission is easier if they
        # are here
        print(f"You have asked for a/an '{self.algorithm}' type search to be"
              f" run on the graph specified by the file '{graph.file}.'")

        if algorithm == 'A*':
            print(f"Heuristic '{self.heuristic}' will be used.")

        print(f"Your start node is '{self.start}' and your goal(s) are"
              f" '{self.goals}'.\nUp to '{self.expansions}' expansions will"
              " be done in search of the goal (0 means no limit)\n")

        self.graph_viz = graph_viz
        if self.graph_viz:
            self._plotStart()
            plt.show()


    def search(self):
        while self.expansions == 0 or self.expansions_taken < self.expansions:
            self.expansions_taken += 1

            self.current_node = self.frontier.pop(0)
            self._update_open_depth()

            if self.verbose:
                print(f"VERBOSE:\nEXPANDED: {self.current_node}")
            # Stop if we are at a goal
            if self.current_node in self.goals:
                break

            # Get our children and add them to the frontier appropriately
            children = self._expand_node()
            self._update_branching(children)
            legal_children = self._find_legal_children(children)
            added_children = self._add_children(legal_children)

            if self.graph_viz:
                self._plotCurrent(added_children)
                plt.show()

            # Print the extra verbose stats
            if self.verbose:
                print(f'CHILDREN:  {children}\nLEGAL_CHILDREN:'
                      f' {legal_children}\nADDED CHILDREN:'
                      f' {added_children}\nFRONTIER: {self.frontier}\n')

        # Divide to get our averagess
        self.avg_open /= self.expansions_taken
        self.avg_depth /= self.expansions_taken
        self.avg_branching /= self.expansions_taken

        if self.graph_viz:
            self._plotEnd()
            plt.show()

        end = self.current_node
        # Formatting print
        if self.verbose:
            print()
        print(f'FOUND GOAL:\nLABEL: {end.label}\nPATH: {end.path}\nCOST:'
            f' {end.cost}\n')
        print('STATS:\n'
            'AVG_OPEN: {:.2f}\nMAX_OPEN: {:.2f}\n\n'
            'AVG_DEPTH: {:.2f}\nMAX_DEPTH: {:.2f}\n\n'
            'AVG_BRANCHING_FACTOR: {:.2f}'.format(
                self.avg_open, self.max_open, self.avg_depth,
                self.max_depth, self.avg_branching))

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
        # Since IDS is DFS at heart, it follows a similar pattern
        elif self.algorithm == 'IDS':
            for child in reversed(legal_children):
                if child in self.frontier:
                    self.frontier.remove(child)

                if child.depth > self.IDS_limit:
                    self.frontier.append(child)
                else:
                    self.frontier.insert(0, child)

                added_children.append(child)

            if self.frontier[0].depth > self.IDS_limit:
                self.IDS_limit += self.IDS_step
        # In BEST remove the worst copy of a node from the frontier and sort
        # the frontier based on distance to goal
        elif self.algorithm == 'BEST':
            for child in legal_children:
                if not child.heuristic_cost:
                    child.heuristic_cost = self._H(child, self._SLD)

                if child in self.frontier:
                    existing = self.frontier[self.frontier.index(child)]
                    if child.cost < existing.cost:
                        self.frontier.remove(existing)
                        self.frontier.append(child)
                else:
                    self.frontier.append(child)

            self.frontier.sort(key=lambda node: node.heuristic_cost)
        # The behavior of A* depends on the heuristic. It keeps whichever path
        # it deems better based on the given heuristic
        elif self.algorithm == 'A*':
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
                        added_children.append(child)
                    elif self.heuristic == 'DIR' and child.cost < existing.cost:
                        self.frontier.remove(existing)
                        self.frontier.append(child)
                        added_children.append(child)
                    else:
                        raise ValueError(f'Unknown heuristic: {self.heuristic}')
                else:
                    self.frontier.append(child)
                    added_children.append(child)
            self.frontier.sort(key=lambda node: node.heuristic_cost)
        else:
            raise ValueError(f'Unknown algorithm: {self.algorithm}')

        return added_children

    def _H(self, node, func):
        """ Takes in a heuristic and finds the min value from current node to a
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

        # I had floating point error give me a 1.0000000000000002 which of
        # course caused acos to fail, so this prevents that
        # This happened from A to I on 10node with a* DIR
        angle = dot_product / (goal_dist * node_dist)
        try:
            angle = acos(angle)
        except ValueError as e:
            if "math domain error" in str(e):
                if angle > 1:
                    angle = 1
                elif angle < -1:
                    angle = -1
            else:
                raise e
        finally:
            return angle

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

    def _plotStart(self):
        self.graph_viz.plot()
        self.graph_viz.markStart(self.start.label)
        for goal in self.goals:
            self.graph_viz.markGoal(goal.label)

    def _plotCurrent(self, added_children):
        self._plotStart()
        self.graph_viz.exploreNode(self.current_node.label, [node.label for node in self.current_node.path])
        self.graph_viz.exploreEdges(self.current_node.label, [child.label for child in added_children])

    def _plotEnd(self):
        self._plotStart()
        self.graph_viz.paintPath(
            [node.label for node in self.current_node.path])

class SearchNode:
    """ This represents a node identified in the search. It holds the node's
    label, the cost to reach it, and the path taken
    """

    def __init__(self, label, coords=Point(0, 0), path=[], cost=0, heuristic_cost=0):
        self.label = label

        self.x = coords.x
        self.y = coords.y

        # The path we took to this node
        self.path = path.copy()
        self.path.append(self)

        # The cost to reach this node from the start
        self.cost = cost
        # The cost assigned to this node under a given heuristic (SLD or DIR)
        self.heuristic_cost = heuristic_cost

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
        if self.heuristic_cost:
            return f'({self.label}, {self.heuristic_cost:0.2f})'
        else:
            return f'({self.label}, {self.cost:0.2f})'


    def __sub__(self, other):
        """ Return a vector between this node and another node
        """
        return Vec2(self.x - other.x, self.y - other.y)
