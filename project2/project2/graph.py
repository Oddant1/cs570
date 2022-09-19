from collections import namedtuple


Point = namedtuple('Point', 'x y')
Edge = namedtuple('Edge', 'label length')

class Graph:
    def __init__(self, graph_fp):
        self.nodes = {}
        self._create_graph(graph_fp)

    def _create_graph(self, graph_fp):
        # Read all lines from our file
        with open(graph_fp) as fp:
            lines = fp.readlines()

        # Clean all lines
        cleanlines = [line.strip() for line in lines]
        for line in cleanlines:
            start, end, length, start_pos, end_pos = self._parse_line(line)

            if start not in self.nodes.keys():
                start_node = Node(start, start_pos)
            else:
                start_node = self.nodes[start]

            if end not in self.nodes.keys():
                end_node = Node(end, end_pos)
            else:
                end_node = self.nodes[end]

            start_node.add_edge(end_node.label, length)
            end_node.add_edge(start_node.label, length)

            # This will overwrite the existing nodes if they did exist with an
            # updated version of themselves which isn't really necessary, but
            # it simplifies the logic. We don't need to check again if the node
            # is already in the dictionary because we don't care
            self.nodes[start] = start_node
            self.nodes[end] = end_node

        # Sort all node's edge lists in alphabetical order
        for node in self.nodes.values():
            node.sort_edges()

    def _parse_line(self, line):
        # Remove extraneous formatting characters
        line = line.replace('(', '').replace(')', '').replace("'", '') \
                   .replace('[', '').replace(']', '').replace(' ', '')

        split_line = line.split(',')
        print(split_line)
        start, end, length = split_line[:3]
        length = int(length)

        start_x, start_y = split_line[3:5]
        start_pos = Point(int(start_x), int(start_y))

        end_x, end_y = split_line[5:7]
        end_pos = Point(int(end_x), int(end_y))

        return start, end, length, start_pos, end_pos

    def __repr__(self):
        repr = ''
        for k, v in self.nodes.items():
            repr = repr + f'{k}: {v.edges}\n'
        return repr


class Node:
    """ The graph creates an instance of this class for every node contained
    within
    """
    def __init__(self, label, pos):
        self.label = label
        self.pos = pos
        self.edges = []

    def add_edge(self, label, length):
        self.edges.append(Edge(label, length))

    def sort_edges(self):
        """ Get the edges in alphabetical order once we've finalized the graph
        so when we open them later they're already alphabetized
        """
        self.edges.sort(key=lambda e: e.label)
