from collections import namedtuple


Point = namedtuple('Point', 'x y')
Edge = namedtuple('Edge', 'start end length')


class Graph:
    def __init__(self, graph_fp):
        self.file = graph_fp
        # Maps node label to its spacial coordinates
        self.nodes = {}
        # Maps node label a list of egdes with the key node as start
        self.edges = {}
        # Populate nodes and edges
        self._create_graph(graph_fp)
        # Sort all edge lists alphabetically by their end node. This will be
        # useful for our searches
        for edges in self.edges.values():
            edges.sort(key=lambda edge: edge.end)

    def _create_graph(self, graph_fp):
        # Read all lines from our file
        with open(graph_fp) as fp:
            lines = fp.readlines()

        # Clean all lines
        cleanlines = [line.strip() for line in lines]
        for line in cleanlines:
            start, end, length, start_pos, end_pos = self._parse_line(line)

            if start not in self.edges.keys():
                self.edges[start] = []

            if end not in self.edges.keys():
                self.edges[end] = []

            # Keep track of the edge under both ends going both ways
            self.edges[start].append(Edge(start, end, length))
            self.edges[end].append(Edge(end, start, length))

            # Doesn't matter if we overwrite these because it will be the same
            # every time (or it should be unless our graph is messed up)
            self.nodes[start] = start_pos
            self.nodes[end] = end_pos

    def _parse_line(self, line):
        # Remove extraneous formatting characters
        line = line.replace('(', '').replace(')', '').replace("'", '') \
                   .replace('[', '').replace(']', '').replace(' ', '')

        split_line = line.split(',')
        start, end, length = split_line[:3]
        length = int(length)

        start_x, start_y = split_line[3:5]
        start_pos = Point(int(start_x), int(start_y))

        end_x, end_y = split_line[5:7]
        end_pos = Point(int(end_x), int(end_y))

        return start, end, length, start_pos, end_pos

    def __repr__(self):
        repr = 'NODES:\n'
        for k, v in self.nodes.items():
            repr = repr + f'{k}: {v}\n'

        repr = repr + '\nEDGES:\n'
        for k, v in self.edges.items():
            repr = repr + f'{k}: {v}\n'

        return repr
