from sys import argv

from util.graphviz import GraphViz
from graph import Graph
from search import Searcher

if __name__ == '__main__':
    """ This should be caled with the following format

    graph_fp: a valid filepath to an existing graph file of format specified in
    graphviz.py

    start: a node inside the provided graph to start searching from

    goals: a list of goals nodes comma seperated that should all exist in the
    graph

    expansions: the number of expansions to run before giving up. A value of 0
    will cause the program to run as many expansions as it needs to find a
    solution (or until you kill it because it was taking too long)

    algorithm: BFS, DFS, BEST, IDS, A* (not case sensitive)

    if algorithm is A*:
        heuristic: sld or dir

    the --verbose flag sets verbose mode
    """
    # Get our arguments
    graph_fp = argv[1]
    start = argv[2]
    goals = argv[3].split(',')
    expansions = int(argv[4])
    algorithm = argv[5].upper()

    heuristic = None
    verbose = False
    visual = False

    if algorithm == 'A*':
        heuristic = argv[6].upper()

    if '--verbose' in argv:
        verbose = True

    graph = Graph(graph_fp)
    searcher = Searcher(
        graph, algorithm, start, goals, expansions, heuristic, verbose)
    goal = searcher.search()

