from sys import argv

import util
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

    print(f"You have asked for a/an '{algorithm}' type search to be run on the"
          f" graph specified by the file '{graph_fp}.'")

    if algorithm == 'A*':
        print(f"Heuristic '{heuristic}' will be used.")

    print(f"Your start node is '{start}' and your goal(s) are '{goals}'.\n"
          f"Up to '{expansions}' expansions will be done in search of the goal"
          " (0 means no limit)\n")
    # graph = util.graphviz.GraphViz()
    # graph.loadGraphFromFile(graph_fp)
    # graph.plot()
    # graph.markStart(start)

    # for goal in goals:
    #     graph.markGoal(goal)
    graph = Graph(graph_fp)
    # print(graph)
    searcher = Searcher(
        graph, algorithm, start, goals, expansions, heuristic, verbose)
    goal = searcher.search()
    print(f'LABEL: {goal.label}\nPATH: {goal.path}\nCOST: {goal.cost}\n')
    print('STATS:\n'
          'AVG_OPEN: {:.2f}\nMAX_OPEN: {:.2f}\n\n'
          'AVG_DEPTH: {:.2f}\nMAX_DEPTH: {:.2f}\n\n'
          'AVG_BRANCHING_FACTOR: {:.2f}'.format(
            searcher.avg_open, searcher.max_open, searcher.avg_depth,
            searcher.max_depth, searcher.avg_branching))
