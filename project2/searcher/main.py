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

    algorithm: bfs, dfs, best, ids, a* (not case sensitive)

    if algorithm is a*:
        heuristic: 1 or 2

    if there is an additional argument, it will be interpreted as the verbose
    flag being set
    """
    # Get our arguments
    graph_fp = argv[1]
    start = argv[2]
    goals = argv[3].split(',')
    expansions = int(argv[4])
    algorithm = argv[5].upper()

    heuristic = None
    verbose = False
    if algorithm == 'a*':
        heuristic = int(argv[6])

        if len(argv) == 8:
            verbose = True
    elif len(argv) == 7:
        verbose = True

    print(f"You have asked for a/an '{algorithm}' type search to be run on the"
          f" graph specified by the file '{graph_fp}.'")

    if algorithm == 'a*':
        print(f"Heuristic '{heuristic}' will be used.")

    print(f"Your start node is '{start}' and your goal(s) are '{goals}'.\n"
          f"Up to '{expansions}' expansions will be done in search of the goal"
          " (0 means no limit)")
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
    print(f'LABEL: {goal.label}\nPATH: {goal.path}\nCOST: {goal.cost}')
