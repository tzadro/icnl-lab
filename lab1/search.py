# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    return search(problem, util.Stack())


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    return search(problem, util.Queue())


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    def priority(state):  # define how to calculate priority of a state
        return state[1]  # position 2 marks value, based on definition of elements in fringe from search function

    return search(problem, util.PriorityQueueWithFunction(priority))


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    def priority(state):  # define how to calculate priority of a state
        return state[1] + state[2]  # position 2 marks g and position 3 marks h,
        # based on definition of elements in fringe from search function

    return search(problem, util.PriorityQueueWithFunction(priority), heuristic)


def search(problem, fringe, heuristic=nullHeuristic):
    """
    :param heuristic: heuristic, needed only for A* search
    :param problem: search space definition
    :param fringe: structure defining priority by which nodes are picked from the fringe
                  [(node, cost up to that point, action leading to that node, parent node)]
    :return: list of actions leading to problem's goal node
    """
    expanded = {}  # {key: node, value: (cost up to that point, action leading to that node, parent node)}
    fringe.push((problem.getStartState(), 0, 0, None, None))  # start search from first node

    while not fringe.isEmpty():  # if this triggers it means we can't reach goal from starting node
        node, value, _, leading, parent = fringe.pop()  # take first from fringe, priority defined by fringe type

        if node in expanded:  # this additional check avoids problem of adding multiple same nodes to fringe
            continue  # because we haven't expanded it yet

        expanded[node] = (value, leading, parent)  # node is considered expanded after we fetch his successors,
        # but we update it here so backtrack could work properly

        if problem.isGoalState(node):  # if current node is our goal
            return backtrack(node, expanded)  # then end search by backtracking through the graph

        for successor, action, cost in problem.getSuccessors(node):  # for all successors of the node
            if successor in expanded:  # if they have already been expanded
                continue  # then continue to the next successor

            g = value + cost  # cost to get from problem's start node to this successor
            h = heuristic(successor, problem)  # approximated cost of getting from this successor to end node
            fringe.push((successor, g, h, action, node))  # add successor to expansion fringe


def backtrack(start, nodes):  # backtracks through expanded nodes to return list of actions leading from problem's start
    actions = []  # action
    node = start  # start from given node

    while True:  # if we reached goal state from problem's start this should always finish
        _, action, parent = nodes[node]  # for current node get action leading to that node and it's parent node

        if parent is None:  # if parent is None it means we have reached problem's start state
            return list(reversed(actions))  # since we started from goal state return reversed list of recorded actions

        actions.append(action)  # add action leading to this state to list of actions
        node = parent  # continue from parent state


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
