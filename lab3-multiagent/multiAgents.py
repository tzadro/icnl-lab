# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects aend autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        if successorGameState.isWin():  # if next state is win return +inf
            return 999999

        # find manhattan distance to closest ghost
        ghostDist = min([util.manhattanDistance(newPos, ghostPos) for ghostPos in successorGameState.getGhostPositions()])

        # find manhattan distance to closest food
        foodDist = min([util.manhattanDistance(newPos, foodPos) for foodPos in successorGameState.getFood().asList()])
        numFood = len(successorGameState.getFood().asList())

        return (successorGameState.getScore() * ghostDist) / (foodDist * numFood)  # prefer being further to ghost and closer to food

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        return self.minimax(gameState, self.depth)[1]

    def minimax(self, gameState, depth, agent=0):
        if gameState.isWin() or gameState.isLose() or depth == 0:  # if terminal state or too deep
            return self.evaluationFunction(gameState)  # return evaluation of current state

        if agent == 0:  # if pacman then maximize
            values = []

            for action in gameState.getLegalActions(agent):  # for every action pacman can take
                nextState = gameState.generateSuccessor(agent, action)  # translate it to state it would lead to

                value = self.minimax(nextState, depth, agent + 1)  # and calculate recursively minimax for next agent

                values.append((value, action))  # add value and action leading to that value so we can use it later

            return max(values)  # since it's player's turn find maximum value
        else:  # else it's one of the ghost and minimize
            values = []

            for action in gameState.getLegalActions(agent):  # for every action current ghost can take
                nextState = gameState.generateSuccessor(agent, action)  # translate it to a state it would lead to

                if agent == gameState.getNumAgents() - 1:  # if we are on the last ghost
                    value = self.minimax(nextState, depth - 1, 0)  # call again for pacman but deduct depth
                else:
                    value = self.minimax(nextState, depth, agent + 1)  # else we find min for next ghost

                values.append(value)

            return min(values)  # since it's opponent's turn find minimum value

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        alpha = -999999  # -infinity
        beta = 999999  # +infinity
        value = (-999999, None)  # None is placeholder for action

        # next part is maximizing player but adapted so we can know what action lead to best result
        # only difference from the maximizing part in alphabeta is that value is tuple also containing action
        # and we don't need to check for pruning since it is root state
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)

            value = max(value, (self.alphabeta(nextState, self.depth, alpha, beta, 1), action))

            alpha = max(alpha, value[0])

        return value[1]

    def alphabeta(self, gameState, depth, alpha, beta, agent):
        if gameState.isWin() or gameState.isLose() or depth == 0:  # if terminal state of too deep
            return self.evaluationFunction(gameState)  # return evaluation of current state

        if agent == 0:  # if pacman then maximize
            value = -999999  # -infinity

            for action in gameState.getLegalActions(0):  # for every action pacman can take
                nextState = gameState.generateSuccessor(agent, action)  # translate it to action it leads to

                value = max(value, self.alphabeta(nextState, depth, alpha, beta, agent + 1))

                if value > beta:  # if current best value is bigger than what is worst value for minimizing player
                    return value  # then further expanding won't make difference and beta prune

                alpha = max(alpha, value)  # update alpha

            return value
        else:  # else it's opponent's turn and minimize
            value = 999999

            for action in gameState.getLegalActions(agent):  # for every action pacman can take
                nextState = gameState.generateSuccessor(agent, action)  # translate it to action it leads to

                if agent == gameState.getNumAgents() - 1:  # if last ghost
                    value = min(value, self.alphabeta(nextState, depth - 1, alpha, beta, 0))  # then go deeper
                else:
                    value = min(value, self.alphabeta(nextState, depth, alpha, beta, agent + 1))  # else next ghost

                if value < alpha:  # if current best value is smaller than what is worst value for maximizing player
                    return value  # then further expanding won't make difference and alpha prune

                beta = min(beta, value)  # update beta

            return value

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

