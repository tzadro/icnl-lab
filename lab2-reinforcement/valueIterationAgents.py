# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        for _ in range(iterations):  # do value iteration for this many times
            nextValues = util.Counter()  # init values_k+1 but still use self.values as values_k

            for state in mdp.getStates():  # for every state
                action = self.computeActionFromValues(state)  # calculate best action
                nextValues[state] = self.computeQValueFromValues(state, action)  # set value_k+1 as val for that action

            self.values = nextValues  # set values_k+1 as values_k for next iteration


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        if action is None:  # if we reached terminal state
            return 0  # value is 0

        return sum([prob * (self.mdp.getReward(state, action, nextState) + self.discount * self.getValue(nextState)) for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action)])
        # return sum of values for every possible outcome of doing action in state
        # where value is probability of getting to next state * (reward of action in state leading to next state + discounted value of next state)

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if self.mdp.isTerminal(state):  # if state is terminal
            return None  # we don't need to take any action

        return max([(self.computeQValueFromValues(state, action), action) for action in self.mdp.getPossibleActions(state)])[1]
        # make list of pairs (value of action in state, action) for all possible actions in given state
        # and return action of maximum pair compared by values

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
