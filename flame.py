# myTeam.py
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

import random
import time

import game
import util
from captureAgents import CaptureAgent
from game import Directions

#################
# Team creation #
#################


def createTeam(firstIndex,
               secondIndex,
               isRed,
               first='ReinforcementAgent',
               second='DummyAgent',
               numTraining=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    # The following line is an example only; feel free to change it.
    return [
        eval(first)(firstIndex, numTraining=numTraining),
        eval(second)(secondIndex)
    ]


##########
# Agents #
##########


class ReinforcementAgent(CaptureAgent):
    """Documentation for ReinforcementAgent

    """

    def __init__(self,
                 index,
                 timeForComputing=0.1,
                 learning_rate=0.2,
                 exploration_rate=0.05,
                 discount=0.8,
                 numTraining=0):
        CaptureAgent.__init__(self, index, timeForComputing)
        self.alpha = float(learning_rate)
        self.epsilon = float(exploration_rate)
        self.gamma = float(discount)
        self.numTraining = int(numTraining)
        self.has_no_observation = True
        self.episodesSoFar = 0
        # Q-VALUES
        self.qValues = util.Counter()

    def isInTraining(self):
        return self.episodesSoFar < self.numTraining

    def isInTesting(self):
        return not self.isInTraining()

    def getLegalActions(self, state):
        return state.getLegalActions(self.index)

    def observationFunction(self, state):
        if not self.has_no_observation and \
           self.getPreviousObservation() is not None:
            delta_reward = self.getCurrentObservation().getScore() - \
                           self.getPreviousObservation().getScore()
            self.update(self.getPreviousObservation(), self.last_action,
                        self.getCurrentObservation(), delta_reward)
            # episode rewards?
        return CaptureAgent.observationFunction(self, state)

    def update(self, state, action, nextState, reward):
        pass

    def chooseAction(self, gameState):
        self.has_no_observation = False
        self.last_action = self.actionSelector(gameState)
        return self.last_action

    def actionSelector(self, gameState):
        return random.choice(self.getLegalActions(gameState))

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        # starting episode?

    def final(self, gameState):
        delta_reward = self.getCurrentObservation().getScore() - \
                           self.getPreviousObservation().getScore()
        self.update(self.getPreviousObservation(), self.last_action,
                    self.getCurrentObservation(), delta_reward)
        # episode rewards?
        self.stopEpisode()
        if self.isInTraining():
            print 'Training:{'
        else:
            print 'Testing:{'
        print self.getCurrentObservation().getScore()
        print '}'
        CaptureAgent.final(self, gameState)

    def stopEpisode(self):
        self.episodesSoFar += 1
        self.has_no_observation = True
        # if self.episodesSoFar < self.numTraining:
        #     self.accumTrainRewards += self.episodeRewards
        # else:
        #     self.accumTestRewards += self.episodeRewards
        # self.episodesSoFar += 1
        if self.episodesSoFar >= self.numTraining:
            # Take off the training wheels
            self.epsilon = 0.0    # no exploration
            self.alpha = 0.0    # no learning

    def getQValue(self, state, action):
        return self.qValues[(state, action)]

    def getPolicyValue(self, state):
        possibleActions = self.getLegalActions(state)
        if possibleActions:
            maximum_q = float('-inf')
            bestAction = None
            for action in possibleActions:
                q = self.getQValue(state, action)
                if q >= maximum_q:
                    maximum_q = q
                    bestAction = action
            return bestAction, maximum_q
        return None, float('-inf')

    def getPolicy(self, state):
        return self.getPolicyValue()[0]

    def getValue(self, state):
        return self.getPolicyValue()[1]


class DummyAgent(CaptureAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at baselineTeam.py for more details about how to
    create an agent as this is the bare minimum.
    """

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)

        IMPORTANT: This method may run for at most 15 seconds.
        """
        '''
        Make sure you do not delete the following line. If you would like to
        use Manhattan distances instead of maze distances in order to save
        on initialization time, please take a look at
        CaptureAgent.registerInitialState in captureAgents.py.
        '''
        CaptureAgent.registerInitialState(self, gameState)
        self.plays = 0

    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        actions = gameState.getLegalActions(self.index)
        '''
        You should change this in your own agent.
        '''
        self.plays += 1
        return random.choice(actions)
