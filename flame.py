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
import math
import game
import util
from captureAgents import CaptureAgent
from game import Directions

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout
import numpy as np

#################
# Team creation #
#################


def createTeam(firstIndex,
               secondIndex,
               isRed,
               first='DeepReinforcementAgent',
               second='DeepReinforcementAgent',
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
        eval(second)(secondIndex, numTraining=numTraining)
    ]


##########
# Agents #
##########


class ReinforcementAgentBase(CaptureAgent):
    """Documentation for ReinforcementAgent
    It only need the update and getQValue to get overrided
    """

    def __init__(self, index, timeForComputing, learning_rate,
                 exploration_rate, discount, numTraining):
        CaptureAgent.__init__(self, index, timeForComputing)
        self.alpha = float(learning_rate)
        self.epsilon = float(exploration_rate)
        self.gamma = float(discount)
        self.numTraining = int(numTraining)
        self.has_no_observation = True
        self.episodesSoFar = 0
        self.actionHistory = []

    def isInTraining(self):
        return self.episodesSoFar < self.numTraining

    def isInTesting(self):
        return not self.isInTraining()

    def getLegalActions(self, state):
        return state.getLegalActions(self.index)

    def makeUpdate(self):
        delta_reward = self.getScore(self.getCurrentObservation()) - \
                       self.getScore(self.getPreviousObservation())
        if self.getCurrentObservation().getAgentState(self.index).isPacman:
            delta_reward += 1
        self.update(self.getPreviousObservation(), self.actionHistory[-2],
                    self.getCurrentObservation(), delta_reward)
        # episode rewards?

    def observationFunction(self, state):
        if not self.has_no_observation and \
           self.getPreviousObservation() is not None and \
           self.isInTraining():
            self.makeUpdate()
        return CaptureAgent.observationFunction(self, state)

    def update(self, state, action, nextState, reward):
        pass

    def chooseAction(self, gameState):
        self.has_no_observation = False
        self.actionHistory.append(self.actionSelector(gameState))
        return self.actionHistory[-1]

    def actionSelector(self, gameState):
        if util.flipCoin(self.epsilon):
            return random.choice(self.getLegalActions(gameState))
        return self.getPolicy(gameState)

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        # starting episode?

    def final(self, gameState):
        self.stopEpisode()
        if self.isInTraining():
            self.makeUpdate()
            print 'Training:{'
        else:
            print 'Testing:{'
        print self.getScore(self.getCurrentObservation())
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
            print 'TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT'
            # Take off the training wheels
            self.epsilon = 0.0    # no exploration
            self.alpha = 0.0    # no learning

    def getQValue(self, state, action):
        pass

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
        return self.getPolicyValue(state)[0]

    def getValue(self, state):
        return self.getPolicyValue(state)[1]


class ReinforcementAgent(ReinforcementAgentBase):
    """Documentation for ReinforcementAgent

    """

    def __init__(self,
                 index,
                 timeForComputing=0.1,
                 learning_rate=0.3,
                 exploration_rate=0.25,
                 discount=0.8,
                 numTraining=0):
        ReinforcementAgentBase.__init__(self, index, timeForComputing,
                                        learning_rate, exploration_rate,
                                        discount, numTraining)
        self.qValues = util.Counter()

    def getQValue(self, state, action):
        return self.qValues[(state, action)]

    def update(self, state, action, nextState, reward):
        estimated_q = reward + self.gamma * self.getValue(nextState)
        error = estimated_q - self.getQValue(state, action)
        self.qValues[(state, action)] += self.alpha * error


class DeepReinforcementAgent(ReinforcementAgentBase):
    """Documentation for DeepReinforcementAgent

    """

    def __init__(self,
                 index,
                 timeForComputing=0.1,
                 learning_rate=0.3,
                 exploration_rate=0.01,
                 discount=0.8,
                 numTraining=0,
                 createNewModel=False):
        ReinforcementAgentBase.__init__(self, index, timeForComputing,
                                        learning_rate, exploration_rate,
                                        discount, numTraining)
        self.createNewModel = createNewModel

    def registerInitialState(self, gameState):
        global nn
        self.startPos = gameState.getAgentPosition(self.index)
        if nn is None:
            if self.createNewModel:
                nn = NeuralNetwork(gameState)
            else:
                nn = NeuralNetwork(gameState, modelPath='twoLayerModel.h5')
        self.q_estimator = nn
        ReinforcementAgentBase.registerInitialState(self, gameState)

    def getQValue(self, state, action):
        if self.q_estimator is None:
            self.q_estimator = NeuralNetwork(state)
        prediction = self.q_estimator.predict(self, state, action)
        return prediction

    def update(self, state, action, nextState, reward):
        if self.q_estimator is None:
            self.q_estimator = NeuralNetwork(state)
        estimated_q = reward + self.gamma * self.getValue(nextState)
        estimated_q -= 4 / math.sqrt(
            self.getMazeDistance(state.getAgentPosition(self.index),
                                 self.startPos) + 1)
        self.q_estimator.update(self, state, action, estimated_q)

    def final(self, gameState):
        self.q_estimator.save('twoLayerModel.h5')
        ReinforcementAgentBase.final(self, gameState)


class NeuralNetwork:
    """Documentation for NeuralNetwork

    """

    def __init__(self,
                 state,
                 firstLayerSize=600,
                 secondLayerSize=300,
                 modelPath=None):
        self.walls = state.getWalls()
        self.width = self.walls.width
        self.height = self.walls.height
        self.area = self.width * self.height
        if modelPath is None:
            self.model = Sequential()
            self.model.add(
                Dense(units=firstLayerSize,
                      activation='relu',
                      input_dim=12 * self.area,
                      init='random_uniform'))
            self.model.add(Dropout(0.2))
            self.model.add(
                Dense(units=secondLayerSize,
                      activation='relu',
                      init='random_uniform'))
            self.model.add(Dense(units=1, activation='linear'))
            self.model.compile(optimizer='adam', loss='mean_squared_error')
            self.model.summary()
        else:
            self.model = load_model(modelPath)

    def predict(self, agent, state, action):
        encodedState = self.encodeState(agent, state, action)
        return self.model.predict(encodedState)[0][0]

    def update(self, agent, state, action, q):
        encodedState = self.encodeState(agent, state, action)
        self.model.fit(encodedState, [q], batch_size=1, epochs=1, verbose=0)

    def encodeState(self, agent, state, action):
        # Get all the foods on the ground
        mapWalls = np.zeros((self.width, self.height))
        for x, y in self.walls.asList():
            mapWalls[x, y] = 1
        allMyFoods = np.zeros((self.width, self.height))
        for x, y in agent.getFood(state).asList():
            allMyFoods[x, y] = 1
        allEnemyFoods = np.zeros((self.width, self.height))
        for x, y in agent.getFoodYouAreDefending(state).asList():
            allEnemyFoods[x, y] = 1
        # get all the capsules
        allMyCapsuls = np.zeros((self.width, self.height))
        for x, y in agent.getCapsules(state):
            allMyCapsuls[x, y] = 1
        allEnemyCapsuls = np.zeros((self.width, self.height))
        for x, y in agent.getCapsulesYouAreDefending(state):
            allEnemyCapsuls[x, y] = 1
        # get all the enemies if you can :D
        allPacmanEnemies = np.zeros((self.width, self.height))
        allGhostEnemies = np.zeros((self.width, self.height))
        for enemyIndex in agent.getOpponents(state):
            enemyPos = state.getAgentPosition(enemyIndex)
            if enemyPos is not None:
                if state.getAgentState(enemyIndex).isPacman:
                    allPacmanEnemies[enemyPos[0], enemyPos[1]] = 1
                else:
                    allGhostEnemies[enemyPos[0], enemyPos[1]] = 1
        # get teammate position and set
        teamMate = np.zeros((self.width, self.height))
        agent_1, agent_2 = agent.getTeam(state)
        if agent.index == agent_1:
            teamMatePosition = state.getAgentPosition(agent_2)
        else:
            teamMatePosition = state.getAgentPosition(agent_1)
        teamMate[teamMatePosition[0], teamMatePosition[1]] = 1
        # set the Agent position
        agentGhostPosition = np.zeros((self.width, self.height))
        agentPacmanPosition = np.zeros((self.width, self.height))
        myPos = state.getAgentPosition(agent.index)
        imPacman = state.getAgentState(agent.index).isPacman
        if imPacman:
            agentPacmanPosition[myPos[0], myPos[1]] = 1
        else:
            agentGhostPosition[myPos[0], myPos[1]] = 1
        # set the next position
        agentPacmanNextPosition = np.zeros((self.width, self.height))
        agentGhostNextPosition = np.zeros((self.width, self.height))
        successor = state.generateSuccessor(agent.index, action)
        nextPos = successor.getAgentPosition(agent.index)
        nextImPacman = successor.getAgentState(agent.index).isPacman
        if nextImPacman:
            agentPacmanNextPosition[int(nextPos[0]), int(nextPos[1])] = 1
        else:
            agentGhostNextPosition[int(nextPos[0]), int(nextPos[1])] = 1
        result = np.concatenate(
            (mapWalls, allMyFoods, allEnemyFoods, allMyCapsuls,
             allEnemyCapsuls, allPacmanEnemies, allGhostEnemies, teamMate,
             agentPacmanPosition, agentGhostPosition, agentPacmanNextPosition,
             agentGhostNextPosition)).reshape((1, -1))
        return result

    def save(self, pathname):
        self.model.save(pathname)


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

    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        actions = gameState.getLegalActions(self.index)
        '''
        You should change this in your own agent.
        '''
        #walls = gameState.getWalls()
        #width = walls.width
        #height = walls.height
        #print width, height
        #print self.index
        #print self.getFood(gameState).asList()
        #print '>>>>>>>>>>', gameState.getAgentPosition(0)
        #print self.getTeam(gameState), self.getOpponents(gameState)
        #print gameState.getAgentState(self.index).getPosition()
        #print gameState.getAgentState(0).getPosition()
        #print gameState.getAgentState(0).getDirection()
        #print gameState.getAgentState(0).isPacman
        #x = self.getCurrentObservation()
        #x = self.getCurrentObservation()
        #x = self.getCurrentObservation()
        #for i in range(width):
        #    for j in range(height):
        #        if x[i][j] != gameState[i][j]:
        #            print i, j, x[i][j], gameState[i][j]
        #print self.getFood(gameState)
        #print '**************************************************'
        #print self.getFoodYouAreDefending(gameState)
        #print '**************************************************'
        #print gameState
        #print self.getCurrentObservation()
        #print np.array(gameState, dtype=np.int)
        return random.choice(actions)


nn = None
