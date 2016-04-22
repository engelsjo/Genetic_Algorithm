import os
import sys
from random import randint

class GeneticAlgorithm(object):
    def __init__(self, initialPopulationSize=100, numChromosomeBits=16, minRange=-100, maxRange=100, evalFunction=None, evalInputs=0):
        self.initPopulationSize = initialPopulationSize
        self.nbrOfChromosomeBits = numChromosomeBits
        self.minRange = minRange
        self.maxRange = maxRange
        self.currentPopulation = []
        self.evalFunction = evalFunction
        self.evalNumberOfInputs = evalInputs
        self.populationInitialization()
        self.shift = self.getShiftValue()
        self.scalar = self.getScalar()
        self.numberOfGenerations = 0
        self.tourneyKval = 2 # using binary tourney selection for now - we can play with this knob later

    ############# Evolutionary Process Methods #############

    def populationInitialization(self):
        # create as many chromosomes as specified in self.initPopulationSize
        for chromosomeCounter in range(self.initPopulationSize):
            chromesome = [] # make a new array which will hold our bits
            for bitCounter in range(self.nbrOfChromosomeBits):
                bit = str(randint(0,1))
                chromesome.append(bit)
            self.currentPopulation.append(chromesome)

    def populationEvaluation(self):
        # make sure that the evaluation function and number of inputs have been set
        if self.evalFunction == None or self.evalNumberOfInputs == 0:
            print("Error: you must first set the evaluation function and the number of input values to said function")
            sys.exit(-2)

        # run our evaluation on all of our chromosomes
        chromosomesWithEvals = []
        for chromosome in self.currentPopulation:
            # get a list of the evaluation function inputs from each chromosome - encoded as a bit str
            inputValues = self.getInputsFromBitStr(chromosome)
            evaluation = self.evalFunction(inputValues)
            chromosomesWithEvals.append((chromosome, evaluation)) # append a tuple
        # sort our chromosomes by evaluation to speed up performance of tourney selection
        return sorted(chromosomesWithEvals, key=lambda x : x[1])[::-1] # reverse because we are using min values

    def populationSelection(self, chromosomesWithEvals):
        selectedParents = []
        while len(selectedParents) < len(self.currentPopulation):
            parentSelection = self.tourneySelect(chromosomesWithEvals)
            selectedParents.append(parentSelection)
        return selectedParents
            
    def populationVariation(self, breeders):
        numberOfVariations = len(breeders) / 2
        children = []
        for i in range(numberOfVariations):
            parent1 = breeders[i][0]
            parent2 = breeders[i + 1][0]
            child1, child2 = self.crossover(parent1, parent2)
            randomNumber = randint(1, 1000)
            if randomNumber == 1: # mutate only 1/1000
                child1 = self.mutate(child1)
            children.append(child1)
            children.append(child2)  
        return breeders, children  
        
    def populationUpdate(self, breeders, children):
        # now choose some portion of parents and some portion of crossover / mutated children
        # I am taking the elitist approach by keep the top 10 percent of parents and replacing the rest of the
        # population with randomly selected children
        survivors = []
        breeders = sorted(breeders, key=lambda x: x[1])[::-1]
        numberOfParentSurvivors = int(.1 * len(self.currentPopulation))
        for i in range(numberOfParentSurvivors):
            survivors.append(breeders[i][0])
        numberOfChildrenSurvivors = len(self.currentPopulation) - numberOfParentSurvivors
        for j in range(numberOfChildrenSurvivors):
            survivors.append(children[j])
        self.currentPopulation = survivors
        #self.printPopulationInputs()

    def termination(self):
        if self.numberOfGenerations == 50:
            return True
        self.numberOfGenerations += 1
        print(self.numberOfGenerations)
        return False

    def evolveUntilTermination(self):
        # run this function until termination condition is met
        while not self.termination():
            chromosomesWithEvals = self.populationEvaluation()
            breeders = self.populationSelection(chromosomesWithEvals)
            parents, children = self.populationVariation(breeders)
            self.populationUpdate(parents, children)
        self.printPopulationInputs()
        self.printPopulationInputAverages()

    ############# Helper methods ##################

    def printPopulationInputs(self):
        print("\n\nPopulation Inputs\n\n")
        for chromosome in self.currentPopulation:
            functionInputs = self.getInputsFromBitStr(chromosome)
            print(functionInputs)

    def printPopulationInputAverages(self):
        print("\n\nInput Averages: \n\n")
        x = 0
        y = 0
        for chromosome in self.currentPopulation:
            functionInputs = self.getInputsFromBitStr(chromosome)
            x += functionInputs[0]
            y += functionInputs[1]
        print("X: {} Y: {}".format(float(x) / float(len(self.currentPopulation)), float(y) / float(len(self.currentPopulation))))
        print("Min: {}".format(self.evalFunction([x, y])))

    def tourneySelect(self, chromosomesWithEvals):
        selectionIndices = []
        for i in range(self.tourneyKval):
            selectionIndices.append(randint(0, len(chromosomesWithEvals) - 1))
        selectionIndices = sorted(selectionIndices)[::-1]
        # randomly select 'k' individuals from the parent population
        tourneyParticipants = []
        for k in range(len(selectionIndices)):
            tourneyParticipants.append(chromosomesWithEvals[selectionIndices[k]])
        # choose the best out of our tourney - we can improve this later rev by allowing 2nd best with prob p *(1-p)^2 and 3rd best p * (1-p)^3 etc
        # since chromosomes and particpants are reverse sorted, the min is just the 0th index element
        tourneyWinner = tourneyParticipants[0]
        return tourneyWinner 

    def crossover(self, parent1, parent2):
        randomPoint = randint(0, len(parent1)-1)
        # from random point to end of string perform the single point cross over
        for i in range(randomPoint, len(parent1)):
            tempVal = parent1[i]
            parent1[i] = parent2[i]
            parent2[i] = tempVal
        return parent1, parent2 # these are the children now

    def mutate(self, child):
        randomPoint = randint(0, len(child)-1)
        # flip a random bit
        if child[randomPoint] == "0":
            child[randomPoint] = "1"
        elif child[randomPoint] == "1":
            child[randomPoint] = "0"
        else:
            print("Your bit string is messed up")
            sys.exit(-1)
        return child # this is now a mutated child

    def getInputsFromBitStr(self, chromosome):
        inputValues = []
        bitsPerValue = len(self.currentPopulation[0]) / self.evalNumberOfInputs
        for i in range(self.evalNumberOfInputs):
            spliceStart = i * bitsPerValue
            spliceEnd = (i+1) * bitsPerValue
            inputBits = chromosome[spliceStart:spliceEnd]
            decimalValue = self.binaryStringToDecimal(inputBits)

            # using our scalar and shift - convert these values to proper inputs
            functionInputVal = self.decodeDecimalToInputWithinRange(decimalValue)
            inputValues.append(functionInputVal)
        return inputValues

    def binaryStringToDecimal(self, binStr):
        reverseStr = binStr[::-1]
        intVal = 0
        for i in range(len(reverseStr)):
            if reverseStr[i] == "1":
                intVal += 2**i
        return intVal

    def decodeDecimalToInputWithinRange(self, decimalValue):
        unscaled = float(decimalValue) / float(self.scalar)
        unscaledAndUnshifted = unscaled - self.shift
        return unscaledAndUnshifted

    def getShiftValue(self):
        if self.minRange < 0:
            shiftVal = abs(self.minRange)
        return shiftVal

    def getScalar(self):
        bitsPerValue = len(self.currentPopulation[0]) / self.evalNumberOfInputs
        maxBitStr = ""
        for i in range(bitsPerValue):
            maxBitStr += "1"
        maxDecimalVal = self.binaryStringToDecimal(maxBitStr)

        maxRangeVal = self.maxRange + self.shift
        scalarVal = float(maxDecimalVal) / float(maxRangeVal)
        return scalarVal

############# Evaluation Functions #############

def goldSteinPriceFunction(inputValues):
    """
    @param inputValues: all functions that we want to pass in must contain a list of inputValues
    """
    x = inputValues[0]
    y = inputValues[1]
    return ((x+y+1)**2*(19-14*x+3*x**2-14*y+6*x*y+3*y**2)+1)*((2*x-3*y)**2*(18-32*x+12*x**2+48*y-36*x*y+27*y**2)+30)

def rosenbrockFunction(inputValues):
    x = inputValues[0]
    y = inputValues[1]
    return (1-x)**2 + 100*(y-x**2)**2

def main(argv):
    if len(argv) == 7:
        initPopulationSize = int(argv[1])
        numberOfChromosomeBits = int(argv[2])
        rangeMin = int(argv[3])
        rangeMax = int(argv[4])
        numberOfInputs = int(argv[5])
        functionOfChoice = argv[6]
        if numberOfChromosomeBits % numberOfInputs != 0:
            print("You need to be able to evenly divide the number of chromosome bits, and the number of evaluation function inputs values")
            sys.exit(-1)

        # set up our genetic algorithm with the function we wish.
        if functionOfChoice == "gold":
            ga = GeneticAlgorithm(initPopulationSize, numberOfChromosomeBits, rangeMin, rangeMax, goldSteinPriceFunction, numberOfInputs)
        elif functionOfChoice == "rose":
            ga = GeneticAlgorithm(initPopulationSize, numberOfChromosomeBits, rangeMin, rangeMax, rosenbrockFunction, numberOfInputs)
        else:
            print("invalid function choice")
        # perform evolution until termination
        ga.evolveUntilTermination()

    else:
        print("Invalid number of arguments dude!")




if __name__ == "__main__":
    main(sys.argv)




