import os
import sys
from random import randint

class GeneticAlgorithm(object):
    def __init__(self, initialPopulationSize=100, numChromosomeBits=16):
        self.initPopulationSize = initialPopulationSize
        self.nbrOfChromosomeBits = numChromosomeBits
        self.currentPopulation = []
        self.populationInitialization()

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
        x = "placeholder"

    def populationSelection(self):
        x = "placeholder"

    def populationVariation(self):
        x = "placeholder"

    def populationUpdate(self):
        x = "placeholder"

    def termination(self):
        return False

    def evolveUntilTermination(self):
        while not self.termination():
            self.populationEvaluation()
            self.populationSelection()
            self.populationVariation()
            self.populationUpdate()

    ############# Helper methods #############


def main(argv):
    if len(argv) == 3:
        initPopulationSize = int(argv[1])
        numberOfChromosomeBits = int(argv[2])
        ga = GeneticAlgorithm(initPopulationSize, numberOfChromosomeBits)

if __name__ == "__main__":
    main(sys.argv)




