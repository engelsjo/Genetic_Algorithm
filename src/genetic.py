import os
import sys

class GeneticAlgorithm(object):
    def __init__(self):
        self.populationInitialization()

    def populationInitialization(self):
        x = "placeholder"

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
def main(argv):
    ga = GeneticAlgorithm()

if __name__ == "__main__":
    main(sys.argv)