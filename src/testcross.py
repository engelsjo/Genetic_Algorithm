def crossover(parent1, parent2):
    print(parent1)
    print(parent2)
    randomPoints = []
    bitsPerValue = 4
    for a in range(2):
        #randomPoint = randint(0, bitsPerValue-1)
        randomPoint = 2
        randomPoints.append(randomPoint)
    # from random point to end of string perform the single point cross over
    for i, randomPoint in enumerate(randomPoints):
        start = randomPoint + (i*bitsPerValue)
        end = (i+1)*bitsPerValue
        print("{} : {}".format(start, end))
        for j in range(start, end):
            #print("flipping bits")
            tempVal = parent1[j]
            parent1[j] = parent2[j]
            parent2[j] = tempVal
    return parent1, parent2 # these are the children now


x = ['1', '0', '1', '1', '1', '0', '1', '0']
y = ['0', '1', '1', '0', '0', '1', '0', '1']
child1, child2 = crossover(x, y)

print(child1)
print(child2)
