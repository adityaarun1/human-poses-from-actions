nJoints = 16
nActionClass = 20
accIdxs = [0, 1, 2, 3, 4, 5, 10, 11, 14, 15]
shuffleRef = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
edges = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], [10, 11], [11, 12],
         [12, 8], [8, 13], [13, 14], [14, 15], [6, 8], [8, 9]]

outputRes = 64
inputRes = 256

eps = 1e-6

momentum = 0.0
weightDecay = 0.0
alpha = 0.99
epsilon = 1e-8

scale = 0.25
rotate = 30
hmGauss = 1
#hmGaussInp = 20
#shiftPX = 50
#disturb = 10

rootDir = '/ssd_scratch/cvit/aditya/human-poses-from-actions/'
dataDir = rootDir + 'data'
expDir = rootDir + 'exp'
mpiiImgDir = '/ssd_scratch/cvit/aditya/mpii/images/'

nThreads = 8
