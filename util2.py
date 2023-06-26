import time
import os
import os.path as osp
import numpy as np
def createLocalLog():
    curTime = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    localPath = osp.join("log", curTime)
    if not os.path.isdir(localPath):
        os.makedirs(localPath)
    return localPath
def saveMatrixToLocalLog(mat, epoch, name, dirName):
    localPath = osp.join(dirName, name);
    if not os.path.isdir(localPath):
        os.makedirs(localPath)
    height = len(mat)
    width = len(mat[0])
    npMat = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            npMat[i][j] = mat[i][j]
    np.save(osp.join(localPath, "epoch" + str(epoch)), npMat)