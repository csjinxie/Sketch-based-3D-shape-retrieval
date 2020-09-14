import numpy as np

def normData(inputFea, fea_mean, fea_std):
    outFea = (inputFea - fea_mean) / fea_std
    index = np.where(np.isnan(outFea))
    outFea[index] = 0       ### reset all nan as 0
    return outFea


