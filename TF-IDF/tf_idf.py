
import pandas as pd


def computeTF(wordDict, bow):
    """
    计算词频
    :param wordDict:
    :param bow:
    :return:
    """
    tfDict = {}
    nbowCount = len(bow)
    
    for word, count in wordDict.items():
        tfDict[word] = count / nbowCount
    
    return tfDict


def computeIDF(wordDictList):
    """
    逆文档频率idf
    :param wordDictList:
    :return:
    """
    # 每个词作为key，初始值为0
    idfDict = dict.fromkeys(wordDictList[0], 0)
    N = len(wordDictList)
    
    import math
    
    for wordDict in wordDictList:
        # 遍历字典中的每个词汇
        for word, count in wordDict.items():
            if count > 0:
                idfDict[word] += 1
                
    for word, ni in idfDict.items():
        idfDict[word] = math.log10((N + 1)/(ni + 1))
    
    return idfDict


def coumputeTFIDF(tf, idfs):
    """
    计算TF-IDF
    :param tf:
    :param idfs:
    :return:
    """
    tfidf = {}
    for word, tfval in tf.items():
        tfidf[word] = tfval * idfs[word]
        
    return tfidf

    
if __name__ == '__main__':
    docA = "The cat sat on my bed"
    docB = "The dog sat on my knees"
    
    bowA = docA.split(' ')
    bowB = docB.split(' ')
    
    # 构建词库
    wordSet = set(bowA).union(set(bowB))
    
    # 词数统计
    wordDictA = dict.fromkeys(wordSet, 0)
    wordDictB = dict.fromkeys(wordSet, 0)
    
    for word in bowA:
        wordDictA[word] += 1
    
    for word in bowB:
        wordDictB[word] += 1
        
    df = pd.DataFrame([wordDictA, wordDictB])
    
    print(df)
    
    tfA = computeTF(wordDictA, bowA)
    tfB = computeTF(wordDictB, bowB)
    
    print(tfA)
    
    idfs = computeIDF([wordDictA, wordDictB])
    print(idfs)
    
    tfidfA = computeTF(tfA, idfs)
    tfidfB = computeTF(tfB, idfs)
    
    df = pd.DataFrame([tfidfA, tfidfB])
    
    print(df)