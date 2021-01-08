import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns; sns.set()
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import MeanShift, estimate_bandwidth
from scipy.sparse.linalg import svds

def dataPreprocessing(filename):
    """
    Encoding of raw data
    :return:
    """
    website = []
    userInteraction = {}
    with open(filename, 'r') as file:
        user = 0
        interaction = []
        for line in file:
            lineContent = line.split(',')
            if lineContent[0] == 'A':
                website.append(int(lineContent[1]))
            if lineContent[0] == 'C':
                if len(interaction) > 0:
                    userInteraction[user] = interaction
                    interaction = []
                user = int(lineContent[2])
            if lineContent[0] == 'V':
                interaction.append(int(lineContent[1]))
        userInteraction[user] = interaction

    sortedWebsite = np.sort(np.asarray(website))
    userWeb = {}
    if filename.startswith('train')==True:
        for user, item in userInteraction.items():
            temp = np.zeros(len(sortedWebsite))
            for website in item:
                index = np.where(sortedWebsite == website)
                temp[index] = 1
            userWeb[user] = temp.tolist()
        return userWeb, userInteraction
    else:
        testLabel={}
        for user, item in userInteraction.items():
            temp = np.zeros(len(sortedWebsite))
            if len(item)==1:
                testLabel[user]=item[0]
                index = np.where(sortedWebsite == item[0])
                temp[index] = 1
            else:
                testLabel[user] = item[len(item)-1]
                for website in item[:len(item)-1]:
                    index = np.where(sortedWebsite == website)
                    temp[index] = 1
            userWeb[user] = temp.tolist()
        return userWeb,testLabel



def svdPrediction(X_test, testUser,topN):
    U, sigma, Vt = svds(X_test, k=50)
    sigma = np.diag(sigma)
    user_pred = np.dot(np.dot(U, sigma), Vt)
    indx = np.argsort(user_pred, axis=1)

    truePredict = 0
    for i in range(10001,15001):
        for web in indx[i-10001][len(X_test[0])-int(topN):]:
            if int (web)==(int(testUser[i]-1000)):
                truePredict+=1

    testAccuracy = truePredict / 5000

    print('Test accuracy using SVD matrix factorization method=')
    print(testAccuracy)


def userBasedNeighboring(X_train,userInteract,X_test, testUser,topN):
    #Training
    cosine=cosine_similarity(X_train)
    indx=np.argsort(cosine,axis=1)
    similarTen={}
    for user,simUser in zip(userInteract.keys(),indx):
        similarTen[user]=list(simUser[len(X_train)-int(topN)+1:len(X_train)-1])
    predictedWeb={}
    for simUs,simVal in similarTen.items():
        temp=[]
        for item in simVal:
            item=(10001+int(item))
            temp.append(userInteract[item])
        flat_list = [item for sublist in temp for item in sublist]

        count=dict(Counter(flat_list))
        clusteList = sorted(count.items(), key=lambda kv: kv[1])
        finalList = []
        for web, count in clusteList[len(clusteList) - int(topN):]:
            finalList.append(web)
        predictedWeb[simUs]= finalList

    #Testing
    testCosine=cosine_similarity(X_test,X_train)
    indx = np.argsort(testCosine, axis=1)
    finalIndex=indx[:,-1:]
    truePredict=0
    i=10001
    for ind in finalIndex:
        for web in predictedWeb[int(ind)+10001]:
            if int (web)==int(testUser[i]):
                truePredict+=1
        i+=1

    testAccuracy = truePredict / 5000

    print('Test accuracy using user-based neighboring method=')
    print(testAccuracy)



def meanShiftClustering(X_train,userInteract,X_test, testUser,topN):
    """
    K-mean implementation
    :return:
    """
    #Training
    bandwidth = estimate_bandwidth(X_train, quantile=0.2, n_samples=500)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(X_train[:5000])
    msLabels = ms.labels_
    cluster_centers = ms.cluster_centers_
    predictDict={}
    indexOfclass = np.asarray(np.where(msLabels == 0))
    predictDict[0] = predictWebsites(userInteract,indexOfclass+10001,topN)
    indexOfclass = np.asarray(np.where(msLabels == 1))
    predictDict[1] = predictWebsites(userInteract, indexOfclass+10001,topN)
    indexOfclass = np.asarray(np.where(msLabels == 2))
    predictDict[2] = predictWebsites(userInteract, indexOfclass+10001,topN)


    # Testing
    testCosine = cosine_similarity(X_test, cluster_centers)
    indx = np.argsort(testCosine, axis=1)
    finalIndex = indx[:, -1:]
    truPredict=0

    for clustNum,target in zip(finalIndex,testUser.values()):
        for web in predictDict[int(clustNum)]:
                if int(web)==int(target):
                    truPredict+=1

    testAccuracy=truPredict/5000

    print('Test accuracy for Meanshift=')
    print(testAccuracy)


def kmeanClustering(X_train,userInteract,X_test, testUser,topN):
    """
    K-mean implementation
    :return:
    """
    #Training
    kmeans = KMeans(n_clusters=7, random_state=0).fit(X_train)
    kmeanLabels = kmeans.labels_
    predictDict={}
    indexOfclass = np.asarray(np.where(kmeanLabels == 0))
    predictDict[0] = predictWebsites(userInteract,indexOfclass+10001,topN)
    indexOfclass = np.asarray(np.where(kmeanLabels == 1))
    predictDict[1] = predictWebsites(userInteract, indexOfclass+10001,topN)
    indexOfclass = np.asarray(np.where(kmeanLabels == 2))
    predictDict[2] = predictWebsites(userInteract, indexOfclass+10001,topN)
    indexOfclass = np.asarray(np.where(kmeanLabels == 3))
    predictDict[3] = predictWebsites(userInteract, indexOfclass+10001,topN)
    indexOfclass = np.asarray(np.where(kmeanLabels == 4))
    predictDict[4] = predictWebsites(userInteract, indexOfclass+10001,topN)
    indexOfclass = np.asarray(np.where(kmeanLabels == 5))
    predictDict[5] = predictWebsites(userInteract, indexOfclass+10001,topN)
    indexOfclass = np.asarray(np.where(kmeanLabels == 6))
    predictDict[6]= predictWebsites(userInteract, indexOfclass+10001,topN)
    centroids=kmeans.cluster_centers_

    # Testing
    testCosine = cosine_similarity(X_test, centroids)
    indx = np.argsort(testCosine, axis=1)
    finalIndex = indx[:, -1:]
    truPredict=0

    for clustNum,target in zip(finalIndex,testUser.values()):
        for web in predictDict[int(clustNum)]:
                if int(web)==int(target):
                    truPredict+=1

    testAccuracy=truPredict/5000

    print('Test accuracy for K-means=')
    print(testAccuracy)

    elbowPlot(kmeans,X_train)


def elbowPlot(model,data):
    visualizer = KElbowVisualizer(model, k=(1, 12))
    visualizer.fit(data)
    visualizer.show()



def predictWebsites(userInteraction,clusterWeb,topN):
    """
    Prediction
    :return:
    """
    webList=[]

    for user in clusterWeb[0]:
        webList.append(userInteraction[user])
    flat_list = [item for sublist in webList for item in sublist]
    count = dict(Counter(flat_list))
    clusteList=sorted(count.items(), key=lambda kv: kv[1])
    finalList=[]
    for web,count in clusteList[len(clusteList)-int(topN):]:
        finalList.append(web)
    return (finalList)



if __name__ == '__main__':
    """
    Main function 
    
    """
    userWeb,userInteraction = dataPreprocessing('training.txt')
    users = np.array(list(userWeb.keys()))
    userValues = np.array(list(userWeb.values()))
    testuserWeb,testLabel=dataPreprocessing('test.txt')
    testusers = np.array(list(testuserWeb.keys()))
    testuserValues = np.array(list(testuserWeb.values()))
    print("Enter topN value=")
    topN=input()

    #Mean shift clustering
    meanShiftClustering(userValues, userInteraction, testuserValues, testLabel,topN)

    #Matrix Factorization with SVD
    svdPrediction(testuserValues, testLabel,topN)

    # User-based neighboring
    userBasedNeighboring(userValues,userInteraction,testuserValues,testLabel,topN)

    # k-mean clustering
    kmeanClustering(userValues,userInteraction,testuserValues,testLabel,topN)

