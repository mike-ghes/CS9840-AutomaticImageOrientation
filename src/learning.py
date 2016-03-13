import csv
import numpy as np
import random
import itertools

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab

from subprocess import check_call

from sklearn import cross_validation
from sklearn import preprocessing
from sklearn import neighbors
from sklearn import mixture
from sklearn import ensemble
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.lda import LDA

data_2012 = "datasets\\_csv\\processed_VOC2012_LUV.csv"
data_2007 = "datasets\\_csv\\processed_VOC2007.csv"
data_indoor = "datasets\\_csv\\processed_IndoorScenes1.csv"
data_easy = "datasets\\_csv\\processed_easy_LUV.csv"

def load_csv(fname = data_indoor):
    """
    Load array from csv file
    :param fname: filename for csv
    :returns X,Y: Seperated feature vectors and rotation labels
    """
    
    reader = csv.reader(open(fname, 'r'))
    
    # Blank list
    data = []
    
    # Don't read the zeroth element of each row (image name), convert to float.
    for row in reader:
        data.append(map(float, row[1:]))
    
    # Convert list to array     
    d = np.array(data)
    
    # Seperate labels from features
    Y = d[:,0]
    X = d[:,1:]
       
    return X,Y

def extend_training_set(X,Y, N_cm = 10):
    """
    Experimented around with this. Expensive for little gain. The idea was to have 4 copies of each image in the training set.
    :params X: feature vector
    :params Y: labels
    :returns: newX, newY: returns new X,Y with 4 times the original length
    """
    
    #Copy arrays
    plus90 = X[:]
    plus180 = X[:]
    plus270 = X[:]
    
    # Loop through each row
    for i, x in enumerate(X):
        
        # Back to original shape
        original_shape = x.reshape((N_cm,N_cm,6))
        
        # Rotate the array
        once = np.rot90(original_shape)
        twice = np.rot90(once)
        thrice = np.rot90(twice)
        
        # Re-flatten
        plus90[i] = once.flatten()
        plus180[i] = twice.flatten()
        plus270[i] = thrice.flatten()
        
        
    # New labels are (old_label + num_rotations) (mod 4)
    newY = np.hstack(( Y,   (Y+1)%4,   (Y+2)%4,   (Y+3)%4))
    
    # Stack all feature vectors
    newX = np.vstack(( X,   plus90,    plus180,   plus270))
    
    return newX, newY
    
def remove_center(X, N=10, border=2):
    """
    Remove the center of the image as recommended by Cingovska et al. border ~ 3 provided an
    improvement to performance for N=10. This was a late addition and I didn't have time to
    repeat all my results in the report
    :params X: feature vectors
    :params N: number of blocks originally used
    :params border: number of blocks to be used along each edge
    :returns: new X with center blocks removed
    """
    
    #Translate X into original block shape
    X2 = X.reshape((len(X),N,N,6))

    # Decompose matrix as follows:
    
    # N N N N N N N N
    # N N N N N N N N
    # W W         E E
    # W W         E E
    # W W         E E
    # W W         E E
    # S S S S S S S S
    # S S S S S S S S
    
    # block is then flattened as well
    North = X2[:, :border].reshape(len(X), N*border*6)
    South = X2[:, -border:].reshape(len(X), N*border*6)
    West = X2[:, border:-border, :border].reshape(len(X), (N-2*border)*border*6)
    East = X2[:, border:-border, -border:].reshape(len(X), (N-2*border)*border*6)
    
    # Stack the vectors horizontally, e.g.
    # N_,1 .., N_x, W_1, .. W_x, E_1, .., E_x, S_1, .., S_x
    return np.hstack((North, West, East, South))
                        
def knn(X,Y):
    """
    Early code to generate plots for comparing k-NN with varying k with and without feature extraction.
    I don't think it made the report
    :params X: feature vectors
    :params Y: labels
    :returns: None
    """
    
    # Transform all X data by PCA. Note that PCA was fit on the testing set as well as training.
    pca = PCA(n_components=100)
    X_r = pca.fit(X).transform(X)
    
    # Transform all X data by LDA. Same problem as above.
    lda = LDA()
    X_r2 = lda.fit(X, Y).transform(X)
    
    # Vary k.
    for k in [1,2,4,8,16,32, 64, 128, 256, 512]:
    
        # Training set was fixed at first 2000 vectors. This was for a smaller dataset at the time
        
        # No feature extraction
        knn = neighbors.KNeighborsClassifier(k)
        knn.fit(X[:2000], Y[:2000])
        
        # PCA
        knn2 = neighbors.KNeighborsClassifier(k)
        knn2.fit(X_r[:2000], Y[:2000])
        
        # LDA
        knn3 = neighbors.KNeighborsClassifier(k)
        knn3.fit(X_r2[:2000], Y[:2000])
        
        #Prediction results. Rather ugly way to code this looking back.
        predict = []
        predict2 = []
        predict3 = []
        for i in range(2000, len(X)):
            predict += [ knn.predict(X[i])  == Y[i] ]
            predict2 += [ knn2.predict(X_r[i])  == Y[i] ]
            predict3 += [ knn3.predict(X_r2[i])  == Y[i] ]
            
        
        # Plot accuracy. R= no feature extraction, G= PCA, B= LDA    
        pylab.scatter(k, float(sum(predict))/len(predict), c='r')
        pylab.scatter(k, float(sum(predict2))/len(predict2), c='g')
        pylab.scatter(k, float(sum(predict3))/len(predict3), c='b')
        
def compare_borders(X,Y, k=50):
    """
    Compare the results of removing borders (using kNN)
    :params X: feature vectors
    :params Y: labels
    :params k: number of neighbors
    :returns: None
    """
    
    # Use sklearn's train/test split
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y)
    
    #Remove the center (note border=5 with N=10 will remove nothing. Anything larger will end up with redundant labels)
    knn1 = neighbors.KNeighborsClassifier(k).fit(remove_center(X_train, border=1), Y_train)
    knn2 = neighbors.KNeighborsClassifier(k).fit(remove_center(X_train, border=2), Y_train)
    knn3 = neighbors.KNeighborsClassifier(k).fit(remove_center(X_train, border=3), Y_train)
    knn4 = neighbors.KNeighborsClassifier(k).fit(remove_center(X_train, border=4), Y_train)
    knn5 = neighbors.KNeighborsClassifier(k).fit(X_train, Y_train)
    
    # Arrays of predictions == actual
    p1 = knn1.predict(remove_center(X_test, border=1)) == Y_test
    p2 = knn2.predict(remove_center(X_test, border=2)) == Y_test
    p3 = knn3.predict(remove_center(X_test, border=3)) == Y_test
    p4 = knn4.predict(remove_center(X_test, border=4)) == Y_test
    p5 = knn5.predict(X_test) == Y_test
    
    # Accuracy function
    acc = lambda X: 1.*sum(X)/len(X)
    
    # Print results
    print "border =1", acc(p1)
    print "border =2", acc(p2)
    print "border =3", acc(p3)
    print "border =4", acc(p4)
    print "no border", acc(p5)
    
    
                                         
def optimize_pca(X,Y):
    """
    Figure 1 in the report. Plots variance after PCA. 95% variance line also plotted.
    :params X: feature vectors
    :params Y: labels
    :returns: None
    """
    # {0, 10, 20, ..., 590}    
    for n in range(0,599,10):
        
        #Fit PCA
        pca = PCA(n_components=n).fit(X)
        # Plot variance
        pylab.scatter(n, sum(pca.explained_variance_ratio_))
    
    #Place 95% line.
    pylab.axhline(y=0.95, color='r')
          
                        
def lda_scatter(X,Y, dim3=True):
    """
    Scatter data after LDA. Used in Figure 3.
    Blue = 0
    Cyan = 90
    Yellow = 180
    Red = 270
    :params X: feature vectors
    :params Y: labels
    :params dim3: 3-D: True(default) or 2-D: False
    
    """
    # Fit data
    lda = LDA()
    lda.fit(X, Y)
    X_r2 = lda.transform(X)    

    # 3-D plot
    if dim3:
        fig = pylab.figure()
        ax = Axes3D(fig)
        ax.scatter3D(X_r2[:,0],X_r2[:,1],X_r2[:,2], c=Y)
        
    #2-D plot
    else:
        plt.scatter(X_r2[:,0], X_r2[:,1], c= Y )

    
    
def fig5(X_r, Y, TRAIN_SIZE=6000):
    """
    Comparing k-NN with PCA/LDA/None, Normalized/Unnormalized, Voting/Distance,  and k={2,4,8,16,32}
    Despite it's name. Code used for generating Figure 4. (Figure 5 in Vailaya et al.)
    :params X_r: feature vectors (un-normalized)
    :params Y: labels
    :params TRAIN_SIZE: Training set size.  
    
    """
    
    # Normalize X_r
    X_n = preprocessing.normalize(X_r)
    
    #kNN weighting and k
    weights = [ "uniform", "distance" ]
    ks = [2,4,8,16,32,64] 
    
    # Little lambda functions to standardize feature extraction
    pca = lambda X,Y: PCA(n_components=128).fit(X).transform(X)
    lda = lambda X,Y: LDA().fit(X, Y).transform(X)
    idn = lambda X,Y: X
    
    # Start the plot
    fig, ax = plt.subplots()
    plt.ylabel("Error %")
    plt.xlabel("k")
    
    
    # Try every combination (product) of weights, feature extraction and normalization
    for weight,   feat_reduce,   X_ in itertools.product(
        weights,   [pca, lda, idn],   [X_r, X_n]):
            
            # Reset error rate
            errors = []
            
            #Flags to make things easier
            reduction = "PCA" if feat_reduce == pca else "LDA"
            normalized = "n" if X_ is X_n else "r"
            
            #Initialize a black (i.e. key - cmy_K_) line
            linestyle = "k"
            
            # Match the point style used in Vailaya
            if weight == "uniform":
                if X_ is X_n:
                    linestyle += "x"
                else:
                    linestyle += "*"
            if weight == "distance":
                if X_ is X_n:
                    linestyle += "o"
                else:
                    linestyle += "+"
                    
            # As well as the line style
            if feat_reduce is pca:
                linestyle += ":"    # Dotted
            elif feat_reduce is lda:
                linestyle += "--"   # Solid
            else:
                linestyle += "-"    # Dashed
            
            # Loop through all k's        
            for k in ks:
                #Initialized classifier parameters
                knn = neighbors.KNeighborsClassifier(warn_on_equidistant=False)
                knn.n_neighbors = k
                knn.weights = weight
                
                #Here's where the lambda's come in handy.
                X = feat_reduce(X_,Y)
                
                # Fit the training set
                knn.fit(X[:TRAIN_SIZE], Y[:TRAIN_SIZE])
                
                # Again ugly code for the predictions
                predictions = []
                for i in range(TRAIN_SIZE, len(X)):
                    predictions += [ knn.predict(X[i])[0] ] 
                
                # Calculate error rate and append it to error rate list
                error = 1.- float(sum(predictions == Y[TRAIN_SIZE:])) / len(predictions)
                errors += [error]
                
                # Print it just for fun. Also in case error rates need to be exported.
                print weight, reduction, normalized, k, error
             
            # Plot the line for all k values   
            ax.plot(ks, errors, linestyle)
    
    # Couldn't specify legends properly
    #ax.legend()
    
       
def svm_comparison(X,Y):
    """
    Compares SVM kernels
    :params X: feature vectors
    :params Y: labels
    :returns: [ [clf, accuracy, [testing_predictions], [training_predictions], ... ]
    """
    
    # kernels to compare
    clfs = [
            svm.SVC(kernel="linear"),
            
            svm.SVC(kernel="poly", degree=2),
            svm.SVC(kernel="poly", degree=3),
            svm.SVC(kernel="poly", degree=4),
            svm.SVC(kernel="poly", degree=5),
            svm.SVC(kernel="poly", degree=6),
            
            svm.SVC(kernel="rbf", gamma=1.0),
            svm.SVC(kernel="rbf", gamma=0.5),
            svm.SVC(kernel="rbf", gamma=0.25),
            svm.SVC(kernel="rbf", gamma=0.125),
            svm.SVC(kernel="rbf", gamma=0),
            
            svm.SVC(kernel="sigmoid", gamma=1.0),
            svm.SVC(kernel="sigmoid", gamma=0.5),
            svm.SVC(kernel="sigmoid", gamma=0.25),
            svm.SVC(kernel="sigmoid", gamma=0.125),
            svm.SVC(kernel="sigmoid", gamma=0.0)
            ]
    
    # Use a common split
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y)
    
    # and LDA features (only fit on the train data
    lda = LDA().fit(X_train, Y_train)
    X_train_lda = lda.transform(X_train)
    X_test_lda = lda.transform(X_test)
    
    #Eventual output
    results = []
    
    # Loop through classifiers
    for clf in clfs:
        
        # Fit the model
        model = clf.fit(X_train_lda, Y_train)
        
        # Get accuracy and predictions for train and test.
        testresults = model.predict(X_test_lda) == Y_test
        trainresults = model.predict(X_train_lda) == Y_train
        accuracy = 1.*sum(testresults)/len(testresults)
        
        # Append to results
        results += [ [ clf, accuracy, testresults, trainresults ] ]
        
    return results
            
def mixture_of_gauss(X,Y):
    """
    Mixture of Gaussians was unsupervised so required a bit more work to get predictions out of.
    :params X: feature vectors
    :params Y: labels
    """
    
    # Split training/testing
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y)
    
    # Fit and transform with LDA
    lda = LDA().fit(X_train, Y_train)
    X_train = lda.transform(X_train)
    X_test = lda.transform(X_test)
    
    # Initialize GMM
    clf = mixture.GMM(n_components=4)
    
    # "Fit" to Y. Specify the component means for each cluster. Component labels are not necesarily the same as Y however.
    clf.means_ = np.array([X_train[Y_train == i].mean(axis=0) for i in range(4)])
    # Fit X
    clf.fit(X_train)
    
    # Break up X into 4 based on the Y label
    x_0t   = [ x for i,x in enumerate(X_train) if Y_train[i] == 0]
    x_90t  = [ x for i,x in enumerate(X_train) if Y_train[i] == 1]
    x_180t = [ x for i,x in enumerate(X_train) if Y_train[i] == 2]
    x_270t = [ x for i,x in enumerate(X_train) if Y_train[i] == 3]
    
    # Matrix of known Y vs. prediction on the train set.
    mat = [ [ sum(clf.predict(x)==i) for i in [0,1,2,3] ] for x in [x_0t, x_90t, x_180t, x_270t] ]

    # Pick the max of each row. If clusters are good then there will be no collisions
    map0 = mat[0].index(max(mat[0]))
    map1 = mat[1].index(max(mat[1]))
    map2 = mat[2].index(max(mat[2]))
    map3 = mat[3].index(max(mat[3]))
    
    #Heavy handed way to make sure that mapping is collision free. If this assertion is false, try again, you probably just got unlucky. 
    num_unique = len(set([map0, map1, map2, map3]))
    assert num_unique == 4,  str(map0) + str(map1) + str(map2) + str(map3) + str(mat)
    
    # Transforms clf cluster prediction to expected Y label.
    def map_predict(X):
        # Make a dictionary
        d = { map0:0, map1:1, map2:2, map3:3 }
        
        # For each prediction, consult dictionary.
        return map(lambda z: d[z], clf.predict(X))
    
    
    # Use our mapped predictions instead of clf.predict
    test = map_predict(X_test) == Y_test
    train = map_predict(X_train) == Y_train
    
    # Little accuracy function. Should have done this sooner.
    accuracy = lambda X: 1.*sum(X)/len(X)
    
    # Print training and testing accuracy
    print "train:", accuracy(train), "test:", accuracy(test)
    
    # Return everything needed to run on a new testing set.
    return test, train, clf, lda, map_predict
                        
                                                
def table(X,Y):
    """
    Used extensively for comparing classifiers. I changed clfs around a lot depending on what I was comparing
    :params X: feature vectors
    :params Y: labels
    :returns: None
    """
    
    # Split training/testing
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y)
    
    # Fit and transform with LDA
    lda = LDA().fit(X_train, Y_train)
    X_train_lda = lda.transform(X_train)
    X_test_lda = lda.transform(X_test)
    
    # Change this list depending on which classifiers you want to compare.
    clfs = [
            (neighbors.KNeighborsClassifier(n_neighbors=50, warn_on_equidistant=False), "k-NN50"),
            (svm.SVC(kernel="linear"), "SVM"),
            #(ensemble.AdaBoostClassifier(n_estimators=5), "AdaBoost - 5"),
            #(ensemble.AdaBoostClassifier(n_estimators=10), "AdaBoost - 10"),
            #(ensemble.AdaBoostClassifier(n_estimators=25), "AdaBoost - 25"),
            (ensemble.AdaBoostClassifier(n_estimators=50), "AdaBoost - 50"),
            #(ensemble.AdaBoostClassifier(n_estimators=100), "AdaBoost - 100"),
            #(ensemble.AdaBoostClassifier(n_estimators=500), "AdaBoost - 500"),
            ]
    
    # First element is classifier, second is name
    for clf, name in clfs: 
        
        # With LDA then without
        for isLDA in [True, False]:
            
            #Switch training and testing X depending
            test = X_test_lda if isLDA else X_test
            train = X_train_lda if isLDA else X_train
           
            # Fit the classifier
            model = clf.fit(train, Y_train)  

            # Get predictions
            train_result = model.prediction(train) == Y_train
            test_result = model.predictions(test) == Y_test
            
            # Print results
            print name, sum(train_result)*1./len(train_result), sum(test_result)*1./len(test_result)
 
    
def toLVQFile(X,Y, trainSz = None, fpath = "datasets\\lvq"):
    """
    Create a dataset that LVQ_PAK can understand.
    :params X: feature vectors
    :params Y: labels
    :params trainSz: specify training Size. If None uses half
    :params fpath: Path name for the lvq dataset files
    :returns: None
    """
    
    # Python handles default parameters at compile time. Awkward solution.
    if trainSz == None:
        trainSz = len(Y)/2
    
    # Fit LDA
    # ... to entire data. oops!
    lda = LDA().fit(X,Y)


    # Intermediate solution to training/testing split. Before I found sklearn.preprocessing
    
    # Shuffle 0...len(dataset)
    shuffled_indices = range(len(Y))
    random.shuffle(shuffled_indices)
    
    #Grab the first n elements for training and set.
    trainset = shuffled_indices[:trainSz]
    testset  = shuffled_indices[trainSz:]

    # Open training set file
    with open(fpath + "_train.txt","w") as outfile:
        
        # First line is number features (3 after LDA)
        outfile.write("3\n")
        
        # Write each vector to a line
        for i in trainset:
            outfile.write("\t".join(map(str,lda.transform(X[i])[0])) + "\t" + str(int(Y[i])*90) +"\n")
    
    # Open testing set file and do the same as above
    with open(fpath + "_test.txt","w") as outfile:
        outfile.write("3\n")
        for i in testset:
            outfile.write("\t".join(map(str,lda.transform(X[i])[0])) + "\t" + str(int(Y[i])*90) +"\n")
                      

def train_lvq(numcbv, run_len_mult = 40, fpath = "datasets\\lvq"):
    """
    Train an LVQ calling external functions.
    
    Creates files in the //lvq folder that need to be deleted by hand. Mine grew to about 40 MB by the end
    of this project without ever clearing the folder.
    
    :params numcbv: number of codebook vectors
    :params run_len_mult: Kohonen recommends at least 40*numcbv iterations. That ratio can be changed here.
    :params fpath: file path for temp files.
    :returns: None
    """
    # Number of iterations recommended by Kohonen is 40 times the number of codebook vectors
    runlen = run_len_mult * numcbv
    
    #run length for 'sammon'. Doesn't affect learning. May not be necessary.
    #runlen2 = 100
    
    #codebook size 40 will create files "lvq/c40e.cod", "lvq/c40o.sam" etc.
    cb = "lvq\\c" + str(numcbv)
    train = fpath + "_train.txt"
    test = fpath + "_test.txt"

    # Little lambdas just to help with readability below.
    cmd = lambda X: "binaries_windows\\"+X+".exe"
    din = lambda X: " -din " + str(X)
    cout = lambda X: " -cout " + str(X)   
    cin = lambda X: " -cin " + str(X)
    rlen = lambda X: " -rlen " + str(X)
    noc = lambda X: " -noc " + str(X)
    cfout = lambda X: " -cfout " + str(X)    
  
    # Initialize LVQ with even codebooks per class
    check_call(cmd("eveninit") + din(train)                     + cout(cb + "e.cod") + noc(numcbv) )
    
    # Balance codebooks. Optional.
    check_call(cmd("balance")  + din(train) + cin(cb + "e.cod") + cout(cb + "b.cod") )
    
    #Codebook Training
    check_call(cmd("olvq1")    + din(train) + cin(cb + "b.cod") + cout(cb + "o.cod") + rlen(runlen) )
    
    # Compute accuracy for training and testing set.
    check_call(cmd("accuracy") + din(train) + cin(cb + "o.cod") + cfout(cb + "_train.cfo") )
    check_call(cmd("accuracy") + din(test)  + cin(cb + "o.cod") + cfout(cb + "_test.cfo") )
        
    #Optional. Slow.
    #call(cmd("sammon")                + cin(cb + "o.cod") + cout(cb + "o.sam") + rlen(runlen2) )
    
def check_accuracy(numcbv, train=False):
    """
    Check an accuracy from a codebook generated with train_LVQ
    :params numcbv: number of codebook vectors
    :params train: if True get testing accuracy else get training accuracy
    :returns: accuracy between 0 and 1
    """
    
    # Default location
    cb = "lvq\\c" + str(numcbv) + ("_train" if train else "_test")
    
    
    accuracy = []
    with open(cb+".cfo", 'r') as accuracyfile:
        for line in accuracyfile:
            accuracy += [int(line)]
    
    #Depending on what you need:
    #return accuracy
    return sum(accuracy)*1./len(accuracy)
    
def try_codebooksizes(rang = range(4, 400, 4)):
    """
    For trying codebook sizes. Plots results
    :params rang: List of codebook lengths to try.
    :returns: training and testing results
    """
    train_results = []
    test_results = []
    
    #Copy rang since we will remove from it in the except block
    for codebooksize in rang[:]:
        
        # Sometimes it fails for small codebook lengths. Weird. 
        try:
            train_lvq(codebooksize)
            train_results += [ check_accuracy(codebooksize, True) ]
            test_results += [ check_accuracy(codebooksize, False) ]
        except:
            print "skipped " + str(codebooksize)
            rang.remove(codebooksize)
            
    # Plot results
    plt.plot(rang, train_results, c='r')
    plt.plot(rang, test_results, c='b')

    # Return results    
    return train_results, test_results


def dataset_difficulty():
    """
    Determine relative dataset difficulty
    :returns [ [dataset_name, training accuracy, testing accuracy, trained classifier, trained lda] ...]
    """
    results = []
    datasets = [ data_2007, data_2012, data_indoor, data_easy ] 
    
    for data in datasets:
        
        #Let the user know where we are
        print data
        X,Y = load_csv(data)
        
        # Training/testing split + LDA fit
        X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y)
        lda = LDA()
        lda.fit(X_train, Y_train)
        
        # Use linear SVC
        clf = svm.SVC(kernel="linear")
        clf.fit(lda.transform(X_train), Y_train)
        
        # Predictions
        train_predict = clf.predict(lda.transform(X_train))
        test_predict = clf.predict(lda.transform(X_test))
        
        #Compute accuracy
        train_acc = 1.*sum(train_predict == Y_train)/len(train_predict)
        test_acc = 1.*sum(test_predict == Y_test)/len(test_predict)
        
        # Append results for that dataset
        results += [ [ data, train_acc, test_acc, clf, lda ] ]
         
    return results
    
    
    
                        
def extended_training(X,Y):
    """
    Early code trying out my extend training set idea. Kind of scrapped this idea.
    :params X: feature vectors
    :params Y: labels
    :returns predictions, lda and classifier
    """
    trainSz = 4000
    
    X_, Y_ = extend_training_set(X[:trainSz],Y[:trainSz])
    
    lda = LDA()
    lda.fit(X, Y)
    X2 = lda.transform(X)
    X_2 = lda.transform(X_)
    
    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_2, Y_)
    
    predict = clf.predict(X2)
    return predict, lda, clf