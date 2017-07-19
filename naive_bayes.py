# -*- coding: utf-8 -*-
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#from sklearn.metrics import confusion_matrix

from math import sqrt
from math import pi
from math import exp


def getPriors(labels):
    """
    Get the class priors by calculating the class probability from the 
    provided set. The prior is computed as 
    
    (prior for class A) = (number of class A samples) / (total number of samples)

    Parameters
    ----------
    labels : target class values    
        
    Returns
    -------
    priors : A dictionary with the class priors. 
             E.g. { ClassA: prior, ClassB: prior, ...}    
    """
    priors = {}
    for className in labels:
        N = labels.size
        class_occurrence = (labels == className).sum()
        priors[className] = class_occurrence/N
    print "the priors are ",priors
    return priors

       
def fit(features, labels):
    """
    Fits coefficients for a Gaussian Naive Bayes. This method computes and
    returns the in-class mean and stadnard deviation for each feature in 
    the training vectors.
    
    Parameters
    ----------
    featires : training vectors     
    labels   : target class values    
        
    Returns
    -------
    priors : A dictionary with with the in-class mean/std for each attribute

    {ClassA: [(attribute1_mean, attribute1_std]), 
              (attribute2_mean, attribute2_std],...)
     ClassB: [(attribute1_mean, attribute1_std]), 
              (attribute2_mean, attribute2_std],...
     ...}
    """
    # Get the unique classes from the sample
    uniqueClasses = np.unique(labels)
    coeffs = {}
    # Loop over the unique classes to compute the mean/std statistics
    for className in uniqueClasses:
      featuresInClass = features[labels == className]
      # Compute the mean/std for each input feature
      statsInClass = [(np.mean(feature), np.std(feature)) for feature in zip(*featuresInClass)]            
      coeffs[className] = statsInClass    
    print "The cofficents are ",coeffs
    return coeffs    
                
def getLikelihood(x, featureIndex, model, className):
    """
    Computes the likelihood (i.e. the probability of the evidence given the
    model parameters) for a single value/class combination. The likelihood
    is computed using a Gaussian probability desnity function
    
    f(x|mu, sigma) = 
        1 / sqrt( 2 * pi * sigma^2 ) * exp ( - ( x-mu )^2 / (2 * sigma^2) )
    
    Parameters
    ----------
    x             : observation value
    featureIndex  : position of this attribute in the input vector. If the 
                    model was fitted against an N-dimenisonal input vector
                    [x_0, x_1, ..., x_N], featureIndex should point to the
                    position of x in the original vector (e.g. 0,1,...,N)
    model         : a dictionary with with the in-class mean/std for each 
                    attribute. See the fit(features, labels) method
    className     : class to asses the observation against
                
        
    Returns
    -------
    f : the (x|className) likelihood based on the Guassian PDF
    """
    classStats = model[className]
    mean = classStats[featureIndex][0]
    std  = classStats[featureIndex][1]
    f = (1/(sqrt(2*pi*pow(std,2))) * exp(-pow((x-mean),2)/(2*pow(std,2))))
    #print "The gaussian value(likelyhood)",f
    return f
    
def getPosterior(x, model, priors):
    """
    Computes the posterior using a Gaussian Naive Bayes.
    
    P(class|x = [x_1, x_2, ..., x_N]) = likelihood(x|class) * prior(class)
    
    We use the naive assumption of conditional independence between the features,
    which means that
    
    P([x_1, x_2, ..., x_N]|class) = P(x_1|class) * P(x_2|class) * ... * P(x_N|class)
    
    Parameters
    ----------
    x             : input vector
    model         : a dictionary with with the in-class mean/std for each 
                    attribute. See the fit(features, labels) method
    priors        : a dictionary with with the in-class mean/std for each attribute
                
        
    Returns
    -------
    p : the posterior for all classes in priors given the input vector
    """
    posteriors = {}
    # Loop over all observed classes
    for className in priors:
        # Compute p(x_1|class) * p(x_2|class) * ... * p(x_N|class) using the
        # likelihood function, then multiply by the prior to get
        # p(class|x = [x_1, x_2, ..., x_N])
        p = 1
        for featureIndex in range(x.size):
            p = p * (getLikelihood (x[featureIndex], featureIndex, model, className) * priors[className])
        posteriors[className] = p
    #print "The posteriors are ",posteriors
    return posteriors

def classify(x, model, priors):
    """
    This method uses Maximum a posteriori estimation (MAP) to make a class
    prediction on an unseen observation. 
    
    Class_MAP = argmax_c posterior(c|x) = argmax_c likelihood(x|c) * prior (c)
    
    Parameters
    ----------
    x             : input vector
    model         : a dictionary with with the in-class mean/std for each 
                    attribute. See the fit(features, labels) method
    priors        : a dictionary with with the in-class mean/std for each attribute
                
        
    Returns
    -------
    The name of the class that maximizes the posterior value
    """    
    posteriors = getPosterior(x, model, priors)
    #print "The max_arg of posteriors",max(posteriors, key=lambda key: posteriors[key])
    return max(posteriors, key=lambda key: posteriors[key])


# Load the data set
# We use Auto MPG from UCI Machine Learning Repository
# https://archive.ics.uci.edu/ml/datasets/Auto+MPGs

car_data = np.genfromtxt("C:/Users/Miracle/Desktop/auto_data.csv", usecols=(4, 3, 7))
car_data = car_data[~np.isnan(car_data).any(axis=1)]
features = car_data[:,[0,1]]
labels = car_data[:,2]

# Split the data into test/train subsets
features_train, features_test, labels_train, labels_test = train_test_split(features,
                                                                            labels, test_size=0.1,
                                                                            random_state = 100)
# Fit the model
priors = getPriors(labels_train)
model = fit(features_train, labels_train)

# Make predictions on the test data
predictions = [classify(x, model, priors) for x in features_test]

# Measure accuracy
print("Prediction accuracy: %.2f\n" % accuracy_score(labels_test, predictions))


