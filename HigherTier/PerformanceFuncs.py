import matplotlib.pyplot as plt
import numpy as np

import sklearn 
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, RocCurveDisplay, ConfusionMatrixDisplay

############################################################################################################################################
# Functions for all links predictions
############################################################################################################################################  

def calculate_accuracy(scores, true_labels):
    
    signal_scores = scores.reshape(-1)
    true_labels = true_labels.reshape(-1)
    
    true_edge_mask = (true_labels == 1)
    false_edge_mask = (true_labels == 0)
    
    true_edge_scores = signal_scores[true_edge_mask]
    false_edge_scores = signal_scores[false_edge_mask]
    
    thresholds = np.arange(0.05, 1.0, 0.05)
    predictions = []
    accuracy = []
    efficiency = []
    metric = []
    purity = []

    for threshold in thresholds:
        prediction = np.array(signal_scores >= threshold, dtype='float')
        true_edge_prediction_count = np.count_nonzero(true_edge_scores >= threshold)
        false_edge_prediction_count = np.count_nonzero(false_edge_scores >= threshold)
        
        n_selected = true_edge_prediction_count + false_edge_prediction_count
        
        this_efficiency = float(true_edge_prediction_count) / float(true_edge_scores.shape[0])
        this_purity = float(true_edge_prediction_count) / float(n_selected) if n_selected != 0 else 0
        
        accuracy.append(accuracy_score(true_labels, prediction))
        efficiency.append(this_efficiency)
        purity.append(this_purity)
        metric.append(this_efficiency * this_purity)

    accuracy = np.array(accuracy)
    efficiency = np.array(efficiency)
    purity = np.array(purity)
    metric = np.array(metric)
    
    plt.clf()
    plt.scatter(thresholds, accuracy, color='blue', label='accuracy', s=10)
    plt.plot(thresholds, accuracy, color='blue')
    plt.scatter(thresholds, efficiency, color='red', label='efficiency', s=10)
    plt.plot(thresholds, efficiency, color='red')
    plt.scatter(thresholds, purity, color='green', label='purity', s=10)
    plt.plot(thresholds, purity, color='green')
    plt.scatter(thresholds, metric, color='violet', label='efficiency*purity', s=10)
    plt.plot(thresholds, metric, color='violet')
    
    plt.xlabel('threshold')
    plt.ylabel('arbitary units')
    plt.legend()
    plt.show()
        
    max_accuracy = np.max(accuracy)
    max_accuracy_index = np.argmax(accuracy)
    optimal_threshold_accuracy = thresholds[max_accuracy_index].item()

    max_metric = np.max(metric)
    max_metric_index = np.argmax(metric)
    optimal_threshold_metric = thresholds[max_metric_index].item()    
    
    return optimal_threshold_accuracy, max_accuracy, optimal_threshold_metric, max_metric

############################################################################################################################################  

def plot_scores(scores, true_labels):
    
    signal_scores = scores.reshape(-1)
    true_labels = true_labels.reshape(-1)
    
    true_edge_mask = (true_labels == 1)
    false_edge_mask = (true_labels == 0)
    
    true_edge_scores = signal_scores[true_edge_mask]
    false_edge_scores = signal_scores[false_edge_mask]
    
    pos_plotting_weights = 1.0 / float(true_edge_scores.shape[0])
    pos_plotting_weights = np.ones(true_edge_scores.shape[0]) * pos_plotting_weights
    
    neg_plotting_weights = 1.0 / float(false_edge_scores.shape[0])
    neg_plotting_weights = np.ones(false_edge_scores.shape[0]) * neg_plotting_weights
    
    plt.figure()
    plt.hist(true_edge_scores, bins=50, range=(0, 1.1), alpha=0.3, color='blue', edgecolor='black', label='true_edge_scores', weights=pos_plotting_weights)
    plt.hist(false_edge_scores, bins=50, range=(0, 1.1), alpha=0.3, color='red', edgecolor='black', label='false_edge_scores', weights=neg_plotting_weights)

    #plt.yscale('log')
    #plt.title(train_or_test)
    plt.legend()
    plt.show()    
    
############################################################################################################################################  
    
def draw_confusion_with_threshold(scores, true_labels, threshold):
    
    nClasses = 2
    
    scores = np.copy(scores)
    true_labels = np.copy(true_labels)
    
    scores = scores.reshape(-1)
    true_labels = true_labels.reshape(-1)
    
    predicted_true_mask = scores > threshold
    predicted_false_mask = np.logical_not(predicted_true_mask)
    scores[predicted_true_mask] = 1
    scores[predicted_false_mask] = 0

    confMatrix = confusion_matrix(true_labels, scores)
    
    print(confMatrix)
    
    trueSums = np.sum(confMatrix, axis=1)
    predSums = np.sum(confMatrix, axis=0)

    trueNormalised = np.zeros(shape=(nClasses, nClasses))
    predNormalised = np.zeros(shape=(nClasses, nClasses))

    for trueIndex in range(nClasses) : 
        for predIndex in range(nClasses) :
            nEntries = confMatrix[trueIndex][predIndex]
            if trueSums[trueIndex] > 0 :
                trueNormalised[trueIndex][predIndex] = float(nEntries) / float(trueSums[trueIndex])
            if predSums[predIndex] > 0 :
                predNormalised[trueIndex][predIndex] = float(nEntries) / float(predSums[predIndex])

    displayTrueNorm = ConfusionMatrixDisplay(confusion_matrix=trueNormalised, display_labels=["False", "True"])
    displayTrueNorm.plot()

    displayPredNorm = ConfusionMatrixDisplay(confusion_matrix=predNormalised, display_labels=["False", "True"])
    displayPredNorm.plot()
    
############################################################################################################################################
# Functions for individual link predictions
############################################################################################################################################ 

def plot_multi(scores_train, true_labels_train, scores_test, true_labels_test, which_class):
       
    plt.figure()
    
    count = 0
        
    for (scores, true_labels) in ((scores_train, true_labels_train), (scores_test, true_labels_test)) :
        
        true_labels = true_labels.argmax(axis=1).reshape(-1)
        true_edge_mask = (true_labels == 1)
        wrong_orientation_edge_mask = (true_labels == 2)
        false_edge_mask = (true_labels == 0)

        scores = scores[:,which_class].reshape(-1)
        true_edge_scores = scores[true_edge_mask]
        wrong_orientation_scores = scores[wrong_orientation_edge_mask]
        false_edge_scores = scores[false_edge_mask]

        pos_plotting_weights = 1.0 / float(true_edge_scores.shape[0])
        pos_plotting_weights = np.ones(true_edge_scores.shape[0]) * pos_plotting_weights

        wrong_orientation_weights = 1.0 / float(wrong_orientation_scores.shape[0])
        wrong_orientation_weights = np.ones(wrong_orientation_scores.shape[0]) * wrong_orientation_weights

        neg_plotting_weights = 1.0 / float(false_edge_scores.shape[0])
        neg_plotting_weights = np.ones(false_edge_scores.shape[0]) * neg_plotting_weights


        plt.hist(true_edge_scores, bins=50, range=(0, 1.1), color='blue', label='true_edge_scores', weights=pos_plotting_weights, fill=False, linestyle=('solid' if count == 0 else 'dashed'), histtype='step')
        plt.hist(wrong_orientation_scores, bins=50, range=(0, 1.1), color='orange', label='wrong_orientation_edge_scores', weights=wrong_orientation_weights, fill=False, linestyle=('solid' if count == 0 else 'dashed'), histtype='step')
        plt.hist(false_edge_scores, bins=50, range=(0, 1.1), color='red', label='false_edge_scores', weights=neg_plotting_weights, fill=False, linestyle=('solid' if count == 0 else 'dashed'), histtype='step')

        count = count + 1
        
        
    if (which_class == 0) :
        plt.title('Background Score')
    elif (which_class == 1) :
        plt.title('Signal Score')
    elif (which_class == 2) :
        plt.title('Signal WO Score')
        
    plt.legend()
    plt.show()    
    
############################################################################################################################################  

def draw_confusion_multi(scores, true_labels):
    
    nClasses = 3
    
    true_labels = true_labels.argmax(axis = 1)
    pred_labels = scores.argmax(axis = 1)
    
    confMatrix = confusion_matrix(true_labels, pred_labels)
    
    print(confMatrix)
    
    trueSums = np.sum(confMatrix, axis=1)
    predSums = np.sum(confMatrix, axis=0)
    
    trueNormalised = np.zeros(shape=(nClasses, nClasses))
    predNormalised = np.zeros(shape=(nClasses, nClasses))

    for trueIndex in range(nClasses) : 
        for predIndex in range(nClasses) :
            nEntries = confMatrix[trueIndex][predIndex]
            if trueSums[trueIndex] > 0 :                
                trueNormalised[trueIndex][predIndex] = float(nEntries) / float(trueSums[trueIndex])
            if predSums[predIndex] > 0 :
                predNormalised[trueIndex][predIndex] = float(nEntries) / float(predSums[predIndex])

    displayTrueNorm = ConfusionMatrixDisplay(confusion_matrix=trueNormalised, display_labels=["False", "True", "WO"])
    displayTrueNorm.plot()

    displayPredNorm = ConfusionMatrixDisplay(confusion_matrix=predNormalised, display_labels=["False", "True", "WO"])
    displayPredNorm.plot()

############################################################################################################################################  