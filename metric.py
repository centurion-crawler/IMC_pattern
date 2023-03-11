
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report,accuracy_score,roc_curve,auc,roc_auc_score
from collections import Counter
from IMC_Dataset import get_dataloader

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def five_scores(labels, predictions):
    fpr, tpr, threshold = roc_curve(labels, predictions)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    auc_value = roc_auc_score(labels, predictions)
    this_class_label = np.array(predictions)
    this_class_label[this_class_label>=threshold_optimal] = 1
    this_class_label[this_class_label<threshold_optimal] = 0
    predictions = this_class_label

    # precision, recall, fscore, _ = precision_recall_fscore_support(bag_labels, bag_predictions, average='binary')
    acc=accuracy_score(labels, predictions)
    # return accuracy, auc_value, precision, recall, fscore
    return acc, auc_value
def return_auc(target_array,possibility_array):
    enc = OneHotEncoder()
    target_onehot = enc.fit_transform(target_array.unsqueeze(1))
    target_onehot = target_onehot.toarray()
    class_auc_list = []
    for i in range(num_classes):
        # print(target_onehot[:,i],possibility_array[:,i])
        class_i_auc = roc_auc_score(target_onehot[:,i], possibility_array[:,i])
        class_auc_list.append(class_i_auc)
    macro_auc = roc_auc_score(np.round(target_onehot,0), possibility_array, average="macro", multi_class="ovo")
    return macro_auc, class_auc_list