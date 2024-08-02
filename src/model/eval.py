from sklearn.metrics import *
from scikitplot.metrics import plot_confusion_matrix

def evaluate_model(y_true, y_pred):
    plot_confusion_matrix(y_true,y_pred)

    print('Accuracy_score: ', accuracy_score(y_true,y_pred))
    print('Precision_score: ', precision_score(y_true,y_pred, average='macro'))
    print('Recall_score: ', recall_score(y_true,y_pred,average='macro'))
    print('F1 score', f1_score(y_true,y_pred,average='macro'))
    print("-"*50)
    cr = classification_report(y_true, y_pred, digits=4)
    print(cr)