# Feel free to add any functions, import statements, and variables.

import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

def importfiles():
    #read files
    traindata = pd.read_csv('train.csv')
    testdata = pd.read_csv('test.csv')
    X_raw_df, y_train = traindata.drop('Class', axis=1), traindata.Class
    #scale
    standardize = ['Time', 'Amount']
    scaled = preprocessing.StandardScaler().fit(X_raw_df[standardize])
    columns_scaled = scaled.transform(X_raw_df[standardize])
    X_train = X_raw_df.copy()
    X_train[standardize] = columns_scaled
    
    scaled2 = preprocessing.StandardScaler().fit(testdata[standardize])
    columns_scaled2 = scaled2.transform(testdata[standardize])
    X_test = testdata.copy()
    X_test[standardize] = columns_scaled2
    return X_train, y_train, X_test
 

def predict(file):
    # Fill in this function. This function should return a list of length 10,000
    #   which is filled with values in {0, 1}. For example, the current
    #   implementation predicts all the instances in test.csv as abnormal, so
    #   the precision is 0.01 and recall is 1.
    
    # call impor files
    X_train, y_train, X_test = importfiles()
    rf_model = RandomForestClassifier(class_weight='balanced', max_features=25, max_depth=9, n_estimators=20)
    rf_model.fit(X=X_train, y=y_train)
    y_predicted_rf = rf_model.predict(X=X_test)
    return list(y_predicted_rf)
    #return list([1 for _ in range(10000)])


def write_result(classes):
    # You don't need to modify this function.
    with open('result.csv', 'w') as f:
        f.write('Index,Class\n')
        for idx, l in enumerate(classes):
            f.write('{0},{1}\n'.format(idx, l))


def main():
    # You don't need to modify this function.
    classes = predict('test.csv')
    write_result(classes)


if __name__ == '__main__':
    # You don't need to modify this part.
    main()
