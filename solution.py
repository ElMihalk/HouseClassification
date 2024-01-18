import os
import requests
import sys
import pandas as pd
import sklearn.model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

if __name__ == '__main__':
    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    # Download data if it is unavailable.
    if 'house_class.csv' not in os.listdir('../Data'):
        sys.stderr.write("[INFO] Dataset is loading.\n")
        url = "https://www.dropbox.com/s/7vjkrlggmvr5bc1/house_class.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/house_class.csv', 'wb').write(r.content)
        sys.stderr.write("[INFO] Loaded.\n")

    initial_data = pd.read_csv(r'C:\Users\User\PycharmProjects\House Classification1\House Classification\Data\house_class.csv', header=0)
    # print(initial_data)
    # print(f'Number of rows: {initial_data.shape[0]}')
    # print(f'Number of columns: {initial_data.shape[1]}')
    # print(f'Misssing values: {initial_data.isna().sum().sum()}')
    # print(f'Maximum number of rooms: {max(initial_data["Room"])}')
    # print(f'Mean area of the houses: {initial_data["Area"].mean()}')
    # print(f'Unique values in Zip_loc: {initial_data["Zip_loc"].nunique()}')

    #result of Stage 1/6: Import & explore
    # print(f'{initial_data.shape[0]}\n{initial_data.shape[1]}\n{bool(initial_data.isna().sum().sum())}\n'
    #       f'{max(initial_data["Room"])}\n{initial_data["Area"].mean()}\n{initial_data["Zip_loc"].nunique()}')

    #Stage 2/6: Split the data
    y = initial_data['Price']
    X = initial_data.loc[:, 'Area':'Zip_loc']
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, stratify=X['Zip_loc'].values, random_state=1)
    # print(dict(X_train['Zip_loc'].value_counts()))

    #Stage 3/6: One-hot encode the data
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score

    enc = OneHotEncoder(drop='first')
    #Encode train data.
    enc.fit(X_train[['Zip_area', 'Zip_loc', 'Room']])
    X_train_transformed = pd.DataFrame(enc.transform(X_train[['Zip_area', 'Zip_loc', 'Room']]).toarray(),
                                       index=X_train.index).add_prefix('enc')
    X_train_final = X_train[['Area', 'Lon', 'Lat']].join(X_train_transformed)

    #Encode test data.
    X_test_transformed = pd.DataFrame(enc.transform(X_test[['Zip_area', 'Zip_loc', 'Room']]).toarray(),
                                       index=X_test.index).add_prefix('enc')
    X_test_final = X_test[['Area', 'Lon', 'Lat']].join(X_test_transformed)

    #Fit and predict with DecisionTree
    clf = DecisionTreeClassifier(criterion='entropy', max_features=3, splitter='best', max_depth=6, min_samples_split=4, random_state=3)
    clf.fit(X_train_final, y_train)
    y_test_predicted = clf.predict(X_test_final)
    # print(accuracy_score(y_true=y_test, y_pred=y_test_predicted))
    result_onehot = precision_recall_fscore_support(y_test, y_test_predicted, average='macro')


    #Stage 4/6: Ordinal encoder
    from sklearn.preprocessing import OrdinalEncoder

    enc = OrdinalEncoder()
    enc.fit(X_train[['Zip_area', 'Zip_loc', 'Room']])

    # Encode train data.
    X_train_transformed = pd.DataFrame(enc.transform(X_train[['Zip_area', 'Zip_loc', 'Room']]),
                                       index=X_train.index).add_prefix('enc')
    X_train_final = X_train[['Area', 'Lon', 'Lat']].join(X_train_transformed)

    # Encode test data.
    X_test_transformed = pd.DataFrame(enc.transform(X_test[['Zip_area', 'Zip_loc', 'Room']]),
                                      index=X_test.index).add_prefix('enc')
    X_test_final = X_test[['Area', 'Lon', 'Lat']].join(X_test_transformed)

    #Fit and predict with DecisionTree
    clf = DecisionTreeClassifier(criterion='entropy', max_features=3, splitter='best', max_depth=6, min_samples_split=4, random_state=3)
    clf.fit(X_train_final, y_train)
    y_test_predicted = clf.predict(X_test_final)
    # print(accuracy_score(y_true=y_test, y_pred=y_test_predicted))
    result_ord = precision_recall_fscore_support(y_test, y_test_predicted, average='macro')

    #Stage 5/6: Target encoder
    from category_encoders import TargetEncoder

    enc = TargetEncoder()
    # enc.fit(X_train[['Zip_area', 'Room', 'Zip_loc']], y_train)
    enc.fit(X_train[['Zip_area', 'Zip_loc', 'Room']], y_train)

    # Encode train data.
    # X_train_transformed = pd.DataFrame(enc.transform(X_train[['Zip_area', 'Room', 'Zip_loc']]),
    #                                    index=X_train.index)
    X_train_transformed = pd.DataFrame(enc.transform(X_train[['Zip_area', 'Zip_loc', 'Room']]),
                                       index=X_train.index)
    X_train_final = X_train[['Area', 'Lon', 'Lat']].join(X_train_transformed)

    # Encode test data.
    # X_test_transformed = pd.DataFrame(enc.transform(X_test[['Zip_area', 'Room', 'Zip_loc']]),
    #                                   index=X_test.index)
    X_test_transformed = pd.DataFrame(enc.transform(X_test[['Zip_area', 'Zip_loc', 'Room']]),
                                      index=X_test.index)
    X_test_final = X_test[['Area', 'Lon', 'Lat']].join(X_test_transformed)

    #Fit and predict with DecisionTree
    clf = DecisionTreeClassifier(criterion='entropy', max_features=3, splitter='best', max_depth=6, min_samples_split=4, random_state=3)
    clf.fit(X_train_final, y_train)
    y_test_predicted = clf.predict(X_test_final)
    # print(accuracy_score(y_true=y_test, y_pred=y_test_predicted))
    result_target = precision_recall_fscore_support(y_test, y_test_predicted, average='macro')

    #Stage 6/6: Performance comparison
    print(f'OneHotEncoder:{round(result_onehot[2], 2)}')
    print(f'OrdinalEncoder:{round(result_ord[2], 2)}')
    print(f'TargetEncoder:{round(result_target[2], 2)}')

