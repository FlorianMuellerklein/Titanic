import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier

train_loc = 'train.csv'
test_loc = 'test.csv'
TREES = 30
NODES = 7

def get_leaf_indices(ensemble, x):
    x = x.astype(np.float32)
    trees = ensemble.estimators_
    n_trees = trees.shape[0]
    indices = []

    for i in range(n_trees):
        tree = trees[i][0].tree_
        indices.append(tree.apply(x))

    indices = np.column_stack(indices)
    return indices
    
# clean data
def clean(data):
    data['Age'].fillna(data['Age'].median(), inplace = True)
    data['FamilySize'] = data['SibSp'] + data['Parch']
    data['AgeClass'] = data.Age * data.Pclass
    data['Gender'] = data['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
    data = data.drop(['Name', 'Ticket', 'Cabin', 'Embarked', 'Sex'], axis = 1)
    
    return data
    
def load_fit_data(path, gbt):
    fit_x = pd.read_csv(path)
    fit_x = clean(fit_x)
    fit_y = fit_x['Survived'].astype(int).values
    fit_x = fit_x.drop('Survived', 1)
		
    fit_x = pd.get_dummies(fit_x).values
    
    gbt.fit(fit_x, fit_y)
	
    return gbt

def vw_ready(data):
	data[data == 0] = -1
	data = (data.astype(str) + ' |C').as_matrix()
	
	return data

def load_data(path, gbt, train):
    reader = pd.read_csv(path, chunksize = 100)
    for chunk in reader:
        if train == True:
            chunk = clean(chunk)
            y = chunk['Survived'].astype(int)
            chunk = chunk.drop('Survived', 1)
            chunk = chunk.drop('PassengerId', 1)
            y = vw_ready(y)
        else:
            chunk = clean(chunk)
            y = chunk['PassengerId']
            chunk = chunk.drop('PassengerId', 1)
            y = (y.astype(str) + ' |C').as_matrix()
        
        orig = []
        for colname in list(chunk.columns.values):
            orig.append(colname + chunk[colname].astype(str))
        
        orig = np.column_stack(orig)

        gbt_tree = get_leaf_indices(gbt, pd.get_dummies(chunk).values).astype(str)
        chunk = chunk.values
        for row in range(0, chunk.shape[0]):
            for column in range(0, TREES, 1):
                gbt_tree[row,column] = ('T' + str(column) + str(gbt_tree[row, column]))
        
        
        out = np.column_stack((y, orig, gbt_tree))
        
        if train == True:
            file_handle = file('tree.train.txt', 'a')
            np.savetxt(file_handle, out, delimiter = ' ', fmt = '%s')
            file_handle.close()
        else:
            file_handle = file('tree.test.txt', 'a')
            np.savetxt(file_handle, out, delimiter = ' ', fmt = '%s')
            file_handle.close()
            
def main():
    gbt = GradientBoostingClassifier(n_estimators = TREES, max_depth = NODES, verbose = 1)
    
    gbt = load_fit_data(train_loc, gbt)
    
    print('transforming and writing training data ... ')
    load_data(train_loc, gbt, train = True)
    
    print('transforming and writing testing data ... ')
    load_data(test_loc, gbt, train = False)
    
if __name__ == '__main__':
    main()
    
