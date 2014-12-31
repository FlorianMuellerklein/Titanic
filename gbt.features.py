import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier

train_loc = 'train.csv'
test_loc = 'test.csv'

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
    
def load_fit_data(path):
    fit_x = pd.read_csv(path)
    fit_x = clean(fit_x)
    fit_y = fit_x['Survived'].astype(int).values
    fit_x = fit_x.drop('Survived', 1)
		
    fit_x = fit_x.values
	
    return fit_x, fit_y

def vw_ready(data):
	data[data == 0] = -1
	data = (data.astype(str) + ' |C').as_matrix()
	
	return data

def load_train_vw(path, gbt):
    reader = pd.read_csv(path, chunksize = 100)
    for chunk in reader:
        chunk = clean(chunk)
        survived = chunk['Survived'].astype(int)
        chunk = chunk.drop('Survived', 1)
        chunk = chunk.drop('PassengerId', 1)
        orig = []
        for colname in list(chunk.columns.values):
            orig.append(colname + chunk[colname].astype(str))
        
        chunk = chunk.values
        orig = np.column_stack(orig)

        gbt_tree = get_leaf_indices(gbt, chunk).astype(str)
        for row in range(0, chunk.shape[0]):
            for column in range(0,500,1):
                gbt_tree[row,column] = ('C' + str(column) + str(gbt_tree[row, column]))
        
        survived = vw_ready(survived)
        
        out = np.column_stack((survived, orig, gbt_tree))
        
        file_handle = file('tree.train.txt', 'a')
        np.savetxt(file_handle, out, delimiter = ' ', fmt = '%s')
        file_handle.close()
        
def load_test_vw(path, gbt):
    reader = pd.read_csv(path, chunksize = 100)
    for chunk in reader:
        chunk = clean(chunk)
        pid = chunk['PassengerId']
        chunk = chunk.drop('PassengerId', 1)
        orig = []
        for colname in list(chunk.columns.values):
            orig.append(colname + chunk[colname].astype(str))
        
        chunk = chunk.values
        orig = np.column_stack(orig)

        gbt_tree = get_leaf_indices(gbt, chunk).astype(str)
        for row in range(0, chunk.shape[0]):
            for column in range(0,500,1):
                gbt_tree[row,column] = ('C' + str(column) + str(gbt_tree[row, column]))
        
        pid = (pid.astype(str) + ' |C').as_matrix()
        
        out = np.column_stack((pid, orig, gbt_tree))
        
        file_handle = file('tree.test.txt', 'a')
        np.savetxt(file_handle, out, delimiter = ' ', fmt = '%s')
        file_handle.close()

	
def main():
    gbt = GradientBoostingClassifier(n_estimators = 500, max_depth = 7, verbose = 1)
	
    fit_x, fit_y = load_fit_data(train_loc)
    
    gbt.fit(fit_x, fit_y)
    fit_x = None
    fit_y = None
    
    print('transforming and writing training data ... ')
    load_train_vw(train_loc, gbt)
    
    print('transforming and writing testing data ... ')
    load_test_vw(test_loc, gbt)
    
if __name__ == '__main__':
    main()
    
