import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from util import set_random_seed

set_random_seed(47)

class RBMO:
    """
    Fu, S., Li, K., Huang, H. et al. Red-billed blue magpie optimizer: a novel metaheuristic
    algorithm for 2D/3D UAV path planning and engineering design problems.
    Artif Intell Rev 57, 134 (2024).
    https://doi.org/10.1007/s10462-024-10716-3
    """

    def __init__(self, obj_func, lb, ub, dim, pop_size, max_iter):
        """
        初始化 RBMO 优化器.

        参数:
        - obj_func: 目标函数 (需要最小化).
        - lb: 决策变量的下界 (list 或 numpy array).
        - ub: 决策变量的上界 (list 或 numpy array).
        - dim: 决策变量的维度.
        - pop_size: 种群大小 (个体数量).
        - max_iter: 最大迭代次数.
        """
        self.obj_func = obj_func
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        
        self.epsilon = 0.5

    def solve(self):

        positions = self.lb + np.random.rand(self.pop_size, self.dim) * (self.ub - self.lb)
        fitness = np.full(self.pop_size, np.inf)

        g_best_val = np.inf
        g_best_pos = np.zeros(self.dim)
        
        convergence_curve = np.zeros(self.max_iter)

        for t in range(self.max_iter):
            for i in range(self.pop_size):
                fitness[i] = self.obj_func(positions[i])

            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < g_best_val:
                g_best_val = fitness[current_best_idx]
                g_best_pos = positions[current_best_idx].copy()
            x_food = g_best_pos
            cf = (1 - (t / self.max_iter)) ** (2 * (t / self.max_iter))
            new_positions = np.zeros_like(positions)
            
            for i in range(self.pop_size):
                if np.random.rand() < self.epsilon:
                    x_rs_idx = np.random.randint(0, self.pop_size)
                    x_rs = positions[x_rs_idx]
                    
                    if np.random.rand() < 0.5: 
                        p = np.random.randint(2, 6) 
                        rand_indices = np.random.choice(self.pop_size, p, replace=False)
                        mean_xm = np.mean(positions[rand_indices], axis=0)
                        new_pos = positions[i] + np.random.rand() * (mean_xm - x_rs)
                    else: 
                        q = np.random.randint(10, self.pop_size + 1) 
                        rand_indices = np.random.choice(self.pop_size, q, replace=False)
                        mean_xm = np.mean(positions[rand_indices], axis=0)
                        new_pos = positions[i] + np.random.rand() * (mean_xm - x_rs)
                else:
                    if np.random.rand() < 0.5: 
                        p = np.random.randint(2, 6)
                        rand_indices = np.random.choice(self.pop_size, p, replace=False)
                        mean_xm = np.mean(positions[rand_indices], axis=0)
                        new_pos = x_food + cf * (mean_xm - positions[i]) * np.random.randn()
                    else: 
                        q = np.random.randint(10, self.pop_size + 1)
                        rand_indices = np.random.choice(self.pop_size, q, replace=False)
                        new_pos = x_food + cf * (mean_xm - positions[i]) * np.random.randn()
                
                new_positions[i] = new_pos

            new_positions = np.clip(new_positions, self.lb, self.ub)


            new_fitness = np.array([self.obj_func(pos) for pos in new_positions])
            
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    positions[i] = new_positions[i]
                    fitness[i] = new_fitness[i]

            convergence_curve[t] = g_best_val
            if (t + 1) % 50 == 0:
                print(f"迭代 {t + 1}: 最佳适应度值 = {g_best_val}")
        
        return g_best_val, g_best_pos, convergence_curve



# inpath = './datasets/featurized/protein_bert/'
inpath = './datasets/featurized/prostT5/'
# outpath = './datasets/RBMO/protein_bert/'
outpath = './datasets/RBMO/prostT5/'
protein_cols = [f'ProstT5_{i}' for i in range(1, 1025)]
# protein_cols = [f'Probert_{i}' for i in range(1, 1025)]
path_train = inpath + 'train.csv'
path_test = inpath + 'test.csv'
df_train = pd.read_csv(path_train)
df_test = pd.read_csv(path_test)
X_train = df_train[protein_cols].values.tolist()
X_test = df_test[protein_cols].values.tolist()
y_train = df_train['Label'].tolist()
y_test = df_test['Label'].tolist()


X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)



def feature_selection_objective_function(solution_vector):


    binary_selection = (np.array(solution_vector) > 0.5).astype(int)
    selected_indices = np.where(binary_selection == 1)[0]
    

    if len(selected_indices) == 0:
        return 1.0 


    X_train_subset = X_train[:, selected_indices]
    

    classifier = KNeighborsClassifier(n_neighbors=3)
    accuracies = cross_val_score(classifier, X_train_subset, y_train, cv=3)
    error_rate = 1.0 - np.mean(accuracies)
    
    alpha = 0.005 
    cost = error_rate + alpha * (len(selected_indices) / X_train.shape[1])
    
    return cost

total_features = X_train.shape[1]

optimizer = RBMO(
    obj_func=feature_selection_objective_function,
    lb=[0] * total_features,
    ub=[1] * total_features,
    dim=total_features,
    pop_size=20, 
    max_iter=50
)

best_cost, best_solution, _ = optimizer.solve()


final_selection = (best_solution > 0.5).astype(int)
final_indices = np.where(final_selection == 1)[0]

X_train_selected = X_train[:, final_indices]
X_test_selected = X_test[:, final_indices]
train_list = X_train_selected.tolist()
test_list = X_test_selected.tolist()

print("\n--- 结果 ---")
print(f"对应的特征列索引: {len(final_indices)}")
print(f"对应的特征列索引: {final_indices}")
np.save('./datasets/RBMO/final_indices.npy', final_indices)

# df_train_smiles = df_train['SMILES']
# df_train_proteins = df_train['Target Sequence']
# df_train_label = df_train['Label']
# df_train_selected = pd.DataFrame()
# df_train_selected['SMILES'] = df_train_smiles
# df_train_selected['Target Sequence'] = df_train_proteins
# df_train_selected['Label'] = df_train_label

# df_test_smiles = df_test['SMILES']
# df_test_proteins = df_test['Target Sequence']
# df_test_label = df_test['Label']
# df_test_selected = pd.DataFrame()
# df_test_selected['SMILES'] = df_test_smiles
# df_test_selected['Target Sequence'] = df_test_proteins
# df_test_selected['Label'] = df_test_label

# for i in range(len(train_list[0])):
#     global_feature_n = [train_list[x][i] for x in range(len(train_list))]
#     # df_train_selected['Probert_'+str(i+1)] = global_feature_n
#     df_train_selected['ProstT5_'+str(i+1)] = global_feature_n
# df_train_selected.to_csv(outpath + 'train.csv', index=False)

# for i in range(len(test_list[0])):
#     global_feature_n = [test_list[x][i] for x in range(len(test_list))]
#     # df_test_selected['Probert_'+str(i+1)] = global_feature_n
#     df_test_selected['ProstT5_'+str(i+1)] = global_feature_n
# df_test_selected.to_csv(outpath + 'test.csv', index=False)


