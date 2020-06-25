import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
import os

# generate data
def generate_dataset(X, Y, G, kernel, clean_ratio, attacker_split, test_split, random_seed):
    """
    kernel: we use sklearn SVM to pre-process the data. kernel is linear or rbf
    clean_ratio: fraction of data left after removing hard examples
    attacker_split: fraction of data given to the attacker
    test_split: train/test split ratio
    """

    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)


    n_clean = int(clean_ratio*Y.shape[0])
    n_attacker = int(n_clean*attacker_split)
    
    if kernel == 'linear':
        model = LinearSVC(max_iter=5000)
    else:
        model = SVC(gamma='auto')
    # Train the model, iterating on the data in batches of 32 samples
    model.fit(X_scaled, Y)

    loss = np.absolute(np.subtract(model.predict(X_scaled), Y))

    index = np.argsort(loss, axis=0)

    x_clean = X_scaled[index[:n_clean],:]
    y_clean = Y[index[:n_clean]]
    g_clean = G[index[:n_clean]]

    x_noise = X_scaled[index[n_clean:],:]
    y_noise = Y[index[n_clean:]]
    g_noise = G[index[n_clean:]]

    attacker_index = np.random.choice(y_clean.shape[0], n_attacker, replace = False)
    x_attacker = x_clean[attacker_index]
    y_attacker = y_clean[attacker_index]
    g_attacker = g_clean[attacker_index]
    x_clean = np.delete(x_clean, attacker_index, 0)
    y_clean = np.delete(y_clean, attacker_index, 0)
    g_clean = np.delete(g_clean, attacker_index, 0)

    x_train,x_test,y_train,y_test,g_train,g_test = train_test_split(x_clean, y_clean, g_clean, test_size = test_split, random_state=random_seed)

    data = {
            'x_train': x_train,
            'y_train': y_train,
            'g_train': g_train,
            'x_test': x_test,
            'y_test': y_test,
            'g_test': g_test,
            'x_attacker': x_attacker,
            'y_attacker': y_attacker,
            'g_attacker': g_attacker,
            'x_noise': x_noise,
            'y_noise': y_noise,
            'g_noise': g_noise

        }
    return data

# accuracy measure
def accuracy(y_true, y_pred):
    if len(y_true)==0:
        return 0
    return 1 - sum([abs(y_true[i] - y_pred[i]) for i in range(len(y_true))])/len(y_true)

# fairness measure
def EO(s, y_pred, y_true):
    y_pred_0_0 = [(1-y_pred[i]) for i in range(len(s)) if y_true[i] == 0 and s[i] == 0]
    y_pred_1_0 = [(1-y_pred[i]) for i in range(len(s)) if y_true[i] == 0 and s[i] == 1]
    y_pred_0_1 = [y_pred[i] for i in range(len(s)) if y_true[i] == 1 and s[i] == 0]
    y_pred_1_1 = [y_pred[i] for i in range(len(s)) if y_true[i] == 1 and s[i] == 1]
    
    loss_0_0 = np.mean(y_pred_0_0)
    loss_1_0 = np.mean(y_pred_1_0)
    loss_0_1 = np.mean(y_pred_0_1)
    loss_1_1 = np.mean(y_pred_1_1)
    
    return abs(loss_0_0 - loss_1_0), abs(loss_0_1 - loss_1_1)

# functions
def cross_entropy(y, t, tol=1e-12):
    pred = np.clip(y, tol, 1-tol)
    pred_n = np.clip(1-pred, tol, 1-tol)
    return - (np.sum(np.multiply(t, np.log(pred)),axis=1) + np.sum(np.multiply(1-t, np.log(pred_n)),axis=1))


# evaluate penalized loss
def eval_loss(A, b, x, y, g, x_reg, y_reg, g_reg, L, num_points):
    
    s = 1 - (np.dot(x, A) + b) * (2*y-1)
    loss = s*(s>=0)

    loss2 = (1-(2*y-1)*(np.dot(x, A) + b))/2
    loss2_reg = (1-(2*y_reg-1)*(np.dot(x_reg, A) + b))/2

    idx00 = np.logical_and(g_reg.flatten()==0, y_reg.flatten()==0)
    idx01 = np.logical_and(g_reg.flatten()==0, y_reg.flatten()==1)
    idx10 = np.logical_and(g_reg.flatten()==1, y_reg.flatten()==0)
    idx11 = np.logical_and(g_reg.flatten()==1, y_reg.flatten()==1)

    s00 = np.sum(loss2_reg[idx00])
    s01 = np.sum(loss2_reg[idx01])
    s10 = np.sum(loss2_reg[idx10])
    s11 = np.sum(loss2_reg[idx11])

    c00 = np.sum(idx00)
    c01 = np.sum(idx01)
    c10 = np.sum(idx10)
    c11 = np.sum(idx11)


    for i in range(len(loss)):
        if g[i][0]==0 and y[i][0]==0:
            s00 += num_points*loss2[i][0]
            c00 += num_points
        elif g[i][0]==0 and y[i][0]==1:
            s01 += num_points*loss2[i][0]
            c01 += num_points
        elif g[i][0]==1 and y[i][0]==0:
            s10 += num_points*loss2[i][0]
            c10 += num_points
        elif g[i][0]==1 and y[i][0]==1:
            s11 += num_points*loss2[i][0]
            c11 += num_points
        reg = L*np.abs(s00/c00-s10/c10) + L*np.abs(s01/c01-s11/c11)
        loss[i][0] += reg
    return loss

# gradient
# note: write 2y-1 to convert to +/-1 prediction to use Hinge loss
def gradient(A, b, x_loss, y_loss, x_reg, y_reg, g_reg, L, r, n, num_points):

    s = 1 - (np.dot(x_loss, A) + b) * (2*y_loss-1)

    dfA_loss = np.sum((-x_loss*(2*y_loss-1))*(s>=0), axis=0).reshape((-1,1))
    dfb_loss = np.sum(-(2*y_loss-1)*(s>=0))

    idx00 = np.logical_and(g_reg.flatten()==0, y_reg.flatten()==0)
    idx01 = np.logical_and(g_reg.flatten()==0, y_reg.flatten()==1)
    idx10 = np.logical_and(g_reg.flatten()==1, y_reg.flatten()==0)
    idx11 = np.logical_and(g_reg.flatten()==1, y_reg.flatten()==1)

    loss2_reg = (1-(2*y_reg-1)*(np.dot(x_reg, A) + b))/2

    s00 = np.sum(loss2_reg[idx00])
    s01 = np.sum(loss2_reg[idx01])
    s10 = np.sum(loss2_reg[idx10])
    s11 = np.sum(loss2_reg[idx11])

    c00 = np.sum(idx00)
    c01 = np.sum(idx01)
    c10 = np.sum(idx10)
    c11 = np.sum(idx11)

    dfA_reg_0 = (L*np.sign(s00/c00-s10/c10)*(np.mean(-(2*y_reg[idx00]-1)*x_reg[idx00]/2, axis=0) - np.mean(-(2*y_reg[idx10]-1)*x_reg[idx10]/2, axis=0))).reshape((-1,1))
    dfb_reg_0 = L*np.sign(s00/c00-s10/c10)*(-sum(2*y_reg[idx00]-1)/2/len(idx00) - -sum(2*y_reg[idx10]-1)/2/len(idx10))


    dfA_reg_1 = (L*np.sign(s01/c01-s11/c11)*(np.mean(-(2*y_reg[idx01]-1)*x_reg[idx01]/2, axis=0) - np.mean(-(2*y_reg[idx11]-1)*x_reg[idx11]/2, axis=0))).reshape((-1,1))
    dfb_reg_1 = L*np.sign(s01/c01-s11/c11)*(-sum(2*y_reg[idx01]-1)/2/len(idx01) - -sum(2*y_reg[idx11]-1)/2/len(idx11))

    dA_reg = (2*A).reshape((-1,1))
    db_reg = 2*b


    return {
        'dA': ((dfA_loss+num_points*dfA_reg_0+num_points*dfA_reg_1)/n+r*dA_reg).reshape((-1,1)) ,
        'db': (dfb_loss+num_points*dfb_reg_0+num_points*dfb_reg_1)/n#+r*db_reg
    }

