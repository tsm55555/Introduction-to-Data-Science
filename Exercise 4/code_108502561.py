import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def mf_sgd(R, K=64, alpha=1e-4, beta=1e-2, iterations=50):
    """
    :param R: user-item rating matrix
    :param K: number of latent dimensions
    :param alpha: learning rate
    :param beta: regularization parameter
    """
    num_users, num_items = R.shape #(610, 9724)
    #print(R.shape)
    
    # Initialize user and item latent feature matrice
    P = np.random.normal(scale=1. / K, size=(num_users, K))
    Q = np.random.normal(scale=1. / K, size=(num_items, K))

    # Initialize the biases 
    b_u = np.zeros(num_users)
    b_i = np.zeros(num_items)
    b = np.mean(R[np.where(R != 0)])

    # Create a list of training samples
    samples = [
        (i, j, R[i, j])
        for i in range(num_users)
        for j in range(num_items)
        if R[i, j] > 0
    ]


    # Perform stochastic gradient descent for number of iterations
    training_loss = []
    
    for iters in range(iterations):
        np.random.shuffle(samples)

        # i user, j items, r rating
        for i, j, r in samples:
            """
            TODO: 
            In this for-loop scope, 
            you need to (1)update "b_u"(vector of rating bias for users) and "b_i"(vector of rating bias for items)
            and (2)update user and item latent feature matrices "P", "Q"
            """
            r_hat = b + b_u[i] + b_i[j] + np.dot(P[i], Q[j]) # r_hat_ij = 𝜇 + b_u + b_i + p_i * q_j
            d = r - r_hat

            # we can derive from teacher's slides that
            # theta = theta - 𝜂*∇ -> 
            # b = b + alpha(d - beta * b) ...
            # and p = p + alpha(d * q - beta * p) ... 
            # according to the hyperparameters given
            b_u[i] += alpha * (d - beta * b_u[i])
            b_i[j] += alpha * (d - beta * b_i[j])
            P[i] += alpha * (d * Q[j] - beta * P[i])
            Q[j] += alpha * (d * P[i] - beta * Q[j])

        pred = compute_training_loss(R, b, b_u, b_i, P, Q, iters, training_loss)

    return pred, b, b_u, b_i, training_loss

def compute_training_loss(R, b, b_u, b_i, P, Q, iters, training_loss):
    users, items = R.nonzero()
    error = 0
    pred = b + b_u[:, np.newaxis] + b_i[np.newaxis:, ] + P.dot(Q.T) 
    for i, j in zip(users, items):
        d = R[i, j] - pred[i, j]
        error += d**2 
    RMSE = np.sqrt(error / len(users))
    training_loss.append((iters, RMSE))
    return pred


def plot_training_loss(training_loss):
    x = [x for x, y in training_loss]
    y = [y for x, y in training_loss]
    plt.figure(figsize=(16, 4))
    plt.plot(x, y)
    plt.xticks(x, x)
    plt.xlabel("Iterations")
    plt.ylabel("Root Mean Square Error")
    plt.grid(axis="y")
    plt.savefig("training_loss.png")
    plt.show()


if __name__ == "__main__":
    data = pd.read_csv('/home/tsm62803/my_code/Introduction-to-Data-Science/Exercise 4/ratings.csv')
    table = pd.pivot_table(data, values='rating', index='userId', columns='movieId', fill_value=0)
    R = table.values
    
    pred, b, b_u, b_i, loss = mf_sgd(R,iterations=50)
    print("P x Q:")
    print(pred)
    print("Global bias:")
    print(b)
    print("User bias:")
    print(b_u)
    print("Item bias:")
    print(b_i)
    plot_training_loss(loss)
