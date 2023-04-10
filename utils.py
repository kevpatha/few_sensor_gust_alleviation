import numpy as np
import matplotlib.pyplot as plt



def plot_learning_curve(x, scores, figure_file, avg_of=100):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-avg_of):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)
    
    
def plot_test_curve(x, scores, figure_file, avg_of=10):
    running_avg = np.zeros(len(scores))
    goal = np.ones(len(scores))*np.mean(scores[:49])
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-avg_of):(i+1)])
    plt.plot(x, running_avg-goal)
    plt.plot(x, goal-goal, 'r--')
    plt.title('Running average of previous 10 lift')
    plt.ylim(-0.15, 0.15)
    plt.savefig(figure_file)