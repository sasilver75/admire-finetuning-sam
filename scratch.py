from scipy.stats import spearmanr

pred_ranks = [1, 2, 3, 4, 5]
true_ranks = [5, 4, 3, 2, 1]
# true_ranks = [1, 2, 3, 4, 5]

correlation = spearmanr(pred_ranks, true_ranks).correlation
print(correlation)
