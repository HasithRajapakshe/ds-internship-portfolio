def bayes_theorem(p_a, p_b_given_a, p_b_given_not_a):

    p_not_a = 1 - p_a
    p_b = (p_b_given_a * p_a) + (p_b_given_not_a * p_not_a)

    p_a_given_b = (p_b_given_a * p_a) / p_b
    return p_a_given_b


result = bayes_theorem(0.01, 0.99, 0.01)
print(f"Probability of having the disease: {result:.2%}")
