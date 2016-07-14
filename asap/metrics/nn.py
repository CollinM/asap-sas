import theano.tensor as T


# FYI: DOES NOT WORK PROPERLY WITH KERAS!
def quadratic_weighted_kappa_loss(y_true, y_pred):
    min_rating = T.minimum(T.min(y_true), T.min(y_pred))
    max_rating = T.maximum(T.max(y_true), T.max(y_pred))

    hist_true = T.bincount(y_true, minlength=max_rating)
    hist_pred = T.bincount(y_pred, minlength=max_rating)
    num_ratings = (max_rating - min_rating) + 1
    num_scored = float(len(y_true))

    numerator = T.zeros(1)
    denominator = T.zeros(1)
    z = T.zeros(len(y_true))
    for i_true in range(min_rating, max_rating + 1):
        for j_pred in range(min_rating, max_rating + 1):
            expected = T.true_div(T.mul(hist_true[i_true], hist_pred[j_pred]), num_scored)
            d = T.true_div(T.sqr(i_true - j_pred), T.sqr(num_ratings - 1.))
            conf_mat_cell = T.sum(T.and_(T.eq(T.sub(y_true, i_true), z), T.eq(T.sub(y_pred, j_pred), z)))
            numerator = T.add(numerator, T.true_div(T.mul(d, conf_mat_cell), num_scored))
            denominator = T.add(denominator, T.true_div(T.mul(d, expected), num_scored))

    return T.true_div(numerator, denominator)
