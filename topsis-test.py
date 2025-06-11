import numpy as np 
from topsis import *

ex_opinion_matrix = np.array([
    [si,wi,si,si,ai,ei,ei,ei],
    [vsi,ei,vsi,ai,si,si,wi,si],
    [ai,ei,vsi,ai,si,ei,ei,vsi]
])

aggregated_opinion_matrix = get_aggregated_opinion_matrix(ex_opinion_matrix)
scored_opinion_matrix = get_scored_opinion_matrix(aggregated_opinion_matrix)
normalized_opinion_matrix = get_normalized_opinion_matrix(scored_opinion_matrix)

alternatives = np.array([
    [mg, g, vg, mg, b, vg, vb, vg],
    [vb, vg, g, b, mg, g, g, g],
    [mg, mg, mg, m, b, mg, g, mg],
    [g, mb, vg, mg, vg, vg, mg, vg],
    [vb, b, b, vg, vb, mg, mg, m],
    [g, mb, mb, vg, vg, vg, mg, g],
    [mb, vb, g, m, m, g, vb, mg],
    [vg, mg, vg, mg, vb, mg, g, vg],
    [vg, vb, vg, vg, mb, vb, mb, vg],
    [mb, b, vg, vg, vb, vg, mb, m],
    [vg, b, vg, vg, g, vg, vg, vg],
    [m, vg, mb, mb, mg, m, m, m],
    [g, b, mg, mg, vb, b, mg, g],
    [b, mb, vg, mb, mg, m, g, vg],
    [mb, vg, m, mb, mg, vg, mb, mb]
])

aggregated_alternatives = get_aggregated_alternatives(alternatives, expert_number=3)
scored_alternatives = get_scored_alternatives(aggregated_alternatives)
normalized_alternatives = get_normalized_alternatives(scored_alternatives)
weighted_alt_matrix = get_weighted_alternative_matrix(normalized_alternatives, normalized_opinion_matrix)

npis = calculate_npis(weighted_alt_matrix, [0, 1, 2, 3, 6, 7], [4, 5])
dp = calculate_dp(weighted_alt_matrix, npis)
nnis = calculate_nnis(weighted_alt_matrix, [0, 1, 2, 3, 6, 7], [4, 5])
dn = calculate_dn(weighted_alt_matrix, nnis)

prox_coeffs = calculate_proximity_coeffs(dp, dn)

# The index of the biggest number in prox_coeffs is the index of the best alternative
# For example, if the index of the biggest number is 2, that means alternative 3 is the best.

print(prox_coeffs)
