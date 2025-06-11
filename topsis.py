import numpy as np 

# Linguistic scale values for criteria importance
wi = [
    [0.2, 0.3, 0.2],
    [0.6, 0.7, 0.8],
    [0.45, 0.75, 0.75]
]

ei = [
    [0.4, 0.3, 0.25],
    [0.45, 0.55, 0.4],
    [0.45, 0.6, 0.55]
]

si = [
    [0.65, 0.55, 0.55],
    [0.4, 0.45, 0.55],
    [0.35, 0.4, 0.35]
]

vsi = [
    [0.8, 0.75, 0.7],
    [0.2, 0.15, 0.3],
    [0.15, 0.1, 0.2]
]

ai = [
    [0.9, 0.85, 0.95],
    [0.1, 0.15, 0.1],
    [0.05, 0.05, 0.1]
]

# Linguistic values for alternative performance
vb = np.array([
    [0.2, 0.2, 0.1],
    [0.65, 0.8, 0.85],
    [0.45, 0.8, 0.7]
])

b = np.array([
    [0.35, 0.35, 0.1],
    [0.5, 0.75, 0.8],
    [0.5, 0.75, 0.65]
])

mb = np.array([
    [0.5, 0.3, 0.5],
    [0.5, 0.35, 0.45],
    [0.45, 0.3, 0.6]
])

m = np.array([
    [0.4, 0.45, 0.5],
    [0.4, 0.45, 0.5],
    [0.35, 0.4, 0.45]
])

mg = np.array([
    [0.6, 0.45, 0.5],
    [0.2, 0.15, 0.25],
    [0.1, 0.25, 0.15]
])

g = np.array([
    [0.7, 0.75, 0.8],
    [0.15, 0.2, 0.25],
    [0.1, 0.15, 0.2]
])

vg = np.array([
    [0.95, 0.9, 0.95],
    [0.1, 0.1, 0.05],
    [0.05, 0.05, 0.05]
])

# T2NN addition (Equation 1)
def t2nn_add(u1, u2): 
    u1 = np.array(u1)
    u2 = np.array(u2)
    return np.array([
        [u1[0,0]+u2[0,0]-(u1[0,0]*u2[0,0]), u1[0,1]+u2[0,1]-(u1[0,1]*u2[0,1]), u1[0,2]+u2[0,2]-(u1[0,2]*u2[0,2])],
        [u1[1,0]*u2[1,0], u1[1,1]*u2[1,1], u1[1,2]*u2[1,2]],
        [u1[2,0]*u2[2,0], u1[2,1]*u2[2,1], u1[2,2]*u2[2,2]],
    ])

# Scalar multiplication (Equation 3)
def t2nn_scalar_mult(u, s):
    u = np.array(u)
    return np.array([
        [(1-(1-u[0,0])**s), (1-(1-u[0,1])**s), (1-(1-u[0,2])**s)],
        [((u[1,0])**s), ((u[1,1])**s), ((u[1,2])**s)],
        [((u[2,0])**s), ((u[2,1])**s), ((u[2,2])**s)]
    ])

# Scoring (Equation 5)
def score(u):
    u = np.array(u)
    return (1/12)*(8 + (u[0,0] + 2*u[0,1] + u[0,2]) - (u[1,0] + 2*u[1,1] + u[1,2]) - (u[2,0] + 2*u[2,1] + u[2,2]))

def get_aggregated_opinion_matrix(opinion_matrix):
    ex_number = opinion_matrix.shape[0]
    criteria_number = opinion_matrix.shape[1]

    def sum_t2nn_column(column):
        result = column[0]
        for number in column[1:]:
            result = t2nn_add(result, number)
        return result
    
    multiplied_matrix = np.zeros((ex_number, criteria_number, 3, 3))
    for i, expert in enumerate(opinion_matrix):
        multiplied_criteria = []
        for criteria in expert:
            multiplied_criteria.append(t2nn_scalar_mult(criteria, (1/3)))
        multiplied_matrix[i] = multiplied_criteria
    
    aggregated_opinion_matrix = np.zeros((criteria_number, 3, 3))

    for j in range(criteria_number):
        column = multiplied_matrix[:, j]
        aggregated_opinion_matrix[j] = sum_t2nn_column(column)
    
    return aggregated_opinion_matrix

def get_scored_opinion_matrix(aggregated_opinion_matrix):
    return np.array([score(u) for u in aggregated_opinion_matrix])

def get_normalized_opinion_matrix(scored_opinion_matrix):
    return scored_opinion_matrix / np.sum(np.sqrt(np.square(scored_opinion_matrix)))

def get_aggregated_alternatives(alternatives, expert_number):
    alt_number = alternatives.shape[0] // expert_number
    crit_number = alternatives.shape[1]
    aggregated_alternatives = np.zeros((alt_number, crit_number, 3, 3))

    for alt_idx in range(alt_number):
        for crit_idx in range(crit_number):
            agg = alternatives[alt_idx, crit_idx]
            for exper_idx in range(1, expert_number):
                agg = t2nn_add(agg, alternatives[alt_idx + (exper_idx * alt_number), crit_idx])
            agg = t2nn_scalar_mult(agg, (1/3))
            aggregated_alternatives[alt_idx, crit_idx] = agg
    return aggregated_alternatives

def get_scored_alternatives(aggregated_alternatives):
    alt_number = aggregated_alternatives.shape[0]
    crit_number = aggregated_alternatives.shape[1]
    scored_alternatives = np.zeros((alt_number, crit_number))
    for index, row in enumerate(aggregated_alternatives):
        scored_alternatives[index] = [score(u) for u in row]
    return scored_alternatives

def get_normalized_alternatives(scored_alternatives):
    alt_number = scored_alternatives.shape[0]
    crit_number = scored_alternatives.shape[1]

    normalized_alternatives = np.empty((alt_number, crit_number))
    for index, row in enumerate(scored_alternatives):
        normalized_alternatives[index] = row / np.sqrt(np.sum(np.square(row)))
    return normalized_alternatives

def get_weighted_alternative_matrix(normalized_alternatives, normalized_opinion_matrix):
    return normalized_alternatives * normalized_opinion_matrix

def calculate_npis(weighted_alternatives, benefit_indices, cost_indices):
    m, n = weighted_alternatives.shape
    npis = np.zeros(n)
    
    for p in benefit_indices:
        npis[p] = np.max(weighted_alternatives[:, p])
        
    for p in cost_indices:
        npis[p] = np.min(weighted_alternatives[:, p])
        
    return npis

def calculate_dp(weighted_alternative_matrix, npis):
    alt_count = weighted_alternative_matrix.shape[0]
    dp = np.zeros(alt_count)
    for i in range(alt_count):
        dp[i] = np.sqrt(np.sum(np.square(weighted_alternative_matrix[i] - npis)))
    return dp

def calculate_nnis(weighted_alternatives, benefit_indices, cost_indices):
    m, n = weighted_alternatives.shape
    nnis = np.zeros(n)
    
    for p in benefit_indices:
        nnis[p] = np.min(weighted_alternatives[:, p])
        
    for p in cost_indices:
        nnis[p] = np.max(weighted_alternatives[:, p])
        
    return nnis

def calculate_dn(weighted_alternative_matrix, nnis):
    alt_count = weighted_alternative_matrix.shape[0]
    dn = np.zeros(alt_count)
    for i in range(alt_count):
        dn[i] = np.sqrt(np.sum(np.square(weighted_alternative_matrix[i] - nnis)))
    return dn

def calculate_proximity_coeffs(dp, dn):
    prox_coeffs = np.zeros(dp.shape[0])
    for i in range(dp.shape[0]):
        prox_coeffs[i] = dn[i] / (dp[i] + dn[i])
    return prox_coeffs
