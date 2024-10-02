def smoothing_function(r, r_s, r_c):
    if r <= r_s:
        return 1.0
    elif r_s < r < r_c:
        x = (r_c - r) / (r_c - r_s)
        return x**3 * (3 - 2 * x)
    else:
        return 0.0

def compute_smoothed_force(r, r_s, r_c):
    S = smoothing_function(r, r_s, r_c)
    F_original = 24 * epsilon / r * (2 * (sigma / r)**12 - (sigma / r)**6)
    return F_original * S
