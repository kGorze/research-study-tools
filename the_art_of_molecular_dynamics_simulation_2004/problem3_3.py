def predictor_with_jerk(x, v, a, j, dt):
    x_p = x + v * dt + 0.5 * a * dt**2 + (1/6) * j * dt**3
    v_p = v + a * dt + 0.5 * j * dt**2
    a_p = a + j * dt
    return x_p, v_p, a_p
