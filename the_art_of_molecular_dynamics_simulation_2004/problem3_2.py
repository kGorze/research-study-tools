def predictor_corrector_k4(x, v, a, dt, t):
    # Predictor step
    x_p = x + v * dt + 0.5 * a * dt**2
    v_p = v + a * dt
    # Corrector step (compute new acceleration)
    a_p = -omega**2 * x_p
    v_c = v + 0.5 * (a + a_p) * dt
    x_c = x + 0.5 * (v + v_c) * dt
    return x_c, v_c, a_p


def predictor_corrector_k5(x_hist, v_hist, a_hist, dt, t):
    # Predictor step using Adams-Bashforth coefficients for 5th order
    x_p = x_hist[-1] + dt * (1901*a_hist[-1] - 2774*a_hist[-2] + 2616*a_hist[-3] - 1274*a_hist[-4] + 251*a_hist[-5]) / 720
    v_p = v_hist[-1] + dt * (1901*a_hist[-1] - 2774*a_hist[-2] + 2616*a_hist[-3] - 1274*a_hist[-4] + 251*a_hist[-5]) / 720
    # Corrector step using Adams-Moulton coefficients for 5th order
    a_p = -omega**2 * x_p
    x_c = x_hist[-1] + dt * (251*a_p + 646*a_hist[-1] - 264*a_hist[-2] + 106*a_hist[-3] - 19*a_hist[-4]) / 720
    v_c = v_hist[-1] + dt * (251*a_p + 646*a_hist[-1] - 264*a_hist[-2] + 106*a_hist[-3] - 19*a_hist[-4]) / 720
    return x_c, v_c, a_p
