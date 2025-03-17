#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting

def potential_energy(x, y):
    """
    Potencjał 'meksykański kapelusz':
    E(x, y) = -(r^2 - 1)^2, gdzie r^2 = x^2 + y^2.
    
    Uwaga: Znak minus sprawia, że środek jest maksimum,
    a minimum znajduje się na okręgu o promieniu r=1.
    """
    r2 = x**2 + y**2
    return -(r2 - 1)**2

def force(position):
    """
    Siła działająca na cząstkę w potencjale.
    F = -∇E(x,y)
    
    Dla potencjału meksykańskiego kapelusza:
    F_x = -4x(r^2-1) * (-1) = 4x(r^2-1)
    F_y = -4y(r^2-1) * (-1) = 4y(r^2-1)
    """
    x, y = position
    r2 = x**2 + y**2
    
    # Pochodna potencjału po x i y (z uwzględnieniem znaku minus w potencjale)
    fx = 4 * x * (r2 - 1)
    fy = 4 * y * (r2 - 1)
    
    return np.array([fx, fy])

def simulate_trajectory(initial_position, initial_velocity, num_steps=1000, dt=0.005, damping=0.02):
    """
    Symulacja ruchu cząstki w potencjale z tłumieniem.
    Używamy metody Runge-Kutta 4. rzędu dla lepszej dokładności.
    """
    positions = np.zeros((num_steps, 2))
    velocities = np.zeros((num_steps, 2))
    
    positions[0] = initial_position
    velocities[0] = initial_velocity
    
    for i in range(1, num_steps):
        # Implementacja metody Runge-Kutta 4. rzędu
        pos = positions[i-1]
        vel = velocities[i-1]
        
        # k1
        k1_v = force(pos)
        k1_x = vel
        
        # k2
        k2_v = force(pos + 0.5 * dt * k1_x)
        k2_x = vel + 0.5 * dt * k1_v - 0.5 * dt * damping * vel
        
        # k3
        k3_v = force(pos + 0.5 * dt * k2_x)
        k3_x = vel + 0.5 * dt * k2_v - 0.5 * dt * damping * vel
        
        # k4
        k4_v = force(pos + dt * k3_x)
        k4_x = vel + dt * k3_v - dt * damping * vel
        
        # Aktualizacja pozycji i prędkości
        positions[i] = pos + (dt/6) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
        velocities[i] = vel + (dt/6) * (k1_v + 2*k2_v + 2*k3_v + k4_v) - dt * damping * vel
    
    return positions, velocities

def main():
    # Zakres wartości w obu osiach
    x_vals = np.linspace(-2, 2, 100)  # Zmniejszona rozdzielczość dla szybszego renderowania 3D
    y_vals = np.linspace(-2, 2, 100)

    # Tworzymy siatkę punktów (X, Y) oraz obliczamy E(X, Y)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = potential_energy(X, Y)

    # Symulujemy trajektorię cząstki
    # Startujemy z punktu blisko środka (maksimum potencjału) z niewielką prędkością początkową
    initial_position = np.array([0.2, 0.0])
    initial_velocity = np.array([0.0, 0.3])
    positions, velocities = simulate_trajectory(
        initial_position, 
        initial_velocity, 
        num_steps=5000,  # Więcej kroków dla płynniejszej symulacji
        dt=0.005,        # Mniejszy krok czasowy dla lepszej dokładności
        damping=0.01     # Mniejsze tłumienie
    )

    # Tworzymy figurę z dwoma wykresami: 3D i oscylacje w czasie
    plt.ion()  # Włączamy tryb interaktywny
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)

    # 1. Powierzchnia energii potencjalnej 3D z trajektorią
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5, label='Energia potencjalna E(x, y)')

    # Obliczamy wartości Z dla trajektorii
    trajectory_z = np.array([potential_energy(x, y) for x, y in positions])

    # Rysujemy trajektorię w 3D
    ax1.plot(positions[:, 0], positions[:, 1], trajectory_z, 'r-', linewidth=2, label='Trajektoria')
    ax1.plot([positions[0, 0]], [positions[0, 1]], [trajectory_z[0]], 'go', markersize=6, label='Start')

    # Etykiety osi
    ax1.set_title('Trajektoria cząstki w potencjale 3D')
    ax1.set_xlabel('Koordynat x')
    ax1.set_ylabel('Koordynat y')
    ax1.set_zlabel('Energia potencjalna E(x, y)')
    ax1.legend()

    # Ustawiamy limity osi
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_zlim(np.min(Z), np.max(Z))

    # 2. Wykres pozycji w funkcji czasu (oscylacje)
    time = np.arange(len(positions)) * 0.005  # Dostosuj do dt
    ax2.plot(time, positions[:, 0], 'b-', label='x(t)')
    ax2.plot(time, positions[:, 1], 'r-', label='y(t)')
    ax2.set_title('Oscylacje współrzędnych w czasie')
    ax2.set_xlabel('Czas')
    ax2.set_ylabel('Pozycja')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    
    # Interaktywna animacja 3D
    print("Interaktywna wizualizacja 3D - możesz obracać wykres za pomocą myszy")
    print("Naciśnij 'q' aby zakończyć")
    
    # Funkcja do obsługi klawiszy
    def on_key(event):
        if event.key == 'q':
            plt.close(fig)
    
    # Podłączamy funkcję obsługi klawiszy
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Pokazujemy wykres
    plt.show(block=True)  # block=True zatrzymuje wykonanie programu do zamknięcia okna

if __name__ == "__main__":
    main()
