import numpy as np
import matplotlib.pyplot as plt

# Лічильник викликів функції
function_calls = 0
trajectory = []

def hooke_jeeves(f, x0, step_size=0.001, alpha=2, epsilon=0.0001, max_iter=1000, constraints=None):
    global function_calls
    global trajectory

    def explore(x, base_step):
        global function_calls
        for i in range(len(x)):
            P = np.copy(x)
            P[i] += base_step
            if constraints and not constraints(P):
                P[i] -= base_step  # Скасовуємо крок, якщо нова точка не задовольняє обмеженням
            function_calls += 1
            if f(P) < f(x):
                x = np.copy(P)
            else:
                P[i] -= 2 * base_step
                if constraints and not constraints(P):
                    P[i] += base_step  # Скасовуємо крок, якщо нова точка не задовольняє обмеженням
                function_calls += 1
                if f(P) < f(x):
                    x = np.copy(P)
        return x

    x_best = np.copy(x0)
    base_step = step_size
    iteration = 0
    trajectory.append(np.copy(x_best))

    while iteration < max_iter:
        x_new = explore(x_best, base_step)

        # Критерій закінчення (перший критерій)
        if (np.linalg.norm(x_new - x_best) / np.linalg.norm(x_best) <= epsilon and
            abs(f(x_new) - f(x_best)) <= epsilon): 
            break

        if np.array_equal(x_new, x_best):
            base_step /= alpha
        else:
            while True:
                x_temp = x_new + (x_new - x_best)
                if constraints and not constraints(x_temp):
                    break  # Скасовуємо крок, якщо нова точка не задовольняє обмеженням
                x_best = np.copy(x_new)
                x_new = explore(x_temp, base_step)
                if f(x_new) >= f(x_best):
                    break

        trajectory.append(np.copy(x_best))
        iteration += 1
        # print(f"Iteration {iteration}: x = {x_best}, f(x) = {f(x_best)}")

    return x_best, f(x_best)

def objective_function(x):
    global function_calls
    function_calls += 1
    return ((10 * (x[0] - x[1])**2 + (x[0] - 1)**2)**(1/4))

# Початкова точка
x0 = np.array([-1.2, 0.0])

# Допустимі області в залежності від розташування локального мінімума

# Допустима область, яка включає точку мінімуму
def constraint_including_minimum(x):
    return (x[0] + x[1]) >= -2 

# Допустима область, яка не включає точку мінімуму
def constraint_excluding_minimum(x):
    return (x[0] + x[1]) <= 1

# Випадок 1: Допустима область включає точку мінімуму
function_calls = 0
trajectory = []
solution_including, value_including = hooke_jeeves(objective_function, x0, constraints=constraint_including_minimum)
trajectory_including = np.array(trajectory)
print("Solution (Including Minimum):", solution_including)
print("Objective function value (Including Minimum):", value_including)
print("Number of function calls (Including Minimum):", function_calls)

# Випадок 2: Допустима область не включає точку мінімуму
function_calls = 0
trajectory = []
solution_excluding, value_excluding = hooke_jeeves(objective_function, x0, constraints=constraint_excluding_minimum)
trajectory_excluding = np.array(trajectory)
print("Solution (Excluding Minimum):", solution_excluding)
print("Objective function value (Excluding Minimum):", value_excluding)
print("Number of function calls (Excluding Minimum):", function_calls)

# Візуалізація
plt.plot(trajectory_including[:, 0], trajectory_including[:, 1], 'o-', label='Trajectory (Including Minimum)')
plt.plot(solution_including[0], solution_including[1], 'rx', markersize=10, label='Solution (Including Minimum)')
plt.plot(x0[0], x0[1], 'go', markersize=10, label='Start Point')

plt.plot(trajectory_excluding[:, 0], trajectory_excluding[:, 1], 'o-', label='Trajectory (Excluding Minimum)')
plt.plot(solution_excluding[0], solution_excluding[1], 'bx', markersize=10, label='Solution (Excluding Minimum)')
plt.xlabel('x[0]')
plt.ylabel('x[1]')
plt.title('Trajectory of Hooke-Jeeves Optimization with Constraints')
plt.legend()
plt.grid(True)
plt.show()