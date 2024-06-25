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

# Допустимі області в залежності від виду допустимої області

# Випукла допустима область
def convex_constraint(x):
    return (x[0] - 0.5)**2 + (x[1] - 0.5)**2 <= 5

# Невипукла допустима область
def non_convex_constraint(x):
    return ((x[0] - 1)**2 + (x[1] - 1)**2 <= 2.56) or ((x[0] + 1.2)**2 + x[1]**2 <= 1.44)

# Лінійні обмеження
def linear_constraints(x):
    return (x[0] + 2*x[1] >= -2.4) and (x[0] - 2*x[1] <= 1.2)


# Випадок 1: Випукла область
function_calls = 0
trajectory = []
solution_convex, value_convex = hooke_jeeves(objective_function, x0, constraints=convex_constraint)
trajectory_convex = np.array(trajectory)

print("Solution (Convex):", solution_convex)
print("Objective function value (Convex):", value_convex)
print("Number of function calls (Convex):", function_calls)

# Випадок 2: Невипукла область
function_calls = 0
trajectory = []
solution_non_convex, value_non_convex = hooke_jeeves(objective_function, x0, constraints=non_convex_constraint)
trajectory_non_convex = np.array(trajectory)

print("Solution (Non-Convex):", solution_non_convex)
print("Objective function value (Non-Convex):", value_non_convex)
print("Number of function calls (Non-Convex):", function_calls)

# Випадок 3: Лінійні обмеження
function_calls = 0
trajectory = []
solution_linear, value_linear = hooke_jeeves(objective_function, x0, constraints=linear_constraints)
trajectory_linear = np.array(trajectory)

print("Solution (Linear Constraints):", solution_linear)
print("Objective function value (Linear Constraints):", value_linear)
print("Number of function calls (Linear Constraints):", function_calls)

# Візуалізація результатів
plt.figure(figsize=(14, 7))

# Графік для випуклої області
plt.subplot(1, 3, 1)
plt.plot(trajectory_convex[:, 0], trajectory_convex[:, 1], 'o-', label='Trajectory (Convex)')
plt.plot(solution_convex[0], solution_convex[1], 'rx', markersize=10, label='Solution (Convex)')
plt.plot(x0[0], x0[1], 'go', markersize=10, label='Start Point')
theta = np.linspace(0, 2*np.pi, 100)
x_circle = 0.5 + np.sqrt(5) * np.cos(theta)
y_circle = 0.5 + np.sqrt(5) * np.sin(theta)
plt.plot(x_circle, y_circle, 'b--', label='Convex Constraint')
plt.xlabel('x[0]')
plt.ylabel('x[1]')
plt.title('Convex Constraint')
plt.legend()
plt.grid(True)

# Графік для невипуклої області
plt.subplot(1, 3, 2)
plt.plot(trajectory_non_convex[:, 0], trajectory_non_convex[:, 1], 'o-', label='Trajectory (Non-Convex)')
plt.plot(solution_non_convex[0], solution_non_convex[1], 'rx', markersize=10, label='Solution (Non-Convex)')
plt.plot(x0[0], x0[1], 'go', markersize=10, label='Start Point')
x_inner_circle_1 = 1 + np.sqrt(2.56) * np.cos(theta)
y_inner_circle_1 = 1 + np.sqrt(2.56) * np.sin(theta)
x_outer_circle_2 = -1.2 + np.sqrt(1.44) * np.cos(theta)
y_outer_circle_2 = np.sqrt(1.44) * np.sin(theta)
plt.plot(x_inner_circle_1, y_inner_circle_1, 'r--', label='Non-Convex Constraint (1)')
plt.plot(x_outer_circle_2, y_outer_circle_2, 'b--', label='Non-Convex Constraint (2)')
plt.xlabel('x[0]')
plt.ylabel('x[1]')
plt.title('Non-Convex Constraint')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(trajectory_linear[:, 0], trajectory_linear[:, 1], 'o-', label='Trajectory (Linear)')
plt.plot(solution_linear[0], solution_linear[1], 'rx', markersize=10, label='Solution (Linear)')
plt.plot(x0[0], x0[1], 'go', markersize=10, label='Start Point')
x_line1 = np.linspace(-2, 2, 100)
y_line1 = (-2.4 - x_line1) / 2
x_line2 = np.linspace(-2, 2, 100)
y_line2 = (x_line2 + 1.2) / 2
plt.plot(x_line1, y_line1, 'b--', label='Linear Constraint 1')
plt.plot(x_line2, y_line2, 'r--', label='Linear Constraint 2')
plt.xlabel('x[0]')
plt.ylabel('x[1]')
plt.title('Linear Constraints')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()