import numpy as np
import matplotlib.pyplot as plt

# Лічильник викликів функції
function_calls = 0
trajectory = []

def hooke_jeeves(f, x0, step_size=0.001, alpha=2, epsilon=0.0001, max_iter=100000):
    global function_calls
    global trajectory

    def explore(x, base_step):
        global function_calls
        for i in range(len(x)):
            P = np.copy(x)
            P[i] += base_step
            function_calls += 1
            if f(P) < f(x):
                x = np.copy(P)
            else:
                P[i] -= 2 * base_step
                function_calls += 1
                if f(P) < f(x):
                    x = np.copy(P)
        return x

    x_best = np.copy(x0)
    base_step = step_size
    iteration = 0
    trajectory.append(np.copy(x_best))  # Додаємо початкову точку до траєкторії

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
                x_best = np.copy(x_new)
                x_new = explore(x_temp, base_step)
                if f(x_new) >= f(x_best):
                    break

        trajectory.append(np.copy(x_best))
        iteration += 1
        print(f"Iteration {iteration}: x = {x_best}, f(x) = {f(x_best)}")

    return x_best, f(x_best)

def objective_function(x):
    global function_calls
    function_calls += 1
    return ((10 * (x[0] - x[1])**2 + (x[0] - 1)**2)**(1/4))

# Initial point
x0 = np.array([-1.2, 0.0])

# Reset function calls
function_calls = 0
trajectory = []

# Run Hooke-Jeeves method
solution, value = hooke_jeeves(objective_function, x0)

print("Solution:", solution)
print("Objective function value:", value)
print("Number of function calls:", function_calls)

# # Plot trajectory
# trajectory = np.array(trajectory)
# plt.plot(trajectory[:, 0], trajectory[:, 1], 'o-', label='Trajectory')
# plt.plot(solution[0], solution[1], 'rx', markersize=10, label='Solution')
# plt.plot(x0[0], x0[1], 'go', markersize=10, label='Start')  # Початкова точка зеленим
# plt.xlabel('x[0]')
# plt.ylabel('x[1]')
# plt.title('Trajectory of Hooke-Jeeves Optimization')
# plt.legend()
# plt.grid(True)
# plt.show()
