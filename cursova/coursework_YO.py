import numpy as np
import matplotlib.pyplot as plt

# Лічильник викликів функції
function_calls = 0
trajectory = []

def sven_algorithm(f, x0, delta=0.01):
    """
    Алгоритм Свена для знаходження інтервалу, що містить мінімум.
    """
    x1 = x0 - delta
    x2 = x0
    x3 = x0 + delta

    f1 = f(x1)
    f2 = f(x2)
    f3 = f(x3)
    
    if f1 >= f2 <= f3:
        return x1, x3
    
    elif f1 <= f2 >= f3:
        return x1 - delta, x3 + delta
    
    elif f1 >= f2 >= f3:
        while True:
            x1 = x2
            x2 = x3
            delta *= 2
            x3 = x3 + delta
            f1 = f2
            f2 = f3
            f3 = f(x3)
            if f2 <= f3:
                break
        return x1, x3
    
    else:
        while True:
            x3 = x2
            x2 = x1
            delta *= 2
            x1 = x1 - delta
            f3 = f2
            f2 = f1
            f1 = f(x1)
            if f2 <= f1:
                break
        return x1, x3

def dsk_powell_search(f, a, b, tol=0.001):
    while abs(b - a) > tol:
        u = (a + b) / 2
        fu = f(u)
        fa = f(a)
        fb = f(b)
        if fu < fa and fu < fb:
            b = u
        elif fa < fb:
            b = u
        else:
            a = u
    return (a + b) / 2

def hooke_jeeves(f, x0, step_size=0.001, alpha=2, epsilon=0.0001, max_iter=1000, constraints=None):
    global function_calls
    global trajectory

    def explore(x, base_step):
        global function_calls

        for i in range(len(x)):
            def objective_step_size(step):
                P = np.copy(x)
                P[i] += step
                if constraints and not constraints(P):
                    return float('inf')  # Повертаємо нескінченність, якщо точка не задовольняє обмеженням
                return f(P)
            
            # Використовуємо алгоритм Свена для знаходження інтервалу
            a, b = sven_algorithm(objective_step_size, 0, delta=0.01)
            step = dsk_powell_search(objective_step_size, a, b)

            P = np.copy(x)
            P[i] += step
            if constraints and not constraints(P):
                P[i] -= step  # Скасовуємо крок, якщо нова точка не задовольняє обмеженням
            function_calls += 1
            if f(P) < f(x):
                x = np.copy(P)
            else:
                P[i] -= 2 * step
                if constraints and not constraints(P):
                    P[i] += step  # Скасовуємо крок, якщо нова точка не задовольняє обмеженням
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
        print(f"Iteration {iteration}: x = {x_best}, f(x) = {f(x_best)}")

    return x_best, f(x_best)

def objective_function(x):
    global function_calls
    function_calls += 1
    return ((10 * (x[0] - x[1])**2 + (x[0] - 1)**2)**(1/4))

# Допустима область, яка включає точку мінімуму
def constraint_including_minimum(x):
    return (x[0] + x[1]) >= -2 

# Початкова точка
x0 = np.array([-1.2, 0.0])

# Використання методу Гука-Дживса з алгоритмом Свена і ДСК-Пауелла
function_calls = 0
trajectory = []
solution_dsk, value_dsk = hooke_jeeves(objective_function, x0, constraints=constraint_including_minimum)
trajectory_dsk = np.array(trajectory)

print("Solution (DSK-Powell):", solution_dsk)
print("Objective function value (DSK-Powell):", value_dsk)
print("Number of function calls (DSK-Powell):", function_calls)

# Візуалізація
plt.plot(trajectory_dsk[:, 0], trajectory_dsk[:, 1], 'o-', label='Trajectory (DSK-Powell)')
plt.plot(solution_dsk[0], solution_dsk[1], 'bx', markersize=10, label='Solution (DSK-Powell)')
plt.plot(x0[0], x0[1], 'go', markersize=10, label='Start Point')
plt.xlabel('x[0]')
plt.ylabel('x[1]')
plt.title('Trajectory of Hooke-Jeeves Optimization with DSK-Powell Search Method')
plt.legend()
plt.grid(True)
plt.show()