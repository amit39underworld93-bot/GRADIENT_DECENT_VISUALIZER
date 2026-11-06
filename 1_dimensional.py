import matplotlib.pyplot as plt 
import numpy as np
from matplotlib.animation import FuncAnimation
import sys

def GRADIENT_DESCENT_in_1D(f, x, h, alpha):

    n = int(input("Enter the number of iterations: "))
    if n <= 0:
        print("Value of n must be greater than zero!!!")
        sys.exit()

    tolerance = float(input("Enter the value of tolerance: "))
    if tolerance <= 0:
        print("Value of tolerance must be greater than zero!!!")
        sys.exit()

    value = f(x)
    converged = False

    x_history = [x]

    print("\n==== GRADIENT DESCENT ====")
    print(f"Initial x : {x} ")
    print(f"Initial f(x) : {value} ")

    for i in range(n):
        try:
            gradient = (f(x+h) - f(x-h))/(h*2)
        except (ValueError, OverflowError, ZeroDivisionError):
            print("\n==== GRADIENT DESCENT STOPPED ====")
            print("Function is not defined around the current x value!!!")
            break

        if abs(gradient) < tolerance:
            converged = True
            print("==== Convergence Achieved ====")
            break

        x_new = x - alpha*gradient

        if x_new < x_min:
            x_new = x_min
        elif x_new > x_max:
            x_new = x_max

        try:
            value_new = f(x_new)
        except (ValueError, OverflowError, ZeroDivisionError):
            print("\n==== GRADIENT DESCENT STOPPED ====")
            print(f"x_new = {x_new} is outside the valid domain of the function.")
            break

        print("\n ==== ITERATION DETAILS ====")
        print(f"Iteration: {i+1}")
        print(f"Current x: {x} ")
        print(f"Gradient: {gradient}")
        print(f"New x: {x_new}")
        print(f"New f(x): {value_new}")

        x = x_new
        value = value_new

        x_history.append(x)

    if not converged:
        final_gradient = (f(x + h) - f(x - h)) / (2 * h)

        if abs(final_gradient) < tolerance:
            converged = True

    print("\n========== FINAL RESULT ==========")
    print(f"Minimum point (approx): {x}")
    print(f"Minimum value (approx): {value}")
    print(f"Converged: {converged}")

    print("\n")

    if not converged:
        print("Maximum iterations reached or convergence not achieved.")

    return {
    "minimum_point" : x,
    "minimum_value" : value,
    "converged" : converged, 
    "x_history" : x_history
    }


# ===============================================================================================================================================================
# FUNCTION INPUT
# ===============================================================================================================================================================

expression = input("Enter the function f(x): ")

allowed = {
    "x" : None, 
    "y" : None, 
    "z" : None,
    "sin" : np.sin,
    "cos" : np.cos,
    "tan" : np.tan,
    "sqrt" : np.sqrt,
    "arcsin" : np.arcsin,
    "arccos" : np.arccos,
    "arctan" : np.arctan,
    "log" : np.log,
    "exp" : np.exp,
    "pi" : np.pi,
    "e" : np.e
}

def f(x):
    values = allowed.copy()
    values["x"] = x

    return eval(expression, {"__builtins__": {}}, values)

# ===============================================================================================================================================================
# RANGE VALIDATION
# ===============================================================================================================================================================

try:
    x_min = float(input("Enter the minimum value for x: "))
    x_max = float(input("Enter the maximum value for x: "))
except ValueError:
    print("Please enter a valid numeric value for x!!!")
    sys.exit()

if x_min >= x_max:
    print("x_min must be smaller than x-max!!!")
    sys.exit()

x = float(input("Enter the value of x: "))

if x < x_min or x > x_max:
    print("Initial x must lie within the given x range!!!")
    sys.exit()

h = float(input("Enter the value of h: "))
if h <= 0:
    print("Value of h must be greater then zero!!!")
    sys.exit()

alpha = float(input("Enter the learning rate: "))
if alpha <= 0:
    print("Value of learning rate must be greater then zero!!!")
    sys.exit()

result = GRADIENT_DESCENT_in_1D(f, x, h, alpha)
print(result)

x_history = result["x_history"]

x_values = np.linspace(x_min, x_max, 2000)
y_values = []

for value in x_values:
    try:
        y = f(value)
        if not np.isfinite(y):
            y = np.nan

    except (ValueError, OverflowError, ZeroDivisionError):
        y = np.nan

    y_values.append(y)

y_values = np.array(y_values, dtype=float)

difference = np.abs(np.diff(y_values))

finite_difference = difference[np.isfinite(difference)]

if len(finite_difference) > 0:
    median_difference = np.median(finite_difference)
else:
    median_difference = 0

if median_difference > 0:
    threshold = 100*median_difference

    jump = difference > threshold

    jump_indices = np.where(jump)[0]

    y_values[jump_indices] = np.nan
    y_values[jump_indices + 1] = np.nan

# ===============================================================================================================================================================
# 1D FIGURE
# ===============================================================================================================================================================

fig, ax = plt.subplots()

ax.plot(x_values, y_values, label = f"{expression}")

ax.axhline(0, color = "black")
ax.axvline(0, color = "black")

ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")
ax.set_title(f"GRADIENT DESCENT VISUALIZER OF f(x) = {expression}")

ax.legend()
ax.grid(linestyle = "--")

point = ax.scatter([x_history[0]], [f(x_history[0])], color = "red")

# ===============================================================================================================================================================
# ANIMATION
# ===============================================================================================================================================================

def update(frame):

    current_x = x_history[frame]
    current_y = f(current_x)

    point.set_offsets([[current_x, current_y]])
    return point, 

animation = FuncAnimation(fig, update, frames = len(x_history), interval = 300, repeat = False)

plt.show()