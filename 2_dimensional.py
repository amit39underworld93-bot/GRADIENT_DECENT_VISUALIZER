import matplotlib.pyplot as plt 
import numpy as np
from matplotlib.animation import FuncAnimation
import sys

def GRADIENT_DESCENT_in_2D(f, x, y, h, alpha):

    n = int(input("Enter the number of iterations: "))
    if n <= 0:
        print("Value of n must be greater than zero!!!")
        sys.exit()

    tolerance_x = float(input("Enter the value of tolerance of x: "))
    if tolerance_x <= 0:
        print("Value of tolerance must be greater than zero!!!")
        sys.exit()

    tolerance_y = float(input("Enter the value of tolerance of y: "))
    if tolerance_y <= 0:
        print("Value of tolerance must be greater than zero!!!")
        sys.exit()

    value = f(x, y)
    converged = False

    x_history = [x]
    y_history = [y]

    print("\n==== GRADIENT DESCENT ====")
    print(f"Initial x : {x} ")
    print(f"Initial y : {y} ")
    print(f"Initial f(x, y) : {value} ")

    for i in range(n):
        gradient_x = (f(x+h, y) - f(x-h, y))/(2*h)
        gradient_y = (f(x, y+h) - f(x, y-h))/(2*h)

        if abs(gradient_x) < tolerance_x and abs(gradient_y) < tolerance_y:
            converged = True
            print("==== Convergence Achieved ====")
            break

        x_new = x - alpha*gradient_x
        y_new = y - alpha*gradient_y    

        if x_new < x_min:
            x_new = x_min
        elif x_new > x_max:
            x_new = x_max

        if y_new < y_min:
            y_new = y_min
        elif y_new > y_max:
            y_new = y_max

        try:
            value_new = f(x_new, y_new)
        except (ValueError, OverflowError, ZeroDivisionError):
            print("==== GRADIENT DESCENT STOPPED ====")
            print(f"x_new = {x_new} is outside the valid domain of the function.")
            sys.exit()

        print("\n ==== ITERATION DETAILS ====")
        print(f"Iteration: {i+1}")
        print(f"Current x: {x} ")
        print(f"Current y: {y} " )
        print(f"Gradient w.r.t x: {gradient_x}")
        print(f"Gradient w.r.t y: {gradient_y}")
        print(f"New x: {x_new}")
        print(f"New y: {y_new}")
        print(f"New f(x, y): {value_new}")

        x = x_new
        y = y_new
        value = value_new

        x_history.append(x)
        y_history.append(y)

    if not converged:
        final_gradient_x = (f(x+h, y) - f(x-h, y))/(2*h)
        final_gradient_y = (f(x, y+h) - f(x, y-h))/(2*h)

        if abs(final_gradient_x) < tolerance_x and abs(final_gradient_y) < tolerance_y:
            converged = True

    print("\n========== FINAL RESULT ==========")
    print(f"Approximate final point: ({x}, {y})")
    print(f"Function value at final point: {value}")
    print(f"Converged: {converged}")

    print("\n")

    if not converged:
        print("Maximum iterations reached or convergence not achieved.")

    return {
    "minimum_point" : (x, y),
    "minimum_value" : value,
    "converged" : converged, 
    "x_history" : x_history, 
    "y_history" : y_history
    }


# ===============================================================================================================================================================
# FUNCTION INPUT
# ===============================================================================================================================================================

expression = input("Enter the function f(x, y): ")

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

def f(x ,y):
    values = allowed.copy()
    values["x"] = x
    values["y"] = y

    with np.errstate(all="ignore"):
        result =  eval(expression, {"__builtins__": {}}, values)
        if isinstance(result, (int, float)):
            result = result*np.ones_like(x)

    return result

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

try:
    y_min = float(input("Enter the minimum value for y: "))
    y_max = float(input("Enter the maximum value for y: "))
except ValueError:
    print("Please enter a valid numeric value for y!!!")
    sys.exit()

if y_min >= y_max:
    print("y_min must be smaller than y_max!!!")
    sys.exit()

x = float(input("Enter the value of x: "))

if x < x_min or x > x_max:
    print("Initial x must lie within the given x range!!!")
    sys.exit()

y = float(input("Enter the value of y: "))

if y < y_min or y > y_max:
    print("Initial y must lie within the given y range!!!")
    sys.exit()

h = float(input("Enter the value of h: "))
if h <= 0:
    print("Value of h must be greater than zero!!!")
    sys.exit()

alpha = float(input("Enter the learning rate: "))
if alpha <= 0:
    print("Value of learning rate must be greater than zero!!!")
    sys.exit()

result = GRADIENT_DESCENT_in_2D(f, x, y, h, alpha)
print(result)

x_history = result["x_history"]
y_history = result["y_history"]

x_values = np.linspace(x_min, x_max, 400)
y_values = np.linspace(y_min, y_max, 400)

X, Y = np.meshgrid(x_values, y_values)

try:
    F = f(X, Y)
except Exception:
    print("Please enter a valid function!!!")
    sys.exit()

# ===============================================================================================================================================================
# 2D FIGURE  
# ===============================================================================================================================================================

fig, ax = plt.subplots()

contour = ax.contour(X, Y, F, levels = 20, cmap = "viridis")
ax.axis("equal")

ax.axvline(0, color = "black")
ax.axhline(0, color = "black")

ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")
ax.set_title(f"GRADIENT DESCENT VISUALIZER OF f(x, y) = {expression}")

ax.grid(linestyle = "--")

point = ax.scatter([x_history[0]], [y_history[0]], color = "red")

# ===============================================================================================================================================================
# ANIMATION
# ===============================================================================================================================================================

def update(frame):
    current_x = x_history[frame]
    current_y = y_history[frame]

    point.set_offsets([[current_x, current_y]])
    return point, 

animation = FuncAnimation(fig, update, frames = len(x_history), interval = 100, repeat = False)

plt.show()