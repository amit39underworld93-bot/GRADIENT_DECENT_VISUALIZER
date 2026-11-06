import matplotlib.pyplot as plt 
import numpy as np
from matplotlib.animation import FuncAnimation
import sys

def GRADIENT_DESCENT_in_3D(f, x, y, z, h, alpha):

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

    tolerance_z = float(input("Enter the value of tolerance of z: "))
    if tolerance_z <= 0:
        print("Value of tolerance must be greater than zero!!!")
        sys.exit()

    value = f(x, y, z)
    converged = False

    x_history = [x]
    y_history = [y]
    z_history = [z]

    print("\n==== GRADIENT DESCENT ====")
    print(f"Initial x : {x} ")
    print(f"Initial y : {y} ")
    print(f"initial z : {z} ")
    print(f"Initial f(x, y, z) : {value} ")

    for i in range(n):
        gradient_x = (f(x+h, y, z) - f(x-h, y, z))/(2*h)
        gradient_y = (f(x, y+h, z) - f(x, y-h, z))/(2*h)
        gradient_z = (f(x, y, z+h) - f(x, y, z-h))/(2*h)

        if abs(gradient_x) < tolerance_x and abs(gradient_y) < tolerance_y and abs(gradient_z) < tolerance_z:
            converged = True
            print("==== Convergence Achieved ====")
            break

        x_new = x - alpha*gradient_x
        y_new = y - alpha*gradient_y    
        z_new = z - alpha*gradient_z

        if x_new < x_min:
            x_new = x_min
        elif x_new > x_max:
            x_new = x_max

        if y_new < y_min:
            y_new = y_min
        elif y_new > y_max:
            y_new = y_max

        if z_new < z_min:
            z_new = z_min
        elif z_new > z_max:
            z_new = z_max

        try:
            value_new = f(x_new, y_new, z_new)
        except (ValueError, ZeroDivisionError ,OverflowError):
            print("==== GRADIENT DESCENT STOPPED ====")
            print(f"New point ({x_new}, {y_new}, {z_new}) is outside the valid domain of the given function!!!")
            sys.exit()

        print("\n ==== ITERATION DETAILS ====")
        print(f"Iteration: {i+1}")
        print(f"Current x: {x} ")
        print(f"Current y: {y} ")
        print(f"Current z: {z} ")
        print(f"Gradient w.r.t x: {gradient_x}")
        print(f"Gradient w.r.t y: {gradient_y}")
        print(f"Gradient w.r.t z: {gradient_z}")
        print(f"New x: {x_new}")
        print(f"New y: {y_new}")
        print(f"New z: {z_new}")
        print(f"New f(x, y, z): {value_new}")

        x = x_new
        y = y_new
        z = z_new
        value = value_new

        x_history.append(x)
        y_history.append(y)
        z_history.append(z)

    if not converged:
        final_gradient_x = (f(x+h, y, z) - f(x-h, y, z))/(2*h)
        final_gradient_y = (f(x, y+h, z) - f(x, y-h, z))/(2*h)
        final_gradient_z = (f(x, y, z+h) - f(x, y, z-h))/(2*h)

        if abs(final_gradient_x) < tolerance_x and abs(final_gradient_y) < tolerance_y and abs(final_gradient_z) < tolerance_z:
            converged = True

    print("\n========== FINAL RESULT ==========")
    print(f"Approximate final point: ({x}, {y}, {z})")
    print(f"Function value at final point: {value}")
    print(f"Converged: {converged}")

    print("\n")

    if not converged:
        print("Maximum iterations reached or convergence not achieved.")

    return {
    "minimum_point": (x, y, z),
    "minimum_value": value,
    "x_history" : x_history, 
    "y_history" : y_history,
    "z_history" : z_history,
    "converged": converged
    }


# ===============================================================================================================================================================
# FUNCTION INPUT
# ===============================================================================================================================================================

expression = input("Enter the function f(x, y, z): ")

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

def f(x ,y, z):
    values = allowed.copy()
    values["x"] = x
    values["y"] = y
    values["z"] = z

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

try:
    z_min = float(input("Enter the minimum value for z: "))
    z_max = float(input("Enter the maximum value for z: "))
except ValueError:
    print("Please enter a valid numeric value for z!!!")
    sys.exit()

if z_min >= z_max:
    print("z_min must be smaller than z_max!!!")
    sys.exit()

x = float(input("Enter the value of x: "))

if x < x_min or x > x_max:
    print("Initial x must lie within the given x range!!!")
    sys.exit()

y = float(input("Enter the value of y: "))

if y < y_min or y > y_max:
    print("Initial y must lie within the given y range!!!")
    sys.exit()

z = float(input("Enter the value of z: "))
if z < z_min or z > z_max:
    print("Initial z must lie within the given z range!!!")
    sys.exit()

h = float(input("Enter the value of h: "))
if h <= 0:
    print("Value of h must be greater than zero!!!")
    sys.exit()

alpha = float(input("Enter the learning rate: "))
if alpha <= 0:
    print("Value of learning rate must be greater than zero!!!")
    sys.exit()

result = GRADIENT_DESCENT_in_3D(f, x, y, z, h, alpha)
print(result)

x_history = result["x_history"]
y_history = result["y_history"]
z_history = result["z_history"]

x_values = np.linspace(x_min, x_max, 100)
y_values = np.linspace(y_min, y_max, 100)
z_values = np.linspace(z_min, z_max, 100)

X, Y, Z = np.meshgrid(x_values, y_values, z_values, indexing = "ij")

try:
    F = f(X, Y, Z)
except Exception:
    print('''
Invalid Function!!!
Please check: 
- Variable names: x, y, z
- Mathematical syntax
- Supported functions
''')
    sys.exit()

if np.isscalar(F):
    np.full(X.shape, F, dtype = float)

F = np.asarray(F, dtype = float)

# ===============================================================================================================================================================
# 3D FIGURE
# ===============================================================================================================================================================

fig = plt.figure(figsize = (10, 8))
ax = fig.add_subplot(111, projection = "3d")

# ===============================================================================================================================================================
# VALIDITY CHECK
# ===============================================================================================================================================================

valid = np.isfinite(F)

if not np.any(valid):
    print("Function produced no valid values in the given range!!!")
    sys.exit()

if not np.all(valid):
    print("Function produced undefined values in the given range!!!")
    print("Please choose a different range or modify the given function!!!")
    sys.exit()

# ===============================================================================================================================================================
# CHECK WEATHER f(x, y, z) = 0 SURFACE EXISTS 
# ===============================================================================================================================================================

if np.min(F) > 0 or np.max(F) < 0:
    print("No real surface is found in the given range!!!")
    sys.exit()

# ===============================================================================================================================================================
# MARCHING CUBES 
# ===============================================================================================================================================================

try:
    from skimage.measure import marching_cubes
except ImportError:
    print("scikit-image is required for 3D implicit plotting!!!")
    print("\nInstall it using:")
    print("\npip install scikit-image")
    sys.exit()

try:
    vertices, faces, normals, values = marching_cubes(F, level = 0)
except RuntimeError:
    print("No surface found for f(x, y, z) = 0 in the given range!!!")
    sys.exit()

# ===============================================================================================================================================================
# CONVERT MARCHING-CUBES INDEX COORDINATES TO ACTUAL x, y, z COORDINATES
# ===============================================================================================================================================================

vertices[:, 0] = np.interp(vertices[:, 0], np.arange(len(x_values)), x_values)
vertices[:, 1] = np.interp(vertices[:, 1], np.arange(len(y_values)), y_values)
vertices[:, 2] = np.interp(vertices[:, 2], np.arange(len(z_values)), z_values)

# ===============================================================================================================================================================
# DRAW 3D SURFACE
# ===============================================================================================================================================================

ax.plot_trisurf(vertices[:, 0], vertices[:, 1], faces, vertices[:, 2], alpha = 0.7)

ax.set_title(f"GRADIENT DESCENT VISUALIZER OF f(x, y, z) = {expression}", color = "maroon", weight = "bold")

ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")
ax.set_zlabel("z-axis")

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(z_min, z_max)

ax.view_init(elev = 25, azim = 45)

point = ax.scatter([x_history[0]], [y_history[0]], [z_history[0]], color = "red", s = 50, depthshade = False)

plt.grid(linestyle = "--")

# ===============================================================================================================================================================
# ANIAMTION
# ===============================================================================================================================================================

def update(frame):
    current_x = x_history[frame]
    current_y = y_history[frame]
    current_z = z_history[frame]

    point._offsets3d = ([current_x], [current_y], [current_z])
    return point, 

animation = FuncAnimation(fig, update, frames = len(x_history), interval = 300, repeat = False)

plt.show()