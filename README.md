# Gradient Descent Visualizer 📉🚀

An interactive, multi-dimensional mathematics visualizer built with Python, NumPy, and Matplotlib. 
This application dynamically animates the Gradient Descent optimization algorithm across 1D curves, 2D contour maps, and complex 3D surfaces.

## 🌟 Key Features

- **Multi-Dimensional Support:** Capable of visualizing optimization trajectories for 1D, 2D, and 3D mathematical functions parsed dynamically from user input.

- **Live Animation:** Utilizes Matplotlib's `FuncAnimation` module to animate the step-by-step descent of a target point towards the local minimum.

- **3D Surface Generation:** Implements the Marching Cubes algorithm (via scikit-image) to extract and render complex 3D implicit surfaces dynamically.

- **Custom Hyperparameters:** Allows users to manually define the Learning Rate ($\alpha$), iteration limits, tolerance thresholds, and finite-difference step sizes ($h$).

- **Robust Math Engine:** Features strict error handling for out-of-bounds calculations, `ZeroDivisionError`, and undefined domain constraints.

## 🛠️ Tech Stack

- **Language:** Python 3

- **Core Logic:** NumPy (Numerical arrays, mathematical functions)

- **Visualization:** Matplotlib (2D plots, contours, 3D scatter, and animations)

- **3D Meshing:** `scikit-image` (Used exclusively for the 3D Marching Cubes surface extraction)

## 💻 How to Run (Installation & Execution)

*Follow these steps to run the visualizer on your local machine:*

1. ### Clone the repository:

   git clone https://github.com/amit39underworld93-bot/GRADIENT_DESCENT_VISUALIZER.git cd GRADIENT_DESCENT_VISUALIZER

2. ### Install dependencies:
   *This project requires specific libraries for arrays, plotting, and 3D rendering.*

   `pip install numpy matplotlib scikit-image`

3. ### Execute the script:
   *Run the main file to start the interactive terminal prompt.*

   `python3 main_4.py`

## 📸 Usage Overview

*Upon execution, the terminal will prompt you to select the dimension and input your mathematical function and hyperparameters:*

Enter the dimension between 1 and 3: 2
Enter the function f(x, y): x**2 + y**2
Enter the minimum value for x: -10
Enter the maximum value for x: 10
Enter the minimum value for y: -10
Enter the maximum value for y: 10
Enter the value of x: 8
Enter the value of y: 8
Enter the value of h: 0.001
Enter the learning rate: 0.1
Enter the number of iterations: 100
Enter the value of tolerance of x: 0.0001
Enter the value of tolerance of y: 0.0001

***Once the parameters are set, a Matplotlib window will open, animating the red marker as it descends the cost function surface toward the minimum.***
