# Optimizing Functions of One Variable: Cost Minimization

## Course Information
- **Course:** Course 1 - Machine Learning: Calculus
- **Platform:** DeepLearning.AI
- **Assignment:** Week 1 - Optimizing Functions of One Variable

## 📌 Overview
This assignment introduced the fundamental concepts of **optimization** through a practical business problem: minimizing costs by deciding how to split product purchases between two suppliers based on historical price data. 

The problem was framed as a **portfolio management** scenario where I needed to find the optimal allocation percentage ω (omega) that minimizes the variance (risk) of historical costs. This served as an accessible introduction to calculus-based optimization before moving to multivariate functions in later weeks.

## 🎯 What I Learned
- **Practical Optimization:** Applied optimization techniques to a real-world business problem (supplier cost minimization).
- **Variance Minimization:** Learned how to construct and minimize a loss function (variance) to find the optimal portfolio allocation.
- **Data Manipulation with Pandas:** Read and explored CSV data using pandas DataFrames, accessing specific columns.
- **Numerical Optimization:** Implemented a brute-force approach by evaluating the loss function at many points (1001 values) to find the minimum with high accuracy.
- **Automatic Differentiation:** Used JAX's `grad()` function to compute derivatives automatically and verify that the derivative approaches zero at the minimum point.
- **Function Optimization:** Understood the relationship between a function's minimum and its derivative being zero.
- **Visualization:** Plotted both the loss function and its derivative to visualize the optimization process.

## ⚙️ Key Skills Practiced
- Data loading with pandas
- Array manipulation with JAX numpy
- Loss function construction from business requirements
- Numerical optimization (grid search)
- Automatic differentiation with JAX
- Visualization with matplotlib
- Vectorized operations vs. loop-based operations

## 🚧 Challenges Faced
1. **Understanding the Loss Function:** Grasping why variance minimization (instead of cost minimization) is the correct approach for portfolio management required some conceptual work.
2. **JAX Array Updates:** Learning to use `.at[<index>].set(<value>)` for updating JAX numpy arrays instead of standard NumPy indexing.
3. **Vectorization vs. Loops:** Deciding between writing vectorized operations for `f_of_omega` and using loops for `L_of_omega_array`.
4. **Automatic Differentiation with JAX:** Understanding how `grad(L_of_omega)` returns a function that can be evaluated at specific points.
5. **Interpreting the Results:** Connecting the mathematical result (ω ≈ 0.702) to the business decision (70.2% from Supplier A, 29.8% from Supplier B).
6. **Derivative Verification:** Confirming that at the minimum point, the derivative is indeed close to zero (dL/dω ≈ -0.129).

## 📋 Important Submission Notes (AutoGrader)
To avoid grader errors from the DeepLearning.AI AutoGrader, I made sure NOT to:
- Delete any exercise cells
- Add solutions in a different cell from the one provided
- Import any new libraries
- Import libraries within graded cells
- Leave any exercise unsolved (replacing None values)

> 💡 **Tip:** Use `# grade-up-to-here` in graded cells to check partial progress. Save and submit the notebook for grading.

## 🛠️ Technologies & Tools
- Python 3
- JAX (automatic differentiation)
- JAX NumPy (jax.numpy)
- Pandas (data manipulation)
- Matplotlib (visualization)
- NumPy (standard, for some operations)

## 📈 Results
- **Optimal Allocation:** Found ω ≈ 0.702 (70.2% from Supplier A, 29.8% from Supplier B)
- **Minimum Loss:** L(ω_min) ≈ 9.2497
- **Derivative at Minimum:** dL/dω ≈ -0.129 (approximately zero, confirming optimality)

### Sample Function Values