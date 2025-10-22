# Self-Assessment: Example Solutions for our Hands-On Exercises

This section contains short, focused exercises that help you practice the concepts introduced in the User Guide. Each exercise is designed to be run as a notebook and to guide you from problem statement to a working solution.

What you will get
- Targeted practice that mirrors real usage of Semantic World
- Clear goals and constraints for each task
- Hints where appropriate, and space for your own solution

Prerequisites
- A working Python environment with the project dependencies installed (see README and requirements.txt)
- Jupytext and Jupyter installed (pip install jupytext jupyter)
- To check if you have set up everything correctly run `bash scripts/test_exercises.sh` from the project root.

How to use these exercises
1. Work through the corresponding topic in the User Guide first, so the terminology and workflows are familiar.
2. In your command line, navigate to scripts and run the command `bash scripts/convert_exercises_for_self_assessment.sh`
3. Open an exercise and read the task description before touching any code.
4. Implement your solution in the dedicated cells. Keep your code small and readable. If your goal is to contribute to the project, consider also reading our [developer guide](https://cram2.github.io/semantic_world/developer_guide.html) first
5. Run the checks in the exercise to validate your work. If they pass, you may assume that your solution is correct.
6. If you are stuck or want to compare your solution to an example solution, you can find a working solution by coming back to this section, open the corresponding solution page on this documentation.

Running exercises locally
- Preview in the documentation: Simply open the exercise pages in this book.
- Execute as notebooks: Each exercise is written in MyST Markdown and can be executed as a notebook locally.
  - Convert to notebooks: scripts/convert_exercises_for_self_assessment.sh
  - Optionally run quick checks: scripts/test_exercises.sh
