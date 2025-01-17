# Profiling

- [hackingcpp: profilers](https://hackingcpp.com/cpp/tools/profilers.html)
- [kdab: C/C++ Profiling Tools](https://www.kdab.com/c-cpp-profiling-tools/)
- [stackoverflow: How do I profile C++ code running on Linux? ](https://stackoverflow.com/questions/375913/how-do-i-profile-c-code-running-on-linux)
- [reddit: What is your favourite profiling tool for C++?](https://www.reddit.com/r/cpp/comments/14mjhxa/what_is_your_favourite_profiling_tool_for_c/)
## Tools

- [gnu: gprof](https://ftp.gnu.org/old-gnu/Manuals/gprof-2.9.1/html_mono/gprof.html)
- [github: orbit](https://github.com/google/orbit)

### `gprof`

```bash
g++ -pg main.cpp -o main    # compile with flag '-pg'
./main abc def              # run
gprof ./main gmon.out > profile_report.txt  # generate report

# use gprof2dot to visual call graph (pip install gprof2dot)
gprof /bin/MDsim gmon.out | gprof2dot | dot -Tpng -o call_graph.png
```

profile report:

- Flat Profile: Lists the functions in order of time spent during execution.
- Call Graph: Displays relationships between functions, including which functions called others and how much time was spent in each.
