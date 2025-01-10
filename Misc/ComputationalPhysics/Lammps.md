# Lammps

[lammps Run_head](https://docs.lammps.org/Run_head.html)

[lammps tutorials](https://www.lammps.org/tutorials.html)

[lammps papers](https://www.lammps.org/papers.html)

LAMMPS executes calculations by reading commands from a input script (text file), one line at a time. When the input script ends, LAMMPS exits. This is different from programs that read and process the entire input before starting a calculation.

Each command causes LAMMPS to take some immediate action without regard for any commands that may be processed later. Commands may set an internal variable, read in a file, or run a simulation. These actions can be grouped into three categories:

1. commands that change a global setting (examples: [timestep](https://docs.lammps.org/timestep.html), [newton](https://docs.lammps.org/newton.html), [echo](https://docs.lammps.org/echo.html), [log](https://docs.lammps.org/log.html), [thermo](https://docs.lammps.org/thermo.html), [restart](https://docs.lammps.org/restart.html)),
2. commands that add, modify, remove, or replace “styles” that are executed during a “run” (examples: [pair_style](https://docs.lammps.org/pair_style.html), [fix](https://docs.lammps.org/fix.html), [compute](https://docs.lammps.org/compute.html), [dump](https://docs.lammps.org/dump.html), [thermo_style](https://docs.lammps.org/thermo_style.html), [pair_modify](https://docs.lammps.org/pair_modify.html)), and
3. commands that execute a “run” or perform some other computation or operation (examples: [print](https://docs.lammps.org/print.html), [run](https://docs.lammps.org/run.html), [minimize](https://docs.lammps.org/minimize.html), [temper](https://docs.lammps.org/temper.html), [write_dump](https://docs.lammps.org/write_dump.html), [rerun](https://docs.lammps.org/rerun.html), [read_data](https://docs.lammps.org/read_data.html), [read_restart](https://docs.lammps.org/read_restart.html))

Commands in category a) have default settings, which means you only need to use the command if you wish to change the defaults.

LAMMPS 通过从输入脚本（文本文件）中读取命令来执行计算，一次一行。当输入脚本结束时，LAMMPS 退出。这与在开始计算之前读取和处理整个输入的程序不同。

每个命令都会使 LAMMPS 立即采取一些行动，而不考虑以后可能处理的任何命令。命令可以设置内部变量、读入文件或运行模拟。这些操作可以分为三类：

1. 更改全局设置的命令（例如：Timestep、Newton、Echo、Log、Thermo、Restart）、

2. 添加、修改、删除或替换在“运行”期间执行的“样式”的命令（例如：pair_style、fix、compute、dump、thermo_style、pair_modify），以及

3. 执行 “run” 或执行某些其他计算或操作的命令（例如：print、run、minimize、temper、write_dump、rerun、read_data、read_restart）

类别 a） 中的命令具有默认设置，这意味着如果您希望更改默认值，则只需使用该命令。

You can use the `-skiprun` command line flag to have LAMMPS skip the execution of any run, minimize, or similar commands to check the entire input for correct syntax to avoid crashes on typos or syntax errors in long runs.

# 2 Parsing rules for input scripts

case sensitive:

- command and command arguments are lower-case
- file names or user-chosen ID strings may be upper-case.

## Ensemble 系综

在数学物理学中，特别是 J. Willard Gibbs 于 1902 年引入统计力学和热力学中，系综（也称为统计系综）是由系统的大量虚拟副本（有时是无限多个）组成的理想化，一次考虑所有副本，每个副本都代表真实系统可能处于的一种可能状态。换句话说，统计集成是系统状态的概率分布。

热力学系综是统计系综的一种特定种类，除其他性质外，它处于统计平衡状态（定义见下文），用于从经典力学或量子力学定律推导出热力学系统的性质。

// [text](https://zhuanlan.zhihu.com/p/350907022)

在统计物理中，系综（英语：ensemble）代表一定条件下一个体系的大量可能状态的集合。也就是说,系综是系统状态的一个概率分布。对一相同性质的体系，其微观状态（比如每个粒子的位置和速度）仍然可以大不相同。（实际上，对于一个宏观体系，所有可能的微观状态数是天文数字。）在概率论和数理统计的文献中，使用“概率空间”指代相同的概念。

统计物理的一个原理（各态历经原理）是：对于一个处于平衡的体系，物理量的时间平均，等于对对应系综里所有体系进行平均的结果。

体系的平衡态的物理性质可以对不同的微观状态求和来得到。系综的概念是由约西亚·吉布斯在1878年提出的。

常用的系综有：

- 微正则系综（microcanonical ensemble）：
  - 系综里的每个体系具同的能量e（通常每个体系的粒子数n和体积v也是相同的）。
  - nve
  - nve系综没有控温的功能，初始条件确定后，在力场的作用下，原子速度发生变化，相应的体系温度发生变化。体系总能量e=势能+动能，温度发生变化，动能就会变化，势能和动能相互转换，总能量保持不变。
  - to reach equilibrium
  - `fix 1 all nve`
- 正则系综 （canonical ensemble）：
  - 系综里的各体系可以和外界环境交换能量（每个体系的粒子数n和体积v仍然是固定且相同的），但系综内各体系有相同的温度t。  
  - nvt
  - nvt系综下，模拟盒子box的尺寸不会发生变化，lammps通过改变原子的速度对体系的温度进行调节。
- 巨正则系综（grand canonical ensemble）：
  - 正则系综的推广，各体系可以和外界环境交换能量和粒子，但系综内各个体系有相同的温度和化学势。
- 等温等压系综（isothermal-isobaric ensemble）：
  - 正则系综的推广，各体系可以和外界环境交换能量和体积，但系综内各个体系有相同的温度和压强。
  - npt
  - npt系综不仅进行控温，还进行控压。和nvt一样，npt系综通过调节原子速度调控温度，不同的是，npt系综下box的尺寸可以发生变化。
  - npt系综通过改变box的尺寸调节压力，比如，当体系压力超过设定值时，扩大box尺寸降低压力。
  - `fix ID group-ID npt temp Tstart Tstop Tdamp Pstart Pstop Pdamp`
- 在系综中，物理量的变化范围（fluctuation）与其本身大小的比值会随着体系变大而减小。于是，对于一个宏观体系，从各种系综计算出的物理量的差异将趋向于零。
