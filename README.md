# WeeGP

Variations in Rust on Riccardo Poli's TinyGP
    <https://cswww.essex.ac.uk/staff/rpoli/TinyGP/>, Java source:
    <https://cswww.essex.ac.uk/staff/rpoli/TinyGP/tiny_gp.java> .

Ongoing additions and variations, including:

* Gradual changes for more idiomatic Rust code
* Crossover expression length is limited to MAX_LEN
* Use a fixed set of numeric constants (pi, 1..10, 100, etc) rather than random constants
* New parameter: Two fitnesses within a given percentage of each other are considered "close", and the competitor with the shorter expression is treated as best.
* When selecting a random terminal, use a specific probability that an independent variable will be selected vs a constant. Currently set at 1/3 independent variables, 2/3 constants.

Example output, using Riccardo's sin-data.txt as input:

```log
-- TINY GP (Rust implementation) --
SEED=-1
MAX_LEN=40
POPSIZE=1000000
DEPTH=5
CROSSOVER_PROB=0.9
PMUT_PER_NODE=0.05
GENERATIONS=2000
TSIZE=2
----------------------------------
...
...
1671secs Generation=292 Avg Fitness=410.06 Best Fitness=0.054514 Avg Size=29.27 Best Size=39
Best Individual: (-1 / ((((((7 / e) + X0) / (((pi - (X0 - pi)) / ((X0 - pi) - (-1 / X0))) * 3)) + 3) / 10) + (pi / ((X0 + (1 / 1000)) * (X0 - pi)))))
```
