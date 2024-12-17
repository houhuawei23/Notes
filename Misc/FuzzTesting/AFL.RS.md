# AFL: American Fuzzy Lop

## Issue 1: Debian12 default python3.11, if use conda env python3.12, and build afl, afl need libpython3.12.so.1.0, ....

solve: use python3.11 to build afl

recommand: change conda base env python version to 3.11
