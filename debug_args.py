#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
print("Python version:", sys.version)
print("Arguments received:", sys.argv)
print("Number of arguments:", len(sys.argv))
for i, arg in enumerate(sys.argv):
    print(f"  arg[{i}]: '{arg}' (length: {len(arg)})")
