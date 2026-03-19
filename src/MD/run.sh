#!/usr/bin/env bash
export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH
/home/kenobi/Workspaces/PHYS400/phys/bin/python lmp.py "$@"
