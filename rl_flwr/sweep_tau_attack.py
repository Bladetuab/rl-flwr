"""sweep_tau_attack.py  –  DEPRECATED
=====================================
The sweep now runs entirely inside server_app_sweep.py with a single
`flwr run .` command.  This file is no longer needed.

See server_app_sweep.py for the implementation.
"""
print("This file is deprecated. Use server_app_sweep.py instead.")
print("Steps:")
print("  cp server_app_sweep.py server_app.py")
print("  flwr run .")