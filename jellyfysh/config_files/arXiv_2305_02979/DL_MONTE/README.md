# DL_MONTE

## Amended Version

Our minimally amended version of [DL_MONTE](https://gitlab.com/dl_monte/DL_MONTE-2) that outputs the electric 
polarization is currently not publicly available. Contact us to request access to it. Alternatively, use the trajectory
files of the simulations to extract the polarization.

## Restarts

The maximum number of Monte-Carlo steps in [DL_MONTE](https://gitlab.com/dl_monte/DL_MONTE-2) is limited by the largest 
possible integer in Fortran. Use the `CONTROL`, `CONFIG`, and `FIELD` files in the respective directories to start 
an initial simulation. Among others, this will create the `REVCON.000`, `RSTART.000`, and `REVIVE.000` files. In order 
to continue the simulation, rename `REVCON.000` to `CONFIG` and `CONTROL_RESTART` to `CONTROL`, and make sure run 
[DL_MONTE](https://gitlab.com/dl_monte/DL_MONTE-2) in a directory that also contains the `FIELD`, `RSTART.000`, and 
`REVIVE.000` files.

