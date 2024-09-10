Data class
- 



Data I/O

input: filename
TODs
 -  `read_{exp_name}_tod`
 -- `read_{exp_name}_tod_from_{filetype}`
 -- `get_{exp_name}_pointing`
 -- `get_{exp_name}_obs_time`
 -- `get_{exp_name}_obs_position`
Auxiliary data
 - `read_bl`
 - `read_slm`
 - `read_bandpass`
output: 
 - list of scan objects



Commander4 will effectively have three+ codes running in parallel:
1. TOD processing      :  f(sky, TODs) -> freqmaps
2. Component separation:  g(freqmaps)  -> sky
3. Zodi estimation     :  h(sky, TODs) -> zodi
