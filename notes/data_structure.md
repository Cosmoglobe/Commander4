Data class



Data I/O

TOD Data Class
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
 - `read_detector_covar_struct`
output: 
 - list of scan objects


CompSep Data Class


Input:
 - Maps
 - Noise model
 - Bandpasses
 - Beams
Output:
 - Sky map `a_lm`'s.



Sky map data classes (default to Commander3 style)
 - alm's
 - SEDs

Gibbs chain file  (~2 TB for DIRBE)
 - iteration
   - sky components (alms, sed parameters, etc.)
   - instrument characteristics (gain, sigma0, etc.)
   - zodi parameters



Note that component separation needs to be done on most `n_band` nodes,
since they require an SHT.

Internal to component separation:
Sampling groups - cannot be done in parallel, since each group is conditioned on the previous one.

MH sampling groups?


Zodi Data Class?


What is the data hierarchy?
  - Martin's model: Detector owns scan, detector group owns detector
            - e.g., 70 GHz/det{1-4}/scan{1-15}
  - Mathew's model: Scan owns detector, detector group owns scan
            - e.g., 70 GHz/scan{1-15}/det{1-4}
            - chosen because we were not talking about detector cross-correlation
  - Commander4: we want cross-scan and cross-detector information. In some sense, we need to
                "transpose" the scan and detector arrays.

We want to make it easier to model cross-talk, which is not done in Commander3.

Commander4 will effectively have three+ codes running in parallel:
1. TOD processing      :  f(sky, TODs) -> freqmaps
2. Component separation:  g(freqmaps)  -> sky
3. Zodi estimation     :  h(sky, TODs) -> zodi





Component separation:
   1. Loop over Conjugate Gradient (CG) groups
   2. For each loop, compute Ax, distribute results among cores

We should generalize this to using other samplers, including, e.g., the MH groups.
