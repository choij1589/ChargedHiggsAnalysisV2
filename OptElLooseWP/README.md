# OptElLooseWP
Submodule to re-optimize electron loose WP. With changed electron tight ID from Top-HN analysis, we found that there is large flavour dependency for non-prompt electrons. Three variables are possible re-optimizable variables, SIP3D / MiniRelIso and MVA WPs.

## Previous WP
### MiniRelIso
Tight: 0.1
Loose: 0.4

### SIP3D
Tight & Loose: 4

### MVANoIso
Tight: pass_MVANoIso_WP90
Loose:
- 2016a: 0.96, 0.93, 0.85
- 2016b: 0.96, 0.93, 0.85
- 2017:  0.94, 0.79, 0.5
- 2018:  0.94, 0.79, 0.5

## New WP
### MiniRelIso
Due to trigger PT threshold, it is not a good idea to change MiniRelIso loose WP

### SIP3D
We can get additional room for optimizing WP with MVA scores by loosening SIP3D WP
Tight: 4
Loose: 6

### MVANoIso
Tight: pass MVANoIso_WP90
Loose:
- 2016a: 0.985, 0.98, 0.75
- 2016b: 0.985, 0.98, 0.75
- 2017:  0.985, 0.96, 0.85
- 2018:  0.985, 0.96, 0.85
