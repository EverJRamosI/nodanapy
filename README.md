<div align="center" rol="img">
<img src="https://raw.githubusercontent.com/EverJRamosI/nodanapy/main/docs/images/NODANAPY.png" width="300">
</div>
<br>

![stars](https://img.shields.io/github/stars/EverJRamosI/nodanapy
)
![forks](https://img.shields.io/github/forks/EverJRamosI/nodanapy
)
![watchers](https://img.shields.io/github/watchers/EverJRamosI/nodanapy
)
![issues](https://img.shields.io/github/issues/EverJRamosI/nodanapy
)
![license]()

# NodAnaPy

NodAnaPy is the fundamental package petroleum, in this package you will find to determine the Well Nodal Analysis for oil and gas.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Theory](#theory)
- [Dependencies](#dependencies)
- [License](#license)

## Installation
You can install [NodAnaPy](https://github.com/EverJRamosI/nodanapy/tree/main/nodanapy) using pip:
```bash
pip install nodanapy
```

## Usage
Here is a quick example of how to use NodAnaPy to calculate the optimal rate of an oil well.

**INPUT**
```python
import matplotlib.pyplot as plt
from nodanapy import WellNAOilR

pr = 4500 # reservoir pressure
tr = 140 + 460 # reservoir temperature
ph = 140 # wellhead pressure
th = 84 + 460 # wellhead temperature
pb = 1500 # bubble pressure

well = WellNAOilR(ph, th, pr, tr, pb)

ipr_well = well.ipr
vlp_well = well.vlp

fig, (ax_ql, ax_qg, ax_qo) = plt.subplots(3, 1)

ax_ql.set_xlabel('Ql(bbl/d)')
ax_ql.set_ylabel('Pwf(psia)')
ax_ql.plot(ipr_well.Ql, ipr_well.Pwf)
ax_ql.plot(vlp_well.Ql, vlp_well.Pwf)
    
ax_qg.set_xlabel('Qg(Mscf/d)')
ax_qg.set_ylabel('Pwf(psia)')
ax_qg.plot(ipr_well.Qg, ipr_well.Pwf)
ax_qg.plot(vlp_well.Qg, vlp_well.Pwf)
    
ax_qo.set_xlabel('Qo(bbl/d)')
ax_qo.set_ylabel('Pwf(psia)')
ax_qo.plot(ipr_well.Qo, ipr_well.Pwf)
ax_qo.plot(vlp_well.Qo, vlp_well.Pwf)
```
**OUTPUT**
![Well Nodal Analysis](https://raw.githubusercontent.com/EverJRamosI/nodanapy/main/docs/images/Figure.png)
> Ql(bbl/d) vs Pwf(psia); Qg(Mscf/d) vs Pwf(psia); Qo(bbl/d) vs Pwf(psia)

## Theory
it will be to calculate inflow performance relationship (IPR) and vertical lift performance (VLP) curves for oil and gas wells. In addition, it will be to determine the holdup liquid well.

<details>
<summary>Click to expand the theory section.</summary>

### Inflow Performance Relationship (IPR)
The curve is calculated using the different correlations, for example:
- Oil
  - Vogel
  - Fetkovich
- Gas
  - LIT
  - Darcy

### Vertical Lift Performance (VLP)
The curve is calculated using the different correlations, for example:
- Oil
  - Hagedorn Brown
  - Beggs and Brill
- Gas
  - Gray

</details>

## Dependencies
NodAnaPy requires the following Python (=>3.12) libraries:
1. PetProPy
2. Numpy
3. Scipy
4. Pandas

```bash
pip install -r requirements.txt
```

## License
This project is licensed under the GNU Lesser General Public License v3.0 - see the [LICENSE](https://github.com/EverJRamosI/nodanapy/blob/main/LICENSE) file for details.

## API Documentation
For detailed API documentation, please refer to the [official documentation](https://github.com/EverJRamosI/nodanapy)