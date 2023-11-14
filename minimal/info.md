# Minimal Infomation

## Data Structure
### File Name

Example:  
> id=1849_skid=2062_masid=1941_rid=2009_type=fine
- $id$: unique index among current minimal files
- $skid$: skeleton index(optitrack $id$)
- $masid$: plcs index(master $id$)
- $rid$: radar index(arbe $id$)
- $type$: skeleton type(fine=OK, nan=skeleton has nan value)

### SMPL Params
#### Pose
pose: 1 * 72(3 coord + (24 joints - 1 pelvis) * 3 pose)  
> pose range is [-1, 1]  

#### Shape
shape: 1 * 10  
> shape range is [0, 1]

## Usage
### Raw Joints and PCLs to SMPL

```python
# init solver
smpl = KinematicModel(config.SMPL_MODEL_1_0_MALE_PATH, armatures.SMPLArmature)

wrapper = KinematicPCAWrapper(smpl)
solver = Solver(wrapper)

# solve full: jnts(29 * 3), pcl(N * 3)
mesh, losses = solver.solve(jnts, pcl, "full")

# solve pose only
mesh, losses = solver.solve(jnts, pcl, "pose")

# get SMPL params: 1 * 82
params = solver.params()
```

### Params to SMPL Model

```python
# params can be either np.lib.npyio.NpzFile(from np.load), ndarray(1 * 72), or ndarray(1 * 82)
solver.update_params(params)
mesh, jnts = solver.model.run(solver.params())
```

### Transform

```python
# KinematicModel defualt returns upperback_coord_jnts
upperback_coord_jnts = smpl_coord_jnts - params[:3]

# R, t, scale are from $root_path$/minimal/trans/*.npz
radar_coord_jnts = (upperback_coord_jnts @ R + t) * scale
```

## Visualization

### Mesh
```python
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes

# mesh is from solver.solve, model.run, or obj file
verts, faces, _ = load_obj(filepath)
mesh = Meshes([verts], [faces[0]])
o3d_plot([o3d_mesh(mesh)])
```