# Third-party dependencies and source provenance

The `sim2act_baselines` research implementation integrates with the following separately distributed projects. Their source trees, datasets and pretrained weights are not bundled here. Installing this repository does not install these upstream projects automatically.

| Project | Role | License evidence available during release assembly |
| --- | --- | --- |
| [CrackStructures / CrackEnsembles](https://github.com/ben-z-original/crackstructures) | Multi-view data, official fusion and evaluation reference | The staged upstream `pyproject.toml` declares the GNU General Public License; its referenced `LICENSE` file was absent from the staged copy. Consult the upstream license before reuse. |
| [ENSTRECT](https://github.com/ben-z-original/enstrect) | Mapping, geometric projection and centerline extraction | The staged upstream `pyproject.toml` declares the GNU General Public License; its referenced `LICENSE` file was absent from the staged copy. Consult the upstream license before reuse. |
| [OmniCrack30k](https://github.com/ben-z-original/omnicrack30k) | Image segmentation model | The staged upstream `LICENSE` contains GNU GPL version 3. |
| [PyTorch3D](https://github.com/facebookresearch/pytorch3d) | Mesh and point-cloud operations | Separately installed upstream dependency; consult its license and compatible build instructions. |

NumPy, PyTorch, h5py, SciPy, NetworkX, scikit-learn, pandas, Pillow, psutil, Open3D, PyntCloud, pc-skeletor and robust-laplacian retain their own licenses. Dependency declarations do not grant rights to datasets, model weights or upstream code.

No project-specific license was present in the supplied Sim2Act source snapshots. This release does not assign an MIT license or claim that upstream integrations are permissively licensed. Rights to the supplied research code, manuscript and figures remain with their respective owners unless an explicit license states otherwise.

See [source provenance](docs/SOURCE_PROVENANCE.md) for the exact local source lineage and the small release packaging changes.
