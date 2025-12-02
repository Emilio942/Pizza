# Wrapper to re-export simulation implementation
import os
import sys
import importlib.util

_module_name = 'logging_system'
_file_path = os.path.join(os.path.dirname(__file__), '..', 'simulation', 'src', f'{_module_name}.py')
# Ensure simulation/src is on sys.path for sub-imports
_sim_src_dir = os.path.join(os.path.dirname(__file__), '..', 'simulation', 'src')
if _sim_src_dir not in sys.path:
	sys.path.insert(0, _sim_src_dir)
_spec = importlib.util.spec_from_file_location(f'simulation_impl.{_module_name}', _file_path)
if _spec and _spec.loader:
	_mod = importlib.util.module_from_spec(_spec)
	_spec.loader.exec_module(_mod)  # type: ignore[attr-defined]
	globals().update({k: v for k, v in vars(_mod).items() if not k.startswith('_')})
else:
	raise ImportError(f'Could not load simulation module for {_module_name} from {_file_path}')
