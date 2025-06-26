import yaml
from sionna.rt import *
import sionna.rt.scene as builtin_scenes


# Load YAML config
with open("/kaggle/input/sionna/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)
    

# Initialize scene
if cfg['scene']['built_in']!='none':
    scene = load_scene(getattr(builtin_scenes, cfg['scene']['name']))
else:
    file_path="/kaggle/input/sionna-yml/"+cfg['scene']['name']
    scene=load_scene(file_path)

scene.tx_array = PlanarArray(**cfg['scene']['transmitter_array'])
scene.rx_array = PlanarArray(**cfg['scene']['receiver_array'])

tx = Transmitter(**cfg['scene']['transmitter'])
rx = Receiver(**cfg['scene']['receiver'])
scene.add(tx)
scene.add(rx)
tx.look_at(rx)

camera = Camera(**cfg['scene']['camera'])


num_rows=cfg['scene']['transmitter_array']['num_rows']
num_cols=cfg['scene']['transmitter_array']['num_cols']
v_spacing=cfg['scene']['transmitter_array']['vertical_spacing']
h_spacing=cfg['scene']['transmitter_array']['horizontal_spacing']