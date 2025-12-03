import yaml
from copy import deepcopy
import os

hm3d_labels = ['camera_lights_1', 'no_lights_1', 'velocity_1', 'camera_lights_2', 'no_lights_2', 'velocity_2']
hm3d_scenes = ['00800-TEEsavR23oF', '00802-wcojb4TFT35', '00803-k1cupFYWXJ6', '00808-y9hTuugGdiq']

replica_cad_labels = ['baseline', 'camera_lights', 'dynamic_lights', 'no_lights', 'velocity']
replica_cad_scenes = [
    'apt_0',
    'apt_3',
    'v3_sc0_staging_00',
    'v3_sc0_staging_12',
    'v3_sc0_staging_16',
    'v3_sc0_staging_19',
    'v3_sc0_staging_20',
    'v3_sc1_staging_00',
    'v3_sc1_staging_06',
    'v3_sc1_staging_12',
    'v3_sc1_staging_19',
    'v3_sc1_staging_20',
    'v3_sc2_staging_00',
    'v3_sc2_staging_11',
    'v3_sc2_staging_13',
    'v3_sc2_staging_19',
    'v3_sc2_staging_20',
    'v3_sc3_staging_03',
    'v3_sc3_staging_04',
    'v3_sc3_staging_08',
    'v3_sc3_staging_15',
    'v3_sc3_staging_20',
]

def generate_configs(base_yaml, labels, scenes, output_dir):
    with open(base_yaml, 'r') as file:
        yaml_data = yaml.safe_load(file)

    for label in labels:
        for scene in scenes:
            config = deepcopy(yaml_data)

            config['dataset']['sequence'] = f"{label}/{scene}"
            config['nodes_constructor']['output_name_nodes'] = f"{label}_{scene}.json"
            config['nodes_constructor']['output_name_objects'] = f"{label}_{scene}.pkl.gz"

            with open(os.path.join(output_dir, f"{label}_{scene}.yaml"), 'w') as file:
                yaml.safe_dump(config, file)

def main():
    generate_configs('examples/configs/hm3d/example.yaml', hm3d_labels, hm3d_scenes, 'examples/configs/hm3d/')
    generate_configs('examples/configs/replica_cad/example.yaml', replica_cad_labels, replica_cad_scenes, 'examples/configs/replica_cad/')

if __name__ == '__main__':
    main()