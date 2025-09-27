import os
from pathlib import Path

PROJECT_NAME = "DrowsinessDetector"

file_list = [
    f'src/{PROJECT_NAME}/__init__.py',
    f'src/{PROJECT_NAME}/components/__init__.py',
    f'src/{PROJECT_NAME}/components/data_ingestion.py',
    f'src/{PROJECT_NAME}/components/data_transformation.py',
    f'src/{PROJECT_NAME}/components/train.py',
    f'src/{PROJECT_NAME}/components/evaluate.py',
    f'src/{PROJECT_NAME}/utils/__init__.py',
    f'src/{PROJECT_NAME}/utils/utils.py',
    f'src/{PROJECT_NAME}/data_config/__init__.py',
    f'src/{PROJECT_NAME}/data_config/data_cfg.py',
    f'src/{PROJECT_NAME}/config_manager/__init__.py',
    f'src/{PROJECT_NAME}/config_manager/config.py',
    f'src/{PROJECT_NAME}/constants/__init__.py',
    f'src/{PROJECT_NAME}/pipeline/__init__.py',
    f'src/{PROJECT_NAME}/pipeline/ingestion_pipeline.py',
    f'src/{PROJECT_NAME}/pipeline/transformation_pipeline.py',
    f'src/{PROJECT_NAME}/pipeline/training_pipeline.py',
    f'src/{PROJECT_NAME}/pipeline/evaluation_pipeline.py',
    f'src/{PROJECT_NAME}/logging/__init__.py',
    f'src/{PROJECT_NAME}/logging/logger.py',
    f'src/{PROJECT_NAME}/exception/__init__.py',
    f'src/{PROJECT_NAME}/exception/exception.py',
    'templates/index.html',
    'experiments/detector.ipynb',
    'config.yaml',
    'requirements.txt',
    'params.yaml',
]

for file_path in file_list:
    file, parent = Path(file_path), Path(file_path).parent

    if not parent.is_dir():
        parent.mkdir(parents=True, exist_ok=True)

    if not file.exists():
        file.touch()

# if __name__ == '__main__':
#     project_struct()
#     print(f"Project structure for '{PROJECT_NAME}' created successfully.")