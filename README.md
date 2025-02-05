# Water Demand Prediction

This repository contains notebooks and scripts for predicting and simulating water demand using various machine learning models. The project leverages Amazon SageMaker Autopilot for model training and inference.

## Repository Structure
. ├── autopilot_demand_inference.ipynb ├── autopilot_demand_simulation_inference_c5m5xlarge.ipynb ├── autopilot_demand_simulation_inference_m4c4.ipynb ├── autopilot_demand_simulation_inference_m5large_c4c5-2x.ipynb ├── autopilot_demand_training.ipynb ├── consolidate_prediction_results.ipynb ├── consolidate_simulation_results.ipynb ├── data/ │ ├── cm.csv │ ├── Lower Hutt xpARA.csv │ ├── North Wellington (Moa) xpARA.csv │ └── ... ├── README.md ├── SageMakerAutopilotCandidateDefinitionNotebook.ipynb ├── SageMakerAutopilotDataExplorationNotebook.ipynb └── scripts/ ├── prep_full_results_inference.py └── prep_simulation_data.py


## Notebooks

- **autopilot_demand_inference.ipynb**: Notebook for running inference on water demand predictions.
- **autopilot_demand_simulation_inference_c5m5xlarge.ipynb**: Simulation inference notebook for c5m5xlarge instance type.
- **autopilot_demand_simulation_inference_m4c4.ipynb**: Simulation inference notebook for m4c4 instance type.
- **autopilot_demand_simulation_inference_m5large_c4c5-2x.ipynb**: Simulation inference notebook for m5large_c4c5-2x instance type.
- **autopilot_demand_training.ipynb**: Notebook for training water demand prediction models using SageMaker Autopilot.
- **consolidate_prediction_results.ipynb**: Notebook for consolidating prediction results.
- **consolidate_simulation_results.ipynb**: Notebook for consolidating simulation results.
- **SageMakerAutopilotCandidateDefinitionNotebook.ipynb**: Notebook for defining SageMaker Autopilot candidates.
- **SageMakerAutopilotDataExplorationNotebook.ipynb**: Notebook for exploring data using SageMaker Autopilot.

## Scripts

- **prep_full_results_inference.py**: Script for preparing full results for inference.
- **prep_simulation_data.py**: Script for preparing simulation data.

## Data

The `data` directory contains various CSV files used for training and inference.

## Usage

1. **Training**: Use the `autopilot_demand_training.ipynb` notebook to train models using SageMaker Autopilot.
2. **Inference**: Use the `autopilot_demand_inference.ipynb` notebook to run inference on the trained models.
3. **Simulation**: Use the simulation inference notebooks for different instance types to run simulations.
4. **Consolidation**: Use the consolidation notebooks to consolidate prediction and simulation results.

## Requirements

- Jupyter Notebook
- Amazon SageMaker
- Python 3.x
- Required Python packages (listed in the notebooks)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the contributors and the open-source community for their valuable work.