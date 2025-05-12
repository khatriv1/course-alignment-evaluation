# Course Learning Objective Alignment Evaluation

This project evaluates the alignment between course descriptions and learning objectives using multiple advanced LLM prompting techniques. It systematically compares the effectiveness of different prompting approaches to see which ones best match human expert judgments.

## Project Overview

The project compares nine different prompting techniques:

- **Zero-shot**: Direct prompting with a rubric but no examples
- **Few-shot**: Prompting with demonstrative examples
- **Chain of Thought (CoT)**: Step-by-step reasoning before answering
- **Self-Consistency**: Sampling multiple reasoning paths and selecting the most consistent answer
- **Auto Chain of Thought (Auto-CoT)**: Automatically generated reasoning chains
- **Contrastive Chain of Thought (CCoT)**: Contrasting correct and incorrect reasoning paths
- **Rephrase and Respond (RaR)**: Rephrasing the question before answering
- **Take a Step Back**: Approaching the problem from an abstract level before detailed analysis
- **Active Prompting**: Selects the most informative examples based on uncertainty metrics

Each technique is evaluated on how well it matches human expert ratings of course-objective alignment.

## Directory Structure

```
course_alignment_evaluation/
├── data/
│   └── human-score.csv
├── prompting/
│   ├── zero_shot.py
│   ├── few_shot.py
│   ├── cot.py
│   ├── self_consistency.py
│   ├── auto_cot.py
│   ├── contrastive_cot.py
│   ├── rephrase_and_respond.py
│   ├── take_a_step_back.py
│   └── active_prompt.py
├── utils/
│   └── metrics.py
├── evaluation/
│   ├── evaluate_zero_shot.py
│   ├── evaluate_few_shot.py
│   ├── evaluate_cot.py
│   ├── evaluate_self_consistency.py
│   ├── evaluate_auto_cot.py
│   ├── evaluate_contrastive_cot.py
│   ├── evaluate_rar.py
│   ├── evaluate_step_back.py
│   └── evaluate_active_prompt.py
├── config.py
├── main.py
└── requirements.txt
```

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/khatriv1/course-alignment-evaluation.git
cd course-alignment-evaluation
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up your OpenAI API key in config.py:
```python
OPENAI_API_KEY = "your-api-key-here"
DATA_PATH = "data/human-score.csv"
```

5. Ensure your dataset is in the data directory, following the expected format.

## Usage

Run the main evaluation script to compare techniques:
```bash
python main.py
```

When running, you'll be prompted to:
1. Enter the number of examples to test with (minimum 10 required for statistical validity)
2. Select which techniques to evaluate (by entering their corresponding numbers)

For individual technique evaluation, run:
```bash
python evaluation/evaluate_technique_name.py
```

## Technical Details

### Metrics
The project evaluates each technique using multiple metrics:
- **Accuracy**: How often the model exactly matches human scores
- **Agreement (κ)**: Cohen's Kappa - How well the model agrees with humans beyond chance
- **Consistency (α)**: Krippendorff's Alpha - How reliably the model follows human scoring patterns
- **Correlation (ICC)**: Intraclass Correlation - How closely model scores align with human scores
<!-- - **Off-by-One Accuracy**: How often the model is within ±1 of the human score
- **Mean Absolute Error (MAE)**: Average absolute difference between human and model scores
- **Root Mean Square Error (RMSE)**: Root mean square difference between human and model scores-->

### Output
The program generates:
- Individual technique evaluations
- Comparison visualizations across all evaluated techniques
- Detailed results in CSV format
- Summary report with all metrics

## Example Results
Results will be stored in the `results/evaluation_report` directory, including:
- technique_comparison.csv (Overall metrics)
- technique_comparison.png (Visualization)
- all_detailed_results.csv (Detailed predictions)
- summary_report.txt (Complete analysis)
- Individual technique results in technique-specific subdirectories

## Requirements
The project requires the following packages:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- scipy
- krippendorff
- openai

## Important Notes
- A minimum of 10 examples is required for statistically valid results
- Ensure you have a valid OpenAI API key with sufficient credits
- The default model used is gpt-3.5-turbo-0125, but you can configure other models in the code

<!-- ## References
- Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., & Zhou, D. (2022). Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems.
- Diao, S., Wang, P., Lin, Y., & Zhang, T. (2023). Active prompting with chain-of-thought for large language models. arXiv preprint arXiv:2302.12246.
- Deng, Y., Zhang, W., Chen, Z., & Gu, Q. (2023). Rephrase and respond: Let large language models ask better questions for themselves. arXiv preprint arXiv:2311.04205.```-->

<!-- ## Citation
If you use this code in your research, please cite:

```
@software{course_alignment_evaluation,
  author = {Your Name},
  title = {Course Learning Objective Alignment Evaluation},
  url = {https://github.com/YOUR_USERNAME/course-alignment-evaluation},
  year = {2025},
}
```-->
