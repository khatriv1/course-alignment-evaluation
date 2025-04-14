# Course Learning Objective Alignment Evaluation

This project evaluates the alignment between course descriptions and learning objectives using three different LLM prompting techniques: Zero-shot, Few-shot, and Chain of Thought.

## Project Overview
The project compares three prompting techniques:
- Zero-shot: Direct prompting with rubric
- Few-shot: Prompting with examples
- Chain of Thought: Step-by-step reasoning

## Directory Structure
```
course_alignment_evaluation/
├── data/
│   └── humanscore.csv
├── prompting/
│   ├── zero_shot.py
│   ├── few_shot.py
│   └── cot.py
├── utils/
│   └── metrics.py
├── evaluation/
│   ├── evaluate_zero_shot.py
│   ├── evaluate_few_shot.py
│   └── evaluate_cot.py
├── config.py
└── main.py
```

## Setup Instructions
1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/course-alignment-evaluation.git
cd course-alignment-evaluation
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Create `config.py` with your OpenAI API key (see config_example.py)

4. Place your dataset in the data directory

## Usage
Run individual evaluations:
```bash
python evaluation/evaluate_zero_shot.py
python evaluation/evaluate_few_shot.py
python evaluation/evaluate_cot.py
```

Or run all evaluations:
```bash
python main.py
```

## Output
The program generates:
- Individual technique evaluations
- Comparison visualizations
- Detailed results in CSV format
- Summary report with metrics (κ, α, ICC)

## Results Format
Results are saved in a timestamped directory containing:
- technique_comparison.csv
- technique_comparison.png
- all_detailed_results.csv
- summary_report.txt