# SIGHT Comment Classification Evaluation

This project evaluates different prompting techniques for classifying YouTube comments on educational videos using the SIGHT (Student Insights Gathered from Higher Education Transcripts) dataset. It compares nine advanced LLM prompting strategies to determine which best matches human expert classifications.

## Project Overview

The project classifies YouTube comments from math education videos into 9 categories:
- **General**: General opinions about video content or instructor
- **Confusion**: Math-related questions or pointing out mistakes
- **Pedagogy**: Comments on instructional methods
- **Setup**: Physical teaching setup (audio, video, board quality)
- **Gratitude**: Comments containing thanks
- **Personal Experience**: User's own learning experiences
- **Clarification**: Clarifying misunderstandings (requires @username)
- **Non-English**: Comments not in English
- **NA**: Jokes, trolls, or unrelated content

## Prompting Techniques Evaluated

1. **Zero-shot**: Direct classification using category definitions
2. **Chain of Thought (CoT)**: Step-by-step reasoning before classification
3. **Few-shot**: Provides examples before asking for classification
4. **Active Prompting**: Selects most informative examples using uncertainty sampling
5. **Auto-CoT**: Automatically generates reasoning chains
6. **Contrastive CoT**: Uses positive and negative reasoning
7. **Rephrase and Respond**: Rephrases comment for clarity before classification
8. **Self-Consistency**: Multiple reasoning paths with majority voting
9. **Take a Step Back**: Derives principles before classification

## Directory Structure

```
sight_alignment_evaluation/
├── data/
│   └── sample_dataset.csv      # SIGHT dataset with YouTube comments
├── prompting/
│   ├── zero_shot.py           # Zero-shot prompting
│   ├── cot.py                 # Chain of Thought
│   ├── few_shot.py            # Few-shot with examples
│   ├── active_prompt.py       # Active learning selection
│   ├── auto_cot.py            # Automatic CoT generation
│   ├── contrastive_cot.py     # Contrastive reasoning
│   ├── rephrase_and_respond.py # Clarification approach
│   ├── self_consistency.py    # Multiple sampling
│   └── take_a_step_back.py    # Abstract reasoning
├── utils/
│   ├── data_loader.py         # Loads and processes SIGHT data
│   ├── sight_rubric.py        # Category definitions and examples
│   └── metrics.py             # Evaluation metrics
├── evaluation/
│   ├── evaluate_zero_shot.py
│   ├── evaluate_cot.py
│   ├── evaluate_few_shot.py
│   ├── evaluate_active_prompt.py
│   ├── evaluate_auto_cot.py
│   ├── evaluate_contrastive_cot.py
│   ├── evaluate_rephrase_respond.py
│   ├── evaluate_self_consistency.py
│   └── evaluate_take_step_back.py
├── results/                   # Generated results directory
├── config.py                  # Configuration and API keys
├── main.py                    # Main evaluation script
└── requirements.txt           # Python dependencies
```

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sight-alignment-evaluation.git
cd sight_alignment_evaluation
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Configure your OpenAI API key in `config.py`:
```python
OPENAI_API_KEY = "your-api-key-here"
DATA_PATH = "data/sample_dataset.csv"
```

5. Ensure the SIGHT dataset is in the data directory.

## Usage

### Run Complete Evaluation
```bash
python main.py
```

You'll be prompted to:
1. Enter number of comments to evaluate (recommended: 50-100 for testing)
2. Select which techniques to evaluate or run all

### Run Individual Technique
```bash
python evaluation/evaluate_zero_shot.py
```

## Dataset Format

The SIGHT dataset contains YouTube comments with annotations from two human annotators (H1 and H2). Each comment can belong to multiple categories. The data loader:
- Processes consensus between annotators
- Converts binary labels (0/1) to category lists
- Prepares data for AI classification

## Evaluation Metrics

The project uses 4 key metrics:

1. **Accuracy**: Percentage of exact matches between AI and human labels
2. **Cohen's Kappa (κ)**: Agreement beyond chance (-1 to 1, higher is better)
3. **Krippendorff's Alpha (α)**: Reliability measure (0 to 1)
4. **Intraclass Correlation (ICC)**: Pattern correlation between human and AI

## Output

Results are saved in timestamped directories containing:
- `sight_all_techniques_comparison.csv` - Overall metrics comparison
- `sight_all_techniques_comparison.png` - Visual comparison chart
- `sight_all_detailed_results.csv` - All predictions
- `sight_comprehensive_report.txt` - Detailed analysis
- Individual technique results in subdirectories

## Key Features

- **Multi-label Classification**: Comments can belong to multiple categories
- **Consensus Handling**: Uses agreement between human annotators
- **Fair Evaluation**: AI only sees comments, not human labels
- **Comprehensive Metrics**: Multiple ways to measure performance
- **Flexible Execution**: Run all or selected techniques

## Requirements

- Python 3.7+
- OpenAI API key with GPT-3.5 access
- Required packages: numpy, pandas, matplotlib, seaborn, scikit-learn, scipy, krippendorff, openai

## Notes

- Minimum 10 comments recommended for meaningful results
- Processing time depends on number of comments and techniques selected
- API rate limits are handled automatically

## Citation

<!-- If using the SIGHT dataset, please cite:```-->
```
SIGHT: A Large Annotated Dataset on Student Insights Gathered from Higher Education Transcripts

@inproceedings{wang2023sight,
    title={SIGHT: A Large Annotated Dataset on Student Insights Gathered from Higher Education Transcripts},
    author={Wang, Rose E. and Wirawarn, Pawan and Goodman, Noah and Demszky, Dorottya},
    booktitle={Proceedings of the 18th Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2023)},
    pages={315--351},
    year={2023},
    month={July},
    publisher={Association for Computational Linguistics},
    address={Toronto, Canada},
    url={https://github.com/rosewang2008/sight}
}
```
Additional Note: The SIGHT dataset is intended for research purposes only to promote better understanding of effective pedagogy and student feedback. It follows MIT's Creative Commons License and should not be used for commercial purposes.

