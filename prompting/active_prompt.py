import re
import openai
import numpy as np

def create_active_prompt(course_title, description, learning_objective, cot_examples=None):
    """Create an Active Prompt with chain-of-thought examples.
    
    Active Prompting is based on the paper by Diao et al., 2023, which selects the most impactful
    examples for annotation based on uncertainty metrics.
    """
    base_prompt = f"""Rate how well this course covers the learning objective:

Course Title: {course_title}
Description: {description}
Learning Objective: {learning_objective}

SCORING RUBRIC:
5 - Direct Match (core skill): The real-world task is a core learning objective explicitly covered in the course description.
4 - Strongly Related: The task aligns with course content; students should be able to perform it after completion.
3 - Indirectly Related but transferable: Not explicitly covered, but students develop transferable skills.
2 - Unrelated but Within Same Domain: Within the discipline but not relevant to course content.
1 - Completely Unrelated: The task is outside the scope of the course and belongs to a different domain."""

    if cot_examples:
        # Add chain-of-thought examples selected through active selection
        example_text = "\n\n".join([
            f"Example {i+1}:\n"
            f"Course: {ex['course']}\n"
            f"Description: {ex['description']}\n"
            f"Learning Objective: {ex['objective']}\n"
            f"Reasoning: {ex['reasoning']}\n"
            f"Rating: {ex['score']}"
            for i, ex in enumerate(cot_examples)
        ])
        
        prompt = f"{base_prompt}\n\nHere are some examples with step-by-step reasoning:\n\n{example_text}\n\nLet's analyze the current case step by step:"
    else:
        prompt = f"{base_prompt}\n\nLet's analyze this step by step:"
    
    return prompt

def calculate_uncertainty(responses):
    """Calculate uncertainty metrics for Active Prompting selection.
    
    This implements various uncertainty metrics as described in the Active Prompting paper (Diao et al., 2023):
    - Disagreement: Variance in model responses
    - Entropy: Uncertainty in probability distribution
    - Self-confidence: Model's confidence in its answers
    """
    if not responses or len(responses) < 2:
        return {
            "disagreement": 0,
            "entropy": 0,
            "confidence": 1.0  # Default high confidence when we don't have enough data
        }
    
    # Convert responses to numpy array
    scores = np.array([r for r in responses if r is not None])
    
    if len(scores) == 0:
        return {
            "disagreement": 0,
            "entropy": 0,
            "confidence": 1.0
        }
    
    # Disagreement metric (variance)
    disagreement = np.var(scores)
    
    # Calculate probability distribution (simplified)
    unique, counts = np.unique(scores, return_counts=True)
    probs = counts / len(scores)
    
    # Entropy metric
    entropy = -np.sum(probs * np.log2(probs + 1e-10))  # Adding small epsilon to avoid log(0)
    
    # Self-confidence metric (simplified as average variance from most common answer)
    most_common_idx = np.argmax(counts)
    most_common = unique[most_common_idx]
    confidence = 1.0 - np.mean(np.abs(scores - most_common)) / 4.0  # Normalize by max range
    
    return {
        "disagreement": disagreement,
        "entropy": entropy,
        "confidence": confidence
    }

def get_examples_for_active_prompt(example_pool, client, num_examples=3):
    """Select the most informative examples from the pool based on uncertainty metrics.
    
    This implements the active selection strategy from the Active Prompting paper (Diao et al., 2023),
    which selects examples that will be most impactful for the model.
    """
    if len(example_pool) <= num_examples:
        return example_pool
    
    # Sample multiple responses for each example to measure uncertainty
    example_uncertainties = []
    
    for ex in example_pool:
        # Get multiple responses to measure uncertainty
        responses = []
        for _ in range(3):  # Sample 3 responses
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo-0125",
                    messages=[
                        {"role": "system", "content": "You are an expert at evaluating how well course descriptions align with learning objectives. Provide a numerical rating from 1-5 based on the rubric."},
                        {"role": "user", "content": f"""Rate how well this course covers the learning objective:
                        
Course: {ex['course']}
Description: {ex['description']}
Learning Objective: {ex['objective']}

SCORING RUBRIC:
5 - Direct Match (core skill): The real-world task is a core learning objective explicitly covered in the course description.
4 - Strongly Related: The task aligns with course content; students should be able to perform it after completion.
3 - Indirectly Related but transferable: Not explicitly covered, but students develop transferable skills.
2 - Unrelated but Within Same Domain: Within the discipline but not relevant to course content.
1 - Completely Unrelated: The task is outside the scope of the course and belongs to a different domain.

Rating (1-5):"""}
                    ],
                    temperature=0.7,  # Higher temperature for diversity in responses
                    max_tokens=10
                )
                
                # Extract score
                score_text = response.choices[0].message.content.strip()
                score_match = re.search(r'[1-5]', score_text)
                if score_match:
                    responses.append(int(score_match.group(0)))
            except Exception as e:
                print(f"Error sampling responses: {str(e)}")
                continue
        
        # Calculate uncertainty metrics
        uncertainty = calculate_uncertainty(responses)
        
        # Store example with its uncertainty metrics
        example_uncertainties.append({
            "example": ex,
            "metrics": uncertainty
        })
    
    # Sort examples by uncertainty (using disagreement as primary metric)
    sorted_examples = sorted(example_uncertainties, key=lambda x: x["metrics"]["disagreement"], reverse=True)
    
    # Return the top N most uncertain examples
    return [item["example"] for item in sorted_examples[:num_examples]]

def extract_score(response):
    """Extract numerical score from model response."""
    if not response:
        return None
        
    # First try to find a standalone digit at the beginning
    first_line = response.strip().split('\n')[0].strip()
    if first_line.isdigit() and 1 <= int(first_line) <= 5:
        return int(first_line)
    
    # Then try to find patterns like "Rating: 4" or "Score: 3"
    rating_pattern = r'(?:rating|score):\s*([1-5])'
    matches = re.search(rating_pattern, response.lower())
    if matches:
        return int(matches.group(1))
    
    # Finally, just find any digit 1-5
    matches = re.findall(r'[1-5]', response)
    if matches:
        return int(matches[0])
        
    return None

def get_active_prompt_prediction(course_title, description, learning_objective, client, example_pool=None):
    """Get prediction using Active Prompting technique.
    
    This implements the Active Prompting approach from Diao et al., 2023,
    which selects the most informative chain-of-thought examples based on uncertainty metrics.
    """
    # Generate or use provided example pool
    if not example_pool:
        # Create a default example pool (in real implementation, this would be more diverse)
        example_pool = [
            {
                "course": "Introduction to Data Science",
                "description": "This course introduces fundamental concepts in data science, including data collection, cleaning, analysis, and visualization using Python.",
                "objective": "Build predictive models using machine learning algorithms",
                "reasoning": "The course explicitly covers data analysis, which is a foundation for building predictive models. Although machine learning algorithms aren't specifically mentioned, the course's focus on data analysis strongly suggests that predictive modeling would be covered as part of the data analysis components.",
                "score": 4
            },
            {
                "course": "Advanced Programming Techniques",
                "description": "This course covers design patterns, algorithms, data structures, and optimization techniques for complex software systems.",
                "objective": "Implement efficient sorting algorithms in Java",
                "reasoning": "The course explicitly mentions algorithms and optimization techniques, which would include sorting algorithms. While Java isn't specifically mentioned, the course is clearly focused on programming techniques that would be applicable to Java or any other language. The core skill is directly related to the course content.",
                "score": 5
            },
            {
                "course": "Database Management",
                "description": "This course covers relational database design, SQL programming, and database administration techniques.",
                "objective": "Write queries to extract and manipulate data in SQL",
                "reasoning": "The course description explicitly mentions SQL programming, making this learning objective a direct match. Writing queries to extract and manipulate data is a core component of SQL programming mentioned in the course description.",
                "score": 5
            },
            {
                "course": "Digital Marketing Fundamentals",
                "description": "This course covers online marketing strategies including social media, SEO, content marketing, and analytics.",
                "objective": "Create a comprehensive content marketing plan",
                "reasoning": "The course explicitly mentions content marketing as one of the online marketing strategies covered. Creating a comprehensive content marketing plan would be a direct application of content marketing knowledge, making this strongly related to the course content.",
                "score": 4
            },
            {
                "course": "Financial Accounting",
                "description": "This course covers principles of accounting, financial statements, and basic bookkeeping procedures.",
                "objective": "Calculate return on investment for a business proposal",
                "reasoning": "The course covers financial statements and accounting principles, which provide a foundation for financial analysis. However, ROI calculations for business proposals are not explicitly mentioned in the description. This would be a transferable skill from the financial knowledge gained in the course.",
                "score": 3
            }
        ]
    
    # Select the most informative examples using active learning principles
    selected_examples = get_examples_for_active_prompt(example_pool, client)
    
    # Create the prompt with the actively selected examples
    prompt = create_active_prompt(course_title, description, learning_objective, selected_examples)
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at evaluating how well course descriptions align with learning objectives. Use step-by-step reasoning to analyze the alignment and provide a single numerical rating (1-5) followed by justification."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=500
        )
        
        return extract_score(response.choices[0].message.content)
        
    except Exception as e:
        print(f"Error getting prediction: {str(e)}")
        return None