import re
import openai
import random

def create_auto_cot_prompt(course_title, description, learning_objective, examples=None):
    """Create an Auto Chain of Thought prompt with automatically generated reasoning."""
    base_prompt = f"""Rate how well this course covers the learning objective:

Course Title: {course_title}
Description: {description}
Learning Objective: {learning_objective}

SCORING RUBRIC:
5 - Direct Match (core skill): The real-world task is a core learning objective explicitly covered in the course description.
4 - Strongly Related: The task aligns with course content; students should be able to perform it after completion.
3 - Indirectly Related but transferable: Not explicitly covered, but students develop transferable skills.
2 - Unrelated but Within Same Domain: Within the discipline but not relevant to course content.
1 - Completely Unrelated: The task is outside the scope of the course and belongs to a different domain.

Let's think step by step to determine the appropriate rating."""

    if examples:
        # Add auto-generated CoT examples
        example_text = "\n\n".join([
            f"Example {i+1}:\n"
            f"Course: {ex['course']}\n"
            f"Description: {ex['description']}\n"
            f"Learning Objective: {ex['objective']}\n"
            f"Reasoning: {ex['reasoning']}\n"
            f"Rating: {ex['score']}"
            for i, ex in enumerate(examples)
        ])
        
        prompt = f"{base_prompt}\n\nHere are some examples of step-by-step reasoning:\n\n{example_text}\n\nNow, let's analyze the current case:"
    else:
        prompt = base_prompt
    
    return prompt

def generate_auto_cot_examples(client, num_examples=3):
    """Generate automatic chain-of-thought examples."""
    # Sample course-objective pairs for diversity
    sample_pairs = [
        {
            "course": "Introduction to Data Science",
            "description": "This course introduces fundamental concepts in data science, including data collection, cleaning, analysis, and visualization using Python.",
            "objective": "Build predictive models using machine learning algorithms",
        },
        {
            "course": "Advanced Programming Techniques",
            "description": "This course covers design patterns, algorithms, data structures, and optimization techniques for complex software systems.",
            "objective": "Implement efficient sorting algorithms in Java",
        },
        {
            "course": "Digital Marketing Fundamentals",
            "description": "This course covers online marketing strategies including social media, SEO, content marketing, and analytics.",
            "objective": "Create a comprehensive content marketing plan",
        },
        {
            "course": "Financial Accounting",
            "description": "This course covers principles of accounting, financial statements, and basic bookkeeping procedures.",
            "objective": "Calculate return on investment for a business proposal",
        },
        {
            "course": "Human Psychology",
            "description": "This course introduces major theories, research methods, and applications of psychology to understand human behavior.",
            "objective": "Design a basic behavioral experiment",
        }
    ]
    
    # Randomly select pairs for examples
    selected_pairs = random.sample(sample_pairs, min(num_examples, len(sample_pairs)))
    
    examples = []
    for pair in selected_pairs:
        # Generate reasoning with "Let's think step by step" prompt
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": "You are an expert at evaluating how well course descriptions align with learning objectives."},
                    {"role": "user", "content": f"""Rate how well this course covers the learning objective:

Course: {pair['course']}
Description: {pair['description']}
Learning Objective: {pair['objective']}

Let's think step by step:"""}
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            reasoning = response.choices[0].message.content.strip()
            
            # Generate score for this reasoning
            score_response = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": "You are an expert at evaluating how well course descriptions align with learning objectives. Provide only a single digit from 1-5."},
                    {"role": "user", "content": f"""Based on this reasoning, what score (1-5) would you give?

{reasoning}

Score (1-5):"""}
                ],
                temperature=0,
                max_tokens=10
            )
            
            score_text = score_response.choices[0].message.content.strip()
            score_match = re.search(r'[1-5]', score_text)
            if score_match:
                score = int(score_match.group(0))
                examples.append({
                    "course": pair['course'],
                    "description": pair['description'],
                    "objective": pair['objective'],
                    "reasoning": reasoning,
                    "score": score
                })
            
        except Exception as e:
            print(f"Error generating Auto-CoT example: {str(e)}")
            continue
    
    return examples

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

def get_auto_cot_prediction(course_title, description, learning_objective, client, examples_cache=None):
    """Get prediction using Auto Chain of Thought prompting."""
    # Generate CoT examples if not provided
    examples = examples_cache
    if not examples:
        examples = generate_auto_cot_examples(client)
    
    prompt = create_auto_cot_prompt(course_title, description, learning_objective, examples)
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at evaluating how well course descriptions align with learning objectives. Use step-by-step reasoning and provide a single numerical rating (1-5) followed by justification."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=500
        )
        
        return extract_score(response.choices[0].message.content)
        
    except Exception as e:
        print(f"Error getting prediction: {str(e)}")
        return None