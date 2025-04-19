import re
import openai
import numpy as np

def create_self_consistency_prompt(course_title, description, learning_objective):
    """Create a prompt for self-consistency prompting."""
    return f"""Rate how well this course covers the learning objective:

Course Title: {course_title}
Description: {description}
Learning Objective: {learning_objective}

SCORING RUBRIC:
5 - Direct Match (core skill): The real-world task is a core learning objective explicitly covered in the course description.
4 - Strongly Related: The task aligns with course content; students should be able to perform it after completion.
3 - Indirectly Related but transferable: Not explicitly covered, but students develop transferable skills.
2 - Unrelated but Within Same Domain: Within the discipline but not relevant to course content.
1 - Completely Unrelated: The task is outside the scope of the course and belongs to a different domain.

Let's think about this step by step and assign a rating according to the rubric."""

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

def get_self_consistency_prediction(course_title, description, learning_objective, client, num_samples=5):
    """Get prediction using self-consistency prompting with multiple samples."""
    prompt = create_self_consistency_prompt(course_title, description, learning_objective)
    
    scores = []
    try:
        # Generate multiple responses
        for _ in range(num_samples):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": "You are an expert at evaluating how well course descriptions align with learning objectives. Always provide a single numerical rating (1-5) followed by justification."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,  # Use higher temperature for diverse reasoning paths
                max_tokens=300
            )
            
            score = extract_score(response.choices[0].message.content)
            if score is not None:
                scores.append(score)
        
        if not scores:
            return None
            
        # Return most common score (majority vote)
        return int(np.round(np.median(scores)))
        
    except Exception as e:
        print(f"Error getting prediction: {str(e)}")
        return None