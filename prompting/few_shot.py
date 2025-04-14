import re
import openai

def get_example_pairs():
    """Get example pairs for few-shot prompting."""
    return [
        {
            "course": "Introduction to Python Programming",
            "description": "This course covers the basics of Python programming, including data types, control structures, functions, and file handling.",
            "objective": "Write basic Python programs using control structures and functions",
            "score": 5,
            "explanation": "Direct Match - The learning objective is explicitly covered in the course description which mentions Python programming, control structures, and functions."
        },
        {
            "course": "Advanced Data Analysis",
            "description": "This course covers statistical methods for analyzing large datasets, including regression, clustering, and hypothesis testing.",
            "objective": "Create visualizations in Tableau",
            "score": 2,
            "explanation": "Unrelated but Within Domain - While both are in the data field, the course focuses on statistical methods and doesn't cover Tableau or visualizations."
        },
        {
            "course": "Business Communication",
            "description": "This course teaches effective business writing, presentation skills, and interpersonal communication in professional settings.",
            "objective": "Develop confidence when speaking in group discussions",
            "score": 4,
            "explanation": "Strongly Related - While not explicitly mentioned, presentation skills and interpersonal communication would directly help with speaking in group discussions."
        }
    ]

def create_few_shot_prompt(course_title, description, learning_objective):
    """Create a few-shot prompt with examples."""
    examples = get_example_pairs()
    examples_text = "\n\n".join([
        f"Example {i+1}:\n"
        f"Course: {ex['course']}\n"
        f"Description: {ex['description']}\n"
        f"Learning Objective: {ex['objective']}\n"
        f"Rating: {ex['score']} - {ex['explanation']}"
        for i, ex in enumerate(examples)
    ])
    
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

Here are some examples:

{examples_text}

Now rate the course and learning objective provided above. Provide a single numerical rating (1-5) on the first line, followed by your justification."""

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

def get_few_shot_prediction(course_title, description, learning_objective, client):
    """Get prediction using few-shot prompting."""
    prompt = create_few_shot_prompt(course_title, description, learning_objective)
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at evaluating how well course descriptions align with learning objectives. Always provide a single numerical rating (1-5) followed by justification based on the official rubric."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=300
        )
        
        return extract_score(response.choices[0].message.content)
        
    except Exception as e:
        print(f"Error getting prediction: {str(e)}")
        return None