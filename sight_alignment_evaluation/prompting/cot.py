# sight_alignment_evaluation/prompting/cot.py

"""
Chain of Thought prompting for SIGHT comment classification.
Uses step-by-step reasoning before making classification decisions.
"""

import time
from typing import List, Optional
from utils.sight_rubric import SIGHTRubric

def get_cot_prediction(comment: str,
                      video_name: str,
                      playlist_name: str, 
                      client,
                      category: str) -> Optional[bool]:
    """
    Get Chain of Thought prediction for a single category.
    
    Args:
        comment: The YouTube comment text
        video_name: Name of the video
        playlist_name: Course playlist name
        client: OpenAI client
        category: Category to classify for
    
    Returns:
        Boolean indicating if comment belongs to category, None if failed
    """
    rubric = SIGHTRubric()
    prompt_descriptions = rubric.get_prompt_descriptions()
    
    if category not in prompt_descriptions:
        raise ValueError(f"Unknown category: {category}")
    
    # Create Chain of Thought prompt with reasoning steps
    prompt = f"""Consider a YouTube comment from a math education video below:

Playlist name: {playlist_name}
Video name: {video_name}
Comment: {comment}

Please analyze this comment step by step to determine if it fits this category:

Category: {category}
Definition: {prompt_descriptions[category]}

Step 1: What is the main content/purpose of this comment?
Step 2: Does this comment fit the definition of the {category} category?
Step 3: What specific words or phrases support your decision?

Based on your step-by-step analysis, respond with "true" if the comment belongs to the {category} category, or "false" if it does not."""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing educational video comments. Think step by step and end your response with either 'true' or 'false'."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=200
        )
        
        result = response.choices[0].message.content.strip().lower()
        
        # Extract the final true/false from the reasoning
        if "true" in result.split()[-5:]:  # Check last few words
            return True
        elif "false" in result.split()[-5:]:
            return False
        else:
            print(f"Could not extract true/false from CoT response for {category}: {result}")
            return None
            
    except Exception as e:
        print(f"Error getting CoT prediction for {category}: {str(e)}")
        return None


def get_cot_prediction_all_categories(comment: str,
                                    video_name: str,
                                    playlist_name: str,
                                    client) -> List[str]:
    """
    Get Chain of Thought predictions for all SIGHT categories.
    
    Args:
        comment: The YouTube comment text
        video_name: Name of the video
        playlist_name: Course playlist name
        client: OpenAI client
    
    Returns:
        List of categories assigned to the comment
    """
    categories = [
        'general', 'confusion', 'pedagogy', 'setup', 'gratitude',
        'personal_experience', 'clarification', 'non_english', 'na'
    ]
    
    assigned_categories = []
    
    for category in categories:
        prediction = get_cot_prediction(
            comment, video_name, playlist_name, client, category
        )
        
        if prediction is True:
            assigned_categories.append(category)
        
        # Rate limiting
        time.sleep(0.5)
    
    return assigned_categories