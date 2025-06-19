# sight_alignment_evaluation/prompting/zero_shot.py

"""
Zero-shot prompting for SIGHT comment classification.
Classifies YouTube comments into SIGHT's 9 categories without examples.
"""

import time
from typing import List, Optional
from utils.sight_rubric import SIGHTRubric

def get_zero_shot_prediction(comment: str, 
                           video_name: str, 
                           playlist_name: str,
                           client,
                           category: str) -> Optional[bool]:
    """
    Get zero-shot prediction for a single category using SIGHT rubric.
    
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
    
    # Create zero-shot prompt
    prompt = f"""Consider a YouTube comment from a math education video below:

Playlist name: {playlist_name}
Video name: {video_name}
Comment: {comment}

If the statement below is true, please respond "true"; otherwise, please respond "false":

{prompt_descriptions[category]}"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing educational video comments. Respond only with 'true' or 'false'."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=10
        )
        
        result = response.choices[0].message.content.strip().lower()
        
        if result == "true":
            return True
        elif result == "false":
            return False
        else:
            print(f"Unexpected response for {category}: {result}")
            return None
            
    except Exception as e:
        print(f"Error getting prediction for {category}: {str(e)}")
        return None


def get_zero_shot_prediction_all_categories(comment: str,
                                          video_name: str, 
                                          playlist_name: str,
                                          client) -> List[str]:
    """
    Get zero-shot predictions for all SIGHT categories.
    
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
        prediction = get_zero_shot_prediction(
            comment, video_name, playlist_name, client, category
        )
        
        if prediction is True:
            assigned_categories.append(category)
        
        # Rate limiting
        time.sleep(0.5)
    
    return assigned_categories