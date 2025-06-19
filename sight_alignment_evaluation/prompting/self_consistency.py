# sight_alignment_evaluation/prompting/self_consistency.py

"""
Self-Consistency prompting for SIGHT comment classification.
Samples multiple reasoning paths and takes majority vote.
"""

import time
from typing import List, Optional, Dict
from collections import Counter
from utils.sight_rubric import SIGHTRubric

def get_single_reasoning_path(comment: str,
                            video_name: str,
                            playlist_name: str,
                            client,
                            category: str,
                            temperature: float = 0.7) -> Optional[bool]:
    """
    Get a single reasoning path for classification.
    
    Args:
        comment: The YouTube comment text
        video_name: Name of the video
        playlist_name: Course playlist name
        client: OpenAI client
        category: Category to classify for
        temperature: Sampling temperature for diversity
    
    Returns:
        Boolean prediction or None if failed
    """
    rubric = SIGHTRubric()
    prompt_descriptions = rubric.get_prompt_descriptions()
    
    prompt = f"""Consider a YouTube comment from a math education video:

Playlist name: {playlist_name}
Video name: {video_name}
Comment: {comment}

Category: {category}
Definition: {prompt_descriptions[category]}

Think through this step-by-step and explain your reasoning:
1. What is the main point of this comment?
2. How does it relate to the category definition?
3. What specific aspects make it fit or not fit?

Based on your analysis, does this comment belong to the '{category}' category? Answer 'true' or 'false'."""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing educational video comments. Provide your reasoning and end with 'true' or 'false'."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=200
        )
        
        result = response.choices[0].message.content.strip().lower()
        
        # Extract answer
        if "true" in result.split()[-5:]:
            return True
        elif "false" in result.split()[-5:]:
            return False
        else:
            return None
            
    except Exception as e:
        print(f"Error in reasoning path: {str(e)}")
        return None

def get_self_consistency_prediction(comment: str,
                                  video_name: str,
                                  playlist_name: str, 
                                  client,
                                  category: str,
                                  n_samples: int = 5) -> Optional[bool]:
    """
    Get Self-Consistency prediction using multiple reasoning paths.
    
    Args:
        comment: The YouTube comment text
        video_name: Name of the video
        playlist_name: Course playlist name
        client: OpenAI client
        category: Category to classify for
        n_samples: Number of reasoning paths to sample
    
    Returns:
        Boolean indicating if comment belongs to category, None if failed
    """
    rubric = SIGHTRubric()
    prompt_descriptions = rubric.get_prompt_descriptions()
    
    if category not in prompt_descriptions:
        raise ValueError(f"Unknown category: {category}")
    
    # Collect predictions from multiple reasoning paths
    predictions = []
    
    for i in range(n_samples):
        # Vary temperature for diversity
        temp = 0.5 + (i * 0.1)  # 0.5, 0.6, 0.7, 0.8, 0.9
        
        prediction = get_single_reasoning_path(
            comment, video_name, playlist_name, client, category, temp
        )
        
        if prediction is not None:
            predictions.append(prediction)
        
        # Small delay between samples
        time.sleep(0.2)
    
    if not predictions:
        print(f"No valid predictions obtained for {category}")
        return None
    
    # Take majority vote
    vote_counts = Counter(predictions)
    majority_vote = vote_counts.most_common(1)[0][0]
    confidence = vote_counts[majority_vote] / len(predictions)
    
    print(f"Self-consistency for {category}: {vote_counts}, confidence: {confidence:.2f}")
    
    return majority_vote


def get_self_consistency_prediction_all_categories(comment: str,
                                                 video_name: str,
                                                 playlist_name: str,
                                                 client,
                                                 n_samples: int = 5) -> List[str]:
    """
    Get Self-Consistency predictions for all SIGHT categories.
    
    Args:
        comment: The YouTube comment text
        video_name: Name of the video
        playlist_name: Course playlist name
        client: OpenAI client
        n_samples: Number of reasoning paths per category
    
    Returns:
        List of categories assigned to the comment
    """
    categories = [
        'general', 'confusion', 'pedagogy', 'setup', 'gratitude',
        'personal_experience', 'clarification', 'non_english', 'na'
    ]
    
    assigned_categories = []
    
    for category in categories:
        prediction = get_self_consistency_prediction(
            comment, video_name, playlist_name, client, category, n_samples
        )
        
        if prediction is True:
            assigned_categories.append(category)
        
        # Rate limiting between categories
        time.sleep(0.5)
    
    return assigned_categories