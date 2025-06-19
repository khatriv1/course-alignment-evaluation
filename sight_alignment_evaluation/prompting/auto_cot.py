# sight_alignment_evaluation/prompting/auto_cot.py

"""
Auto-CoT (Automatic Chain of Thought) prompting for SIGHT comment classification.
Automatically generates reasoning chains for classification.
"""

import time
from typing import List, Optional, Dict
from utils.sight_rubric import SIGHTRubric

def generate_auto_cot_examples(category: str) -> str:
    """Generate automatic reasoning chains for examples."""
    reasoning_examples = {
        'general': [
            {
                "comment": "This is the best math lecture I've ever watched!",
                "reasoning": "Let's think step by step. The comment expresses a very positive opinion ('best...ever') about the lecture as a whole. It's not about specific math content or teaching methods, but a general sentiment. This matches the 'general' category.",
                "answer": "true"
            },
            {
                "comment": "What time does the cafeteria open?",
                "reasoning": "Let's think step by step. This comment is asking about cafeteria hours, which has nothing to do with the math video content or teaching. It doesn't express any opinion about the video. This doesn't match the 'general' category.",
                "answer": "false"
            }
        ],
        'confusion': [
            {
                "comment": "I don't understand how you got x=5 from that equation",
                "reasoning": "Let's think step by step. The commenter explicitly states they don't understand something mathematical ('how you got x=5'). They're confused about a specific mathematical step. This is asking for clarification about math content.",
                "answer": "true"
            },
            {
                "comment": "Great explanation, very clear!",
                "reasoning": "Let's think step by step. This comment expresses positive feedback about clarity. There's no confusion or question being asked. The commenter understands the content well.",
                "answer": "false"
            }
        ],
        'pedagogy': [
            {
                "comment": "Using color coding for different variables really helps visualize the problem",
                "reasoning": "Let's think step by step. The comment specifically mentions an instructional method (color coding) and how it helps with understanding. This is feedback about the teaching technique used.",
                "answer": "true"
            },
            {
                "comment": "I love mathematics!",
                "reasoning": "Let's think step by step. This is a general expression of enthusiasm for math, not a comment about teaching methods or instructional techniques. It doesn't mention how anything is taught.",
                "answer": "false"
            }
        ]
    }
    
    # Get examples for the category
    examples = reasoning_examples.get(category, [])
    
    # Format with reasoning chains
    formatted = []
    for ex in examples:
        formatted.append(f"Comment: {ex['comment']}\n{ex['reasoning']}\nAnswer: {ex['answer']}")
    
    return "\n\n".join(formatted)

def get_auto_cot_prediction(comment: str,
                          video_name: str,
                          playlist_name: str, 
                          client,
                          category: str) -> Optional[bool]:
    """
    Get Auto-CoT prediction for a single category.
    
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
    
    # Get auto-generated CoT examples
    cot_examples = generate_auto_cot_examples(category)
    
    # Create Auto-CoT prompt
    prompt = f"""You are classifying YouTube comments from educational math videos.

Category: {category}
Definition: {prompt_descriptions[category]}

Here are some examples with reasoning:

{cot_examples}

Now classify this comment using the same step-by-step reasoning:

Playlist name: {playlist_name}
Video name: {video_name}
Comment: {comment}

Let's think step by step."""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing educational video comments. Always think step by step and end with 'Answer: true' or 'Answer: false'."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=200
        )
        
        result = response.choices[0].message.content.strip().lower()
        
        # Extract answer from reasoning
        if "answer: true" in result or "answer is true" in result or result.endswith("true"):
            return True
        elif "answer: false" in result or "answer is false" in result or result.endswith("false"):
            return False
        else:
            print(f"Could not extract answer from Auto-CoT response for {category}: {result}")
            return None
            
    except Exception as e:
        print(f"Error getting Auto-CoT prediction for {category}: {str(e)}")
        return None


def get_auto_cot_prediction_all_categories(comment: str,
                                         video_name: str,
                                         playlist_name: str,
                                         client) -> List[str]:
    """
    Get Auto-CoT predictions for all SIGHT categories.
    
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
        prediction = get_auto_cot_prediction(
            comment, video_name, playlist_name, client, category
        )
        
        if prediction is True:
            assigned_categories.append(category)
        
        # Rate limiting
        time.sleep(0.5)
    
    return assigned_categories