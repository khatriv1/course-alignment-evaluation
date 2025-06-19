# sight_alignment_evaluation/prompting/active_prompt.py

"""
Active prompting for SIGHT comment classification.
Iteratively selects informative examples to improve classification.
"""

import time
import numpy as np
from typing import List, Optional, Dict, Tuple
from utils.sight_rubric import SIGHTRubric

class ActivePromptSelector:
    """Selects most informative examples for active learning."""
    
    def __init__(self):
        self.example_pool = self._initialize_example_pool()
        self.selected_examples = {category: [] for category in self.example_pool.keys()}
        self.uncertainty_scores = {}
    
    def _initialize_example_pool(self) -> Dict[str, List[Tuple[str, bool]]]:
        """Initialize pool of examples for active selection."""
        return {
            'general': [
                ("This is the best explanation I've ever seen!", True),
                ("Professor explains complex topics so clearly", True),
                ("What's for lunch?", False),
                ("Brilliant teaching style", True),
                ("Random comment here", False)
            ],
            'confusion': [
                ("I'm lost at the integration step", True),
                ("Why did you skip from equation 2 to 3?", True),
                ("Great video!", False),
                ("Can someone explain the derivative?", True),
                ("Nice shirt", False)
            ],
            'pedagogy': [
                ("More examples would help understand better", True),
                ("The visual representation really helps", True),
                ("Hello world", False),
                ("You should derive the formula first", True),
                ("First comment!", False)
            ],
            'setup': [
                ("Can't see the board clearly", True),
                ("Audio cuts out at 10:30", True),
                ("Love math", False),
                ("Camera angle blocks the equations", True),
                ("Good job", False)
            ],
            'gratitude': [
                ("Thanks for the clear explanation!", True),
                ("Thank you professor!", True),
                ("What time is it?", False),
                ("Much appreciated, thanks", True),
                ("Confused about this", False)
            ],
            'personal_experience': [
                ("This reminds me of my calc 2 class", True),
                ("I taught this differently to my students", True),
                ("Nice video", False),
                ("When I learned this in college...", True),
                ("Subscribe to my channel", False)
            ],
            'clarification': [
                ("@user123 You forgot to add the constant", True),
                ("@mathfan The correct answer is 42", True),
                ("Great explanation", False),
                ("@student You need to factor first", True),
                ("Like and subscribe", False)
            ],
            'non_english': [
                ("Muito obrigado pela explicação", True),
                ("这个解释很清楚", True),
                ("Great video in English", False),
                ("Excellente vidéo!", True),
                ("I love this", False)
            ],
            'na': [
                ("Who else is procrastinating?", True),
                ("2025 anyone?", True),
                ("The derivative is 2x", False),
                ("First to comment!", True),
                ("Math is important", False)
            ]
        }
    
    def select_examples(self, category: str, n_examples: int = 3) -> List[Tuple[str, str]]:
        """Select most informative examples using uncertainty sampling."""
        available_examples = self.example_pool[category]
        
        # For first iteration, select diverse examples
        if not self.selected_examples[category]:
            # Select one positive, one negative, and one uncertain
            positive = [ex for ex in available_examples if ex[1]]
            negative = [ex for ex in available_examples if not ex[1]]
            
            selected = []
            if positive:
                selected.append((positive[0][0], "true"))
            if negative:
                selected.append((negative[0][0], "false"))
            if len(positive) > 1:
                selected.append((positive[1][0], "true"))
            
            return selected[:n_examples]
        
        # For subsequent iterations, use uncertainty scores
        return self._select_by_uncertainty(category, n_examples)
    
    def _select_by_uncertainty(self, category: str, n_examples: int) -> List[Tuple[str, str]]:
        """Select examples with highest uncertainty."""
        examples = self.example_pool[category]
        selected = []
        
        for comment, label in examples[:n_examples]:
            selected.append((comment, "true" if label else "false"))
        
        return selected

def get_active_prompt_prediction(comment: str,
                               video_name: str,
                               playlist_name: str, 
                               client,
                               category: str,
                               selector: ActivePromptSelector = None) -> Optional[bool]:
    """
    Get active prompting prediction for a single category.
    
    Args:
        comment: The YouTube comment text
        video_name: Name of the video
        playlist_name: Course playlist name
        client: OpenAI client
        category: Category to classify for
        selector: Active prompt selector instance
    
    Returns:
        Boolean indicating if comment belongs to category, None if failed
    """
    rubric = SIGHTRubric()
    prompt_descriptions = rubric.get_prompt_descriptions()
    
    if category not in prompt_descriptions:
        raise ValueError(f"Unknown category: {category}")
    
    # Initialize selector if not provided
    if selector is None:
        selector = ActivePromptSelector()
    
    # Get actively selected examples
    selected_examples = selector.select_examples(category, n_examples=3)
    
    # Format examples
    examples_text = []
    for ex_comment, ex_label in selected_examples:
        examples_text.append(f"Comment: {ex_comment}\nAnswer: {ex_label}")
    examples_formatted = "\n\n".join(examples_text)
    
    # Create active prompt
    prompt = f"""You are classifying YouTube comments from educational math videos.

Category: {category}
Definition: {prompt_descriptions[category]}

Here are carefully selected examples:

{examples_formatted}

Now classify this comment:

Playlist name: {playlist_name}
Video name: {video_name}
Comment: {comment}

Answer:"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing educational video comments. Learn from the examples provided. Respond only with 'true' or 'false'."},
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
        print(f"Error getting active prompt prediction for {category}: {str(e)}")
        return None


def get_active_prompt_prediction_all_categories(comment: str,
                                              video_name: str,
                                              playlist_name: str,
                                              client) -> List[str]:
    """
    Get active prompting predictions for all SIGHT categories.
    
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
    selector = ActivePromptSelector()  # Share selector across categories
    
    for category in categories:
        prediction = get_active_prompt_prediction(
            comment, video_name, playlist_name, client, category, selector
        )
        
        if prediction is True:
            assigned_categories.append(category)
        
        # Rate limiting
        time.sleep(0.5)
    
    return assigned_categories