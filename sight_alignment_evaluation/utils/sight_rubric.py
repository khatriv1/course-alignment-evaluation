# sight_alignment_evaluation/utils/sight_rubric.py

"""
SIGHT Rubric Implementation - 9 Categories for YouTube Comment Classification
Based on "SIGHT: A Large Annotated Dataset on Student Insights Gathered from Higher Education Transcripts"
"""

class SIGHTRubric:
    """
    SIGHT's 9-category rubric for classifying YouTube comments on educational videos.
    Categories derived from the SIGHT paper's qualitative analysis.
    """
    
    @staticmethod
    def get_category_definitions():
        """Return definitions for all 9 SIGHT categories."""
        return {
            "general": {
                "description": "General/big-picture opinion about video content and/or teaching characteristics",
                "examples": [
                    "Best video I have watched so far, I was with him all the way and my concentration never dipped.",
                    "Amazing lectures!",
                    "Great teacher!"
                ]
            },
            
            "confusion": {
                "description": "Math-related questions, confusion, or pointing out mistakes",
                "examples": [
                    "Can anyone explain what professor meant by pivot variables?",
                    "34:43 why \"directional second derivative\" would not give us a clue of whether it is a min or max?",
                    "At 43:32, shouldn't Professor Strang have put *infinite* as opposed to *1* solutions?"
                ]
            },
            
            "pedagogy": {
                "description": "Comments on instructional methods (examples, applications, proofs, visualizations)",
                "examples": [
                    "From this lecture, I really understand Positive Definite Matrices thanks to Dr. Gilbert Strang. The examples really help me.",
                    "Terrific lecture, esp his way of using linear combination / \"column picture\" to solve equations.",
                    "His teaching style seems casual and intuitive."
                ]
            },
            
            "setup": {
                "description": "Physical teaching setup (chalk, board, microphone, camera)",
                "examples": [
                    "Oh.. my god.. the board and chalk are phenomenal..!",
                    "The mic noise and hiss is distracting in this lecture",
                    "If the camera could move less frequently, the camera is following the instructor too closely"
                ]
            },
            
            "personal_experience": {
                "description": "User's personal math learning/teaching experiences",
                "examples": [
                    "sweet, did this like a term and a half ago in highschool. aced the test for it too :D",
                    "Wish this guy taught me Math 293 and 294 at Cornell.",
                    "I already had this class in college, I keep reading about it"
                ]
            },
            
            "clarification": {
                "description": "Clarifying math misunderstandings (requires @username)",
                "examples": [
                    "@[USERNAME] Actually, if a constant k=1/1m is used, then in the final formula for V you will end up with subtracting m^1 from m^2.",
                    "@[USERNAME] it's the math dragon theorem"
                ]
            },
            
            "gratitude": {
                "description": "Contains 'thanks' or 'thank'",
                "examples": [
                    "Thank you very much! Amazing lectures!",
                    "Thanks! I prepared my high school final exam from this lecture.",
                    "Thank you Prof. Strang!!!"
                ]
            },
            
            "non_english": {
                "description": "Comments not in English",
                "examples": [
                    "Tłumaczenie na polski wymiata",
                    "이게계속쓰지말라던로피탈이구나"
                ]
            },
            
            "na": {
                "description": "Jokes, troll comments, or content unrelated to video",
                "examples": [
                    "sounds drunk on 0.5 speed",
                    "Watching this to make me feel better about college algebra. lol",
                    "i couldnt resist xD"
                ]
            }
        }
    
    @staticmethod
    def get_prompt_descriptions():
        """Return prompt-friendly descriptions for each category."""
        return {
            "general": "The comment expresses a general sentiment/adjective about or expresses a general/big-picture opinion about the video's content and/or about the teaching/professional characteristics of the instructor.",
            
            "confusion": "The comment asks a specific mathematical question and/or points out a mathematical mistake in the video.",
            
            "pedagogy": "The comment mentions the teacher's instructional method, which includes but is not limited to the use of examples, applications, worked out problems, proofs, visualizations, elaboration, and analogies.",
            
            "setup": "The comment mentions the lecture's physical teaching setup, which includes but is not limited to the chalk, board, microphone or audio-related aspects, and camera or camera-related aspects (e.g., angle).",
            
            "personal_experience": "The comment mentions the user's personal experience learning or teaching math on their own outside of watching this lecture/series.",
            
            "clarification": "The comment clarifies someone's math-related misunderstanding or elaborates content from the video, and the comment includes an '@' that is immediately followed by a username.",
            
            "gratitude": "The comment contains the word 'thanks' or 'thank'.",
            
            "non_english": "The comment is not in English.",
            
            "na": "The comment expresses a joke or is a troll comment, and/or the comment says something that is hard to connect to the video content."
        }