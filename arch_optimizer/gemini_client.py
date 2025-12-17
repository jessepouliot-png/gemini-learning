#!/usr/bin/env python3
"""
Gemini AI Integration Module

This module handles communication with Google's Gemini AI API for
analyzing Arch Linux system information and generating optimization
recommendations.
"""

import os
import json
from typing import Dict, Any, Optional, List
import google.generativeai as genai
from dotenv import load_dotenv


class GeminiClient:
    """Client for interacting with Google's Gemini AI API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        """
        Initialize the Gemini client.
        
        Args:
            api_key: Google Gemini API key. If None, will try to load from environment.
            model: Model name to use (default: gemini-1.5-flash)
        """
        # Load environment variables
        load_dotenv()
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Gemini API key not provided. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model_name = model
        self.model = genai.GenerativeModel(model)
        
        # Generation configuration
        self.generation_config = {
            'temperature': 0.7,
            'top_p': 0.95,
            'top_k': 40,
            'max_output_tokens': 8192,
        }
        
        # Safety settings (for production use)
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]

    def create_analysis_prompt(
        self, 
        system_info: Dict[str, Any],
        focus_areas: Optional[List[str]] = None,
        analysis_depth: str = "standard"
    ) -> str:
        """
        Create a prompt for system analysis.
        
        Args:
            system_info: Dictionary containing system information
            focus_areas: List of specific areas to focus on
            analysis_depth: Level of analysis detail (basic, standard, deep)
            
        Returns:
            Formatted prompt string
        """
        focus_areas = focus_areas or [
            'performance', 'memory', 'disk', 'services', 
            'packages', 'boot_time', 'security'
        ]
        
        prompt = f"""You are an expert Arch Linux system administrator and performance optimization specialist. 
I need you to analyze the following system information and provide optimization recommendations.

SYSTEM INFORMATION:
{json.dumps(system_info, indent=2, default=str)}

ANALYSIS REQUIREMENTS:
1. Focus Areas: {', '.join(focus_areas)}
2. Analysis Depth: {analysis_depth}
3. Safety: All recommendations must be safe and reversible

TASK:
Analyze this Arch Linux system and provide detailed optimization recommendations. For each recommendation:
1. Categorize by impact level (HIGH, MEDIUM, LOW)
2. Explain the issue or opportunity
3. Provide specific commands or steps to implement
4. Include any warnings or precautions
5. Explain the expected benefit

SPECIFIC AREAS TO ANALYZE:
"""
        
        if 'performance' in focus_areas:
            prompt += "\n- Performance bottlenecks (CPU, I/O, etc.)"
        if 'memory' in focus_areas:
            prompt += "\n- Memory usage optimization (RAM and swap)"
        if 'disk' in focus_areas:
            prompt += "\n- Disk space and I/O optimization"
        if 'services' in focus_areas:
            prompt += "\n- Unnecessary services running at startup"
        if 'packages' in focus_areas:
            prompt += "\n- Package optimization (orphaned packages, alternatives)"
        if 'boot_time' in focus_areas:
            prompt += "\n- Boot time improvements"
        if 'security' in focus_areas:
            prompt += "\n- Security recommendations specific to Arch Linux"
        
        prompt += """

OUTPUT FORMAT:
Please structure your response as follows:

# SYSTEM OVERVIEW
[Brief overview of the system state]

# RECOMMENDATIONS

## HIGH IMPACT
[High priority recommendations with immediate benefits]

## MEDIUM IMPACT
[Medium priority recommendations with moderate benefits]

## LOW IMPACT
[Low priority recommendations with minor benefits]

# WARNINGS
[Any important warnings or considerations]

# SUMMARY
[Brief summary of key findings and recommendations]

Please be specific with commands and configuration changes. Remember that all recommendations 
should be safe, reversible, and appropriate for an Arch Linux system.
"""
        
        return prompt

    def analyze_system(
        self,
        system_info: Dict[str, Any],
        focus_areas: Optional[List[str]] = None,
        analysis_depth: str = "standard"
    ) -> Dict[str, Any]:
        """
        Analyze system information using Gemini AI.
        
        Args:
            system_info: Dictionary containing system information
            focus_areas: List of specific areas to focus on
            analysis_depth: Level of analysis detail
            
        Returns:
            Dictionary containing analysis results and recommendations
        """
        try:
            # Create the prompt
            prompt = self.create_analysis_prompt(system_info, focus_areas, analysis_depth)
            
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            
            # Extract the text response
            if not response.candidates:
                return {
                    'success': False,
                    'error': 'No response candidates returned from Gemini'
                }
            
            recommendations = response.text
            
            return {
                'success': True,
                'recommendations': recommendations,
                'model': self.model_name,
                'prompt_length': len(prompt),
                'response_length': len(recommendations)
            }
            
        except TimeoutError as e:
            return {
                'success': False,
                'error': f'Request timeout: {str(e)}. Please try again.'
            }
        except ConnectionError as e:
            return {
                'success': False,
                'error': f'Network connection error: {str(e)}. Check your internet connection.'
            }
        except ValueError as e:
            return {
                'success': False,
                'error': f'Invalid request or API key: {str(e)}'
            }
        except Exception as e:
            error_msg = str(e)
            # Check for common API errors
            if 'API key' in error_msg:
                return {
                    'success': False,
                    'error': f'API key error: {error_msg}'
                }
            elif 'quota' in error_msg.lower() or 'limit' in error_msg.lower():
                return {
                    'success': False,
                    'error': f'API quota or rate limit exceeded: {error_msg}'
                }
            elif 'safety' in error_msg.lower():
                return {
                    'success': False,
                    'error': f'Content safety filter triggered: {error_msg}'
                }
            else:
                return {
                    'success': False,
                    'error': f'Unexpected error: {error_msg}'
                }

    def parse_recommendations(self, analysis_result: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Parse the recommendations from the analysis result.
        
        Args:
            analysis_result: Result from analyze_system()
            
        Returns:
            Dictionary categorizing recommendations by impact level
        """
        if not analysis_result.get('success'):
            return {
                'error': analysis_result.get('error', 'Analysis failed'),
                'high': [],
                'medium': [],
                'low': []
            }
        
        recommendations_text = analysis_result.get('recommendations', '')
        
        # Simple parsing - split by impact level headers
        parsed = {
            'high': [],
            'medium': [],
            'low': [],
            'warnings': [],
            'overview': '',
            'summary': ''
        }
        
        # Extract sections
        lines = recommendations_text.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line_upper = line.upper()
            
            if '# SYSTEM OVERVIEW' in line_upper or 'SYSTEM OVERVIEW' in line_upper:
                if current_section:
                    parsed[current_section].append('\n'.join(current_content))
                current_section = 'overview'
                current_content = []
            elif 'HIGH IMPACT' in line_upper:
                if current_section:
                    parsed[current_section] = '\n'.join(current_content)
                current_section = 'high'
                current_content = []
            elif 'MEDIUM IMPACT' in line_upper:
                if current_section and current_section != 'overview':
                    parsed[current_section].append('\n'.join(current_content))
                elif current_section == 'overview':
                    parsed[current_section] = '\n'.join(current_content)
                current_section = 'medium'
                current_content = []
            elif 'LOW IMPACT' in line_upper:
                if current_section and current_section != 'overview':
                    parsed[current_section].append('\n'.join(current_content))
                elif current_section == 'overview':
                    parsed[current_section] = '\n'.join(current_content)
                current_section = 'low'
                current_content = []
            elif '# WARNINGS' in line_upper or 'WARNINGS' in line_upper:
                if current_section and current_section != 'overview':
                    parsed[current_section].append('\n'.join(current_content))
                elif current_section == 'overview':
                    parsed[current_section] = '\n'.join(current_content)
                current_section = 'warnings'
                current_content = []
            elif '# SUMMARY' in line_upper:
                if current_section and current_section != 'overview':
                    parsed[current_section].append('\n'.join(current_content))
                elif current_section == 'overview':
                    parsed[current_section] = '\n'.join(current_content)
                current_section = 'summary'
                current_content = []
            elif current_section:
                current_content.append(line)
        
        # Add the last section
        if current_section and current_content:
            if current_section in ['overview', 'summary']:
                parsed[current_section] = '\n'.join(current_content)
            else:
                parsed[current_section].append('\n'.join(current_content))
        
        return parsed


if __name__ == '__main__':
    # Test the Gemini client
    import sys
    
    # Check if API key is available
    if not os.getenv('GEMINI_API_KEY'):
        print("Error: GEMINI_API_KEY environment variable not set")
        print("Please set it with: export GEMINI_API_KEY='your-api-key'")
        sys.exit(1)
    
    try:
        client = GeminiClient()
        print(f"Gemini client initialized successfully with model: {client.model_name}")
        
        # Test with minimal system info
        test_info = {
            'cpu': {'usage_average': 45.2, 'count_logical': 8},
            'memory': {'ram': {'percent': 67.3, 'total_gb': 16}},
            'disk': {'partitions': [{'mountpoint': '/', 'percent': 82}]}
        }
        
        print("\nTesting analysis with minimal system info...")
        result = client.analyze_system(test_info, focus_areas=['performance', 'memory'])
        
        if result['success']:
            print(f"Analysis successful!")
            print(f"Response length: {result['response_length']} characters")
            print("\nParsed recommendations:")
            parsed = client.parse_recommendations(result)
            for level in ['high', 'medium', 'low']:
                print(f"\n{level.upper()} impact items: {len(parsed.get(level, []))}")
        else:
            print(f"Analysis failed: {result.get('error')}")
            
    except Exception as e:
        print(f"Error: {e}")
