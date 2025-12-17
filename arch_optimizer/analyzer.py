#!/usr/bin/env python3
"""
System Analyzer Module

This module coordinates system analysis using collected information
and Gemini AI to generate optimization recommendations.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from .system_info import SystemInfoCollector
from .gemini_client import GeminiClient


class ArchOptimizer:
    """Main analyzer class that coordinates system analysis and optimization."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        init_gemini: bool = True
    ):
        """
        Initialize the Arch Optimizer.
        
        Args:
            api_key: Gemini API key
            config: Configuration dictionary
            init_gemini: Whether to initialize Gemini client (default: True)
        """
        self.config = config or {}
        
        # Initialize components
        self.system_collector = SystemInfoCollector()
        self.gemini_client = None
        
        # Only initialize Gemini client if requested and API key is available
        if init_gemini:
            self.gemini_client = GeminiClient(
                api_key=api_key or self.config.get('api_key'),
                model=self.config.get('model', 'gemini-1.5-flash')
            )
        
        # Store results
        self.system_info = None
        self.analysis_result = None
        self.parsed_recommendations = None

    def collect_system_info(self) -> Dict[str, Any]:
        """
        Collect system information.
        
        Returns:
            Dictionary containing system information
        """
        print("Collecting system information...")
        self.system_info = self.system_collector.collect_all()
        print("System information collected successfully.")
        return self.system_info

    def analyze(
        self,
        focus_areas: Optional[List[str]] = None,
        analysis_depth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform system analysis using Gemini AI.
        
        Args:
            focus_areas: List of areas to focus on
            analysis_depth: Level of analysis (basic, standard, deep)
            
        Returns:
            Analysis results
        """
        # Check if Gemini client is initialized
        if not self.gemini_client:
            return {
                'success': False,
                'error': 'Gemini client not initialized. API key required for analysis.'
            }
        
        # Collect system info if not already collected
        if not self.system_info:
            self.collect_system_info()
        
        # Use config values if not provided
        focus_areas = focus_areas or self.config.get('focus_areas', [
            'performance', 'memory', 'disk', 'services',
            'packages', 'boot_time', 'security'
        ])
        analysis_depth = analysis_depth or self.config.get('analysis_depth', 'standard')
        
        print(f"\nAnalyzing system with Gemini AI...")
        print(f"Focus areas: {', '.join(focus_areas)}")
        print(f"Analysis depth: {analysis_depth}")
        
        # Perform analysis
        self.analysis_result = self.gemini_client.analyze_system(
            self.system_info,
            focus_areas=focus_areas,
            analysis_depth=analysis_depth
        )
        
        if self.analysis_result.get('success'):
            print("Analysis completed successfully.")
            # Parse recommendations
            self.parsed_recommendations = self.gemini_client.parse_recommendations(
                self.analysis_result
            )
        else:
            print(f"Analysis failed: {self.analysis_result.get('error')}")
        
        return self.analysis_result

    def generate_report(
        self,
        output_file: Optional[str] = None,
        format: str = 'text'
    ) -> str:
        """
        Generate a comprehensive optimization report.
        
        Args:
            output_file: Path to output file. If None, uses config or default.
            format: Report format (text, json, markdown)
            
        Returns:
            Path to generated report file
        """
        if not self.analysis_result:
            raise ValueError("No analysis results available. Run analyze() first.")
        
        # Determine output file
        if not output_file:
            output_file = self.config.get('report_file', 'arch_optimization_report.txt')
        
        # Generate report content based on format
        if format == 'json':
            content = self._generate_json_report()
        elif format == 'markdown':
            content = self._generate_markdown_report()
        else:  # text
            content = self._generate_text_report()
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write(content)
        
        print(f"\nReport generated: {output_file}")
        return output_file

    def _generate_text_report(self) -> str:
        """Generate text format report."""
        lines = []
        lines.append("=" * 80)
        lines.append("ARCH LINUX OPTIMIZATION REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)
        lines.append("")
        
        # System summary
        lines.append("SYSTEM SUMMARY")
        lines.append("-" * 80)
        if self.system_info:
            lines.append(self.system_collector.get_formatted_summary())
        lines.append("")
        
        # Recommendations
        lines.append("OPTIMIZATION RECOMMENDATIONS")
        lines.append("-" * 80)
        
        if self.analysis_result.get('success'):
            lines.append(self.analysis_result.get('recommendations', ''))
        else:
            lines.append(f"ERROR: {self.analysis_result.get('error', 'Unknown error')}")
        
        lines.append("")
        lines.append("=" * 80)
        lines.append("IMPORTANT SAFETY NOTES")
        lines.append("=" * 80)
        lines.append("1. Always backup your system before making changes")
        lines.append("2. Review each recommendation carefully before implementing")
        lines.append("3. Test changes in a non-production environment when possible")
        lines.append("4. Keep a record of changes made for easy rollback")
        lines.append("5. Some recommendations may not apply to your specific use case")
        lines.append("=" * 80)
        
        return '\n'.join(lines)

    def _generate_markdown_report(self) -> str:
        """Generate markdown format report."""
        lines = []
        lines.append("# Arch Linux Optimization Report")
        lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append("---\n")
        
        # System summary
        lines.append("## System Summary\n")
        if self.system_info:
            summary = self.system_collector.get_formatted_summary()
            lines.append("```")
            lines.append(summary)
            lines.append("```\n")
        
        # Recommendations
        lines.append("## Optimization Recommendations\n")
        if self.analysis_result.get('success'):
            lines.append(self.analysis_result.get('recommendations', ''))
        else:
            lines.append(f"**ERROR:** {self.analysis_result.get('error', 'Unknown error')}")
        
        lines.append("\n---\n")
        lines.append("## ⚠️ Important Safety Notes\n")
        lines.append("1. **Always backup your system before making changes**")
        lines.append("2. Review each recommendation carefully before implementing")
        lines.append("3. Test changes in a non-production environment when possible")
        lines.append("4. Keep a record of changes made for easy rollback")
        lines.append("5. Some recommendations may not apply to your specific use case")
        
        return '\n'.join(lines)

    def _generate_json_report(self) -> str:
        """Generate JSON format report."""
        report = {
            'generated_at': datetime.now().isoformat(),
            'system_info': self.system_info,
            'analysis': {
                'success': self.analysis_result.get('success'),
                'model': self.analysis_result.get('model'),
                'recommendations': self.analysis_result.get('recommendations'),
                'error': self.analysis_result.get('error')
            },
            'parsed_recommendations': self.parsed_recommendations,
            'config': self.config
        }
        return json.dumps(report, indent=2, default=str)

    def print_summary(self):
        """Print a summary of the analysis to console."""
        if not self.parsed_recommendations:
            print("No analysis results available.")
            return
        
        print("\n" + "=" * 80)
        print("OPTIMIZATION SUMMARY")
        print("=" * 80)
        
        if 'overview' in self.parsed_recommendations:
            print("\nOVERVIEW:")
            print(self.parsed_recommendations['overview'])
        
        print(f"\nRECOMMENDATION COUNT:")
        for level in ['high', 'medium', 'low']:
            count = len(self.parsed_recommendations.get(level, []))
            print(f"  {level.upper()} impact: {count} items")
        
        if self.parsed_recommendations.get('warnings'):
            print("\n⚠️  WARNINGS:")
            if isinstance(self.parsed_recommendations['warnings'], list):
                for warning in self.parsed_recommendations['warnings']:
                    print(f"  - {warning}")
            else:
                print(f"  {self.parsed_recommendations['warnings']}")
        
        if 'summary' in self.parsed_recommendations:
            print("\nSUMMARY:")
            print(self.parsed_recommendations['summary'])
        
        print("=" * 80)


def load_config(config_file: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    return config


if __name__ == '__main__':
    # Test the analyzer
    import sys
    
    if not os.getenv('GEMINI_API_KEY'):
        print("Error: GEMINI_API_KEY environment variable not set")
        sys.exit(1)
    
    try:
        # Create optimizer
        optimizer = ArchOptimizer()
        
        # Collect system info
        optimizer.collect_system_info()
        print("\n" + optimizer.system_collector.get_formatted_summary())
        
        # Analyze (with limited focus for testing)
        optimizer.analyze(focus_areas=['performance', 'memory'], analysis_depth='basic')
        
        # Print summary
        optimizer.print_summary()
        
        # Generate report
        optimizer.generate_report('test_report.txt', format='text')
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
