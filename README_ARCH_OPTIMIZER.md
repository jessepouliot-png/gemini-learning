# Arch Linux Optimizer using Gemini AI

A Python application that leverages Google's Gemini AI to analyze and optimize Arch Linux systems. This tool collects comprehensive system information and uses AI-powered analysis to provide actionable optimization recommendations.

## Features

### System Information Collection
- **CPU**: Usage statistics, load average, core count, and model information
- **Memory**: RAM and swap usage with detailed metrics
- **Disk**: Partition usage and I/O statistics
- **Services**: Systemd service analysis (running, enabled, failed)
- **Packages**: Package count, explicitly installed packages, and orphaned package detection
- **Boot**: Boot time analysis using systemd-analyze
- **Network**: Interface configuration and usage statistics
- **Kernel**: Version, architecture, and loaded modules count

### AI-Powered Analysis
- Performance bottleneck identification
- Memory optimization suggestions
- Disk space and I/O optimization
- Service optimization (unnecessary startup services)
- Package management recommendations
- Boot time improvement suggestions
- Security recommendations specific to Arch Linux

### Intelligent Recommendations
- Categorized by impact level (HIGH, MEDIUM, LOW)
- Specific commands and implementation steps
- Safety warnings and precautions
- Expected benefits for each recommendation

### Flexible Output
- Multiple report formats: Text, Markdown, JSON
- Console summary output
- Detailed report file generation
- Configurable analysis depth and focus areas

## Requirements

- Python 3.8 or higher
- Arch Linux system (or Arch-based distribution)
- Google Gemini API key
- System utilities: `systemctl`, `pacman`, `systemd-analyze` (usually pre-installed on Arch)

## Installation

1. Clone or download this repository:
```bash
git clone https://github.com/jessepouliot-png/gemini-learning.git
cd gemini-learning
```

2. Install required Python packages:
```bash
pip install -r requirements.txt
```

3. Set up your Gemini API key:
   - Get an API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Set it as an environment variable:
     ```bash
     export GEMINI_API_KEY='your-api-key-here'
     ```
   - Or create a `.env` file:
     ```bash
     echo "GEMINI_API_KEY=your-api-key-here" > .env
     ```

## Usage

### Basic Usage

Run with default settings (API key from environment):
```bash
python arch_optimizer.py
```

### Using Configuration File

1. Copy the example configuration:
```bash
cp config.example.json config.json
```

2. Edit `config.json` with your settings:
```json
{
  "api_key": "your-api-key-here",
  "model": "gemini-pro",
  "analysis_depth": "standard",
  "focus_areas": ["performance", "memory", "disk"],
  "output_format": "markdown",
  "report_file": "my_report.md"
}
```

3. Run with configuration:
```bash
python arch_optimizer.py --config config.json
```

### Command-Line Options

```bash
# Focus on specific areas
python arch_optimizer.py --focus performance memory services

# Change analysis depth
python arch_optimizer.py --depth deep

# Generate markdown report
python arch_optimizer.py --format markdown --output optimization_report.md

# System information only (no AI analysis)
python arch_optimizer.py --system-info-only

# Show help
python arch_optimizer.py --help
```

### Available Options

- `--config, -c`: Path to configuration file (JSON)
- `--api-key, -k`: Gemini API key (overrides environment)
- `--focus, -f`: Specific areas to analyze (performance, memory, disk, services, packages, boot_time, security)
- `--depth, -d`: Analysis depth (basic, standard, deep)
- `--format`: Output format (text, markdown, json)
- `--output, -o`: Output file path
- `--no-report`: Skip report generation (console only)
- `--system-info-only`: Only collect system info (no AI)
- `--version, -v`: Show version
- `--help`: Show help message

## Configuration

The configuration file (`config.json`) supports the following options:

```json
{
  "api_key": "YOUR_GEMINI_API_KEY_HERE",
  "model": "gemini-pro",
  "analysis_depth": "standard",
  "focus_areas": [
    "performance",
    "memory",
    "disk",
    "services",
    "packages",
    "boot_time",
    "security"
  ],
  "output_format": "detailed",
  "report_file": "arch_optimization_report.txt",
  "safety_mode": true,
  "max_recommendations": 50
}
```

### Configuration Options

- **api_key**: Your Google Gemini API key
- **model**: Gemini model to use (default: "gemini-pro")
- **analysis_depth**: Level of detail - "basic", "standard", or "deep"
- **focus_areas**: Array of areas to analyze
- **output_format**: Report format - "text", "markdown", or "json"
- **report_file**: Default output file name
- **safety_mode**: Enable safety checks (recommended)
- **max_recommendations**: Maximum number of recommendations

## Example Output

```
================================================================================
ARCH LINUX OPTIMIZATION REPORT
================================================================================
Generated: 2024-12-17 10:30:15
================================================================================

SYSTEM SUMMARY
--------------------------------------------------------------------------------
CPU Model: Intel Core i7-9750H
CPU Cores: 6 physical, 12 logical
CPU Usage: 23.4%
Load Average: (1.2, 1.5, 1.3)

RAM: 8.45GB / 15.52GB (54.4% used)
Swap: 0.00GB / 8.00GB (0.0% used)

Services: 142 running, 87 enabled
Packages: 1247 total, 234 explicit
Orphaned Packages: 15

================================================================================
OPTIMIZATION RECOMMENDATIONS
================================================================================

# HIGH IMPACT

1. Remove Orphaned Packages
   - 15 orphaned packages detected consuming disk space
   - Command: sudo pacman -Rns $(pacman -Qdtq)
   - Expected benefit: Free up ~500MB disk space
   - Warning: Review list before removal

...
```

## Safety Considerations

⚠️ **IMPORTANT SAFETY NOTES**

1. **Backup First**: Always backup your system before making changes
2. **Review Carefully**: Read each recommendation and understand its impact
3. **Test Safely**: Test changes in a non-production environment when possible
4. **Keep Records**: Document all changes for easy rollback
5. **Context Matters**: Some recommendations may not apply to your specific use case
6. **No Auto-Execution**: This tool NEVER automatically executes system modifications
7. **Reversible Changes**: All recommendations should be safe and reversible

## Use Cases

### Performance Optimization
```bash
python arch_optimizer.py --focus performance memory disk --depth deep
```

### Security Audit
```bash
python arch_optimizer.py --focus security services --output security_audit.md --format markdown
```

### Package Cleanup
```bash
python arch_optimizer.py --focus packages --depth basic
```

### Boot Time Analysis
```bash
python arch_optimizer.py --focus boot_time services
```

## Module Usage

You can also use the modules programmatically:

```python
from arch_optimizer import ArchOptimizer

# Initialize
optimizer = ArchOptimizer(api_key='your-key')

# Collect system info
system_info = optimizer.collect_system_info()

# Analyze
result = optimizer.analyze(
    focus_areas=['performance', 'memory'],
    analysis_depth='standard'
)

# Generate report
optimizer.generate_report('report.md', format='markdown')

# Print summary
optimizer.print_summary()
```

## Troubleshooting

### API Key Issues
```bash
# Check if API key is set
echo $GEMINI_API_KEY

# Set for current session
export GEMINI_API_KEY='your-key'

# Set permanently (add to ~/.bashrc or ~/.zshrc)
echo 'export GEMINI_API_KEY="your-key"' >> ~/.bashrc
```

### Permission Issues
Some system information requires elevated privileges:
```bash
# Run with sudo if needed
sudo -E python arch_optimizer.py
```

### Package Manager Not Found
This tool is designed for Arch Linux. On other systems, package information may not be available.

### Import Errors
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt --upgrade
```

## Project Structure

```
gemini-learning/
├── arch_optimizer/          # Main package
│   ├── __init__.py         # Package initialization
│   ├── system_info.py      # System information collector
│   ├── gemini_client.py    # Gemini AI integration
│   └── analyzer.py         # Analysis coordinator
├── arch_optimizer.py        # Main application entry point
├── requirements.txt         # Python dependencies
├── config.example.json      # Example configuration
└── README_ARCH_OPTIMIZER.md # This file
```

## Dependencies

- **google-generativeai**: Gemini AI API client
- **psutil**: System and process utilities
- **python-dotenv**: Environment variable management

## Contributing

Contributions are welcome! Please ensure:
- Code follows Python best practices
- All new features include documentation
- Safety considerations are maintained
- Changes are tested on Arch Linux

## License

This project is part of the gemini-learning repository. Please refer to the repository license.

## Disclaimer

This tool provides recommendations based on AI analysis. Always:
- Verify recommendations before implementing
- Understand the impact of changes
- Maintain backups of your system
- Use at your own risk

The authors are not responsible for any system issues arising from following the recommendations.

## Support

For issues, questions, or contributions, please visit:
https://github.com/jessepouliot-png/gemini-learning

## Acknowledgments

- Built with Google's Gemini AI
- Uses psutil for system information collection
- Designed for the Arch Linux community
