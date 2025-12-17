# Quick Start Guide - Arch Linux Optimizer

Get started with the Arch Linux Optimizer in 5 minutes!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Get a Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy your new API key

## Step 3: Set Your API Key

**Option A: Environment Variable (Recommended)**
```bash
export GEMINI_API_KEY='your-api-key-here'
```

**Option B: .env File**
```bash
cp .env.example .env
# Edit .env and add your API key
```

**Option C: Configuration File**
```bash
cp config.example.json config.json
# Edit config.json and add your API key
```

## Step 4: Run Your First Analysis

**View System Information (No API Key Required)**
```bash
python arch_optimizer.py --system-info-only
```

**Run Full AI Analysis**
```bash
python arch_optimizer.py
```

**Focus on Specific Areas**
```bash
python arch_optimizer.py --focus performance memory
```

**Generate Markdown Report**
```bash
python arch_optimizer.py --format markdown --output my_report.md
```

## Common Use Cases

### Performance Optimization
```bash
python arch_optimizer.py --focus performance memory disk --depth deep
```

### Security Audit
```bash
python arch_optimizer.py --focus security services
```

### Package Cleanup
```bash
python arch_optimizer.py --focus packages
```

### Boot Time Analysis
```bash
python arch_optimizer.py --focus boot_time services
```

## What You'll Get

The optimizer will:
1. ✅ Collect comprehensive system information
2. ✅ Analyze it using Gemini AI
3. ✅ Provide categorized recommendations (HIGH, MEDIUM, LOW impact)
4. ✅ Generate a detailed report with specific commands
5. ✅ Include safety warnings for each recommendation

## Example Output

```
================================================================================
OPTIMIZATION RECOMMENDATIONS
================================================================================

# HIGH IMPACT

1. Remove Orphaned Packages
   - 15 orphaned packages detected consuming disk space
   - Command: sudo pacman -Rns $(pacman -Qdtq)
   - Expected benefit: Free up ~500MB disk space
   - ⚠️ Warning: Review list before removal

2. Disable Unnecessary Services
   - cups.service is running but printer not in use
   - Command: sudo systemctl disable cups.service
   - Expected benefit: Faster boot time, reduced memory usage
   ...

# MEDIUM IMPACT
   ...

# LOW IMPACT
   ...
```

## Need Help?

- 📖 **Full Documentation**: See [README_ARCH_OPTIMIZER.md](README_ARCH_OPTIMIZER.md)
- 💻 **Example Code**: Check [example_usage.py](example_usage.py)
- ❓ **Issues**: Visit the [GitHub repository](https://github.com/jessepouliot-png/gemini-learning)

## Safety First! ⚠️

**Important:**
- Always backup your system before making changes
- Review each recommendation carefully
- Test in a safe environment when possible
- The tool NEVER auto-executes changes - you have full control

## Next Steps

1. ✅ Review your optimization report
2. ✅ Research any unfamiliar recommendations
3. ✅ Backup your system
4. ✅ Implement changes one at a time
5. ✅ Monitor system behavior after changes

Happy optimizing! 🚀
