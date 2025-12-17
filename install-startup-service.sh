#!/bin/bash
# Arch Optimizer Startup Service Installer

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_FILE="$SCRIPT_DIR/arch-optimizer.service"
SYSTEMD_DIR="/etc/systemd/system"
SERVICE_NAME="arch-optimizer.service"

echo "Installing Arch Linux Optimizer startup service..."

# Check if running as root for installation
if [[ $EUID -eq 0 ]]; then
    echo "Error: Please run this script as a regular user, not as root."
    echo "The script will use sudo when needed."
    exit 1
fi

# Check if service file exists
if [[ ! -f "$SERVICE_FILE" ]]; then
    echo "Error: Service file not found at $SERVICE_FILE"
    exit 1
fi

# Copy service file to systemd directory
echo "Copying service file to $SYSTEMD_DIR..."
sudo cp "$SERVICE_FILE" "$SYSTEMD_DIR/$SERVICE_NAME"

# Set proper permissions
sudo chmod 644 "$SYSTEMD_DIR/$SERVICE_NAME"

# Create log file in user home directory
touch "$HOME/arch-optimizer-startup.log"

# Reload systemd daemon
echo "Reloading systemd daemon..."
sudo systemctl daemon-reload

# Enable the service
echo "Enabling arch-optimizer service..."
sudo systemctl enable "$SERVICE_NAME"

echo ""
echo "✓ Arch Optimizer startup service installed and enabled!"
echo ""
echo "The service will now run automatically at every system startup."
echo ""
echo "Useful commands:"
echo "  Check status:    sudo systemctl status arch-optimizer"
echo "  View logs:       sudo journalctl -u arch-optimizer"
echo "  View reports:    cat /home/jpouliot/arch-optimizer-startup.log"
echo "  Disable service: sudo systemctl disable arch-optimizer"
echo "  Manual run:      sudo systemctl start arch-optimizer"
echo ""
echo "Log files:"
echo "  Service logs: /home/jpouliot/arch-optimizer-startup.log"
echo "  Reports will be saved to: arch_optimization_report.txt in project directory"
echo ""