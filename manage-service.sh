#!/bin/bash
# Arch Optimizer Service Management Script

SERVICE_NAME="arch-optimizer.service"

show_help() {
    echo "Arch Linux Optimizer Service Manager"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  status    - Show service status"
    echo "  start     - Start the service manually"
    echo "  stop      - Stop the service"
    echo "  enable    - Enable automatic startup"
    echo "  disable   - Disable automatic startup"
    echo "  logs      - View service logs"
    echo "  report    - View the latest analysis report"
    echo "  remove    - Completely remove the service"
    echo "  help      - Show this help message"
    echo ""
}

case "$1" in
    "status")
        sudo systemctl status $SERVICE_NAME --no-pager
        ;;
    "start")
        echo "Starting Arch Optimizer service..."
        sudo systemctl start $SERVICE_NAME
        echo "✓ Service started"
        ;;
    "stop")
        echo "Stopping Arch Optimizer service..."
        sudo systemctl stop $SERVICE_NAME
        echo "✓ Service stopped"
        ;;
    "enable")
        echo "Enabling automatic startup..."
        sudo systemctl enable $SERVICE_NAME
        echo "✓ Service will start automatically at boot"
        ;;
    "disable")
        echo "Disabling automatic startup..."
        sudo systemctl disable $SERVICE_NAME
        echo "✓ Service will not start automatically at boot"
        ;;
    "logs")
        echo "Recent service logs:"
        sudo journalctl -u $SERVICE_NAME --no-pager -n 50
        echo ""
        echo "Full analysis log:"
        if [[ -f "$HOME/arch-optimizer-startup.log" ]]; then
            tail -30 "$HOME/arch-optimizer-startup.log"
        else
            echo "Log file not found"
        fi
        ;;
    "report")
        if [[ -f "arch_optimization_report.txt" ]]; then
            echo "Latest analysis report:"
            echo "======================"
            cat arch_optimization_report.txt
        else
            echo "No report found. Run the service first."
        fi
        ;;
    "remove")
        echo "Removing Arch Optimizer service..."
        sudo systemctl stop $SERVICE_NAME 2>/dev/null
        sudo systemctl disable $SERVICE_NAME 2>/dev/null
        sudo rm -f "/etc/systemd/system/$SERVICE_NAME"
        sudo systemctl daemon-reload
        rm -f "/home/jpouliot/arch-optimizer-startup.log"
        echo "✓ Service removed completely"
        ;;
    "help"|"")
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        show_help
        exit 1
        ;;
esac