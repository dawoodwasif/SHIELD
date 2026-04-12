#!/bin/bash
echo "🔍 Monitoring IRS Evaluation Progress..."
echo "======================================="
while true; do
    echo ""
    echo "📊 Current Status ($(date)):"
    echo "----------------------------"
    
    for dir in ./train_dir/irs_*; do
        if [ -d "$dir" ]; then
            exp_name=$(basename "$dir")
            if [ -f "$dir/sf_log.txt" ]; then
                last_line=$(tail -1 "$dir/sf_log.txt" 2>/dev/null || echo "No log data")
                echo "• $exp_name: $last_line"
            else
                echo "• $exp_name: Starting..."
            fi
        fi
    done
    
    echo ""
    echo "🛡️ Looking for IRS activity..."
    grep -h "IRS\|SHIELD\|attack\|detection" ./train_dir/irs_*/sf_log.txt 2>/dev/null | tail -5 || echo "No IRS activity logged yet"
    
    sleep 30
done
