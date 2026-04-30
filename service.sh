#!/bin/bash

# ====================== 核心配置 ======================
CONDA_ENV="langchain-db"

# MCP 服务列表（脚本路径、端口、日志）
MCP_SERVICES=(
    "mcp_server/machine_learning_mcp.py:9003:log.machine_learning_mcp"
    "mcp_server/python_chart_mcp.py:9002:log.python_chart_mcp"
    "mcp_server/statistic_db_mcp_tools.py:9004:log.statistic_db_mcp_tools"
)

# 主服务
MAIN_SERVICE="mcp_server/multi_mcp_service.py"
MAIN_PORT=8000
MAIN_LOG="log.multi_mcp_service"
# ======================================================

# ---------------- 自动激活 conda 环境 ----------------
__conda_setup="$($HOME/miniconda3/bin/conda 'shell.bash' 'hook' 2>/dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/miniconda3/etc/profile.d/conda.sh"
    fi
fi
conda activate "$CONDA_ENV" >/dev/null 2>&1

# ---------------- 工具函数 ----------------
# 检查端口是否存活
port_alive() {
    ss -tuln 2>/dev/null | grep -q ":$1 "
    return $?
}

# 停止所有服务
stop_all() {
    pkill -f "python mcp_server/" 2>/dev/null
    sleep 1
    echo "🛑 所有服务已停止"
}

# 启动单个服务
start_service() {
    local script=$1
    local port=$2
    local log=$3
    if port_alive $port; then
        return
    fi
    nohup python "$script" > "$log" 2>&1 &
    sleep 0.8
}

# 启动全部（后台运行）
start_all() {
    stop_all
    echo "🚀 启动 3 个 MCP 服务..."
    for s in "${MCP_SERVICES[@]}"; do
        IFS=':' read -r script port log <<< "$s"
        start_service "$script" "$port" "$log"
    done

    echo "🔴 启动主服务..."
    start_service "$MAIN_SERVICE" $MAIN_PORT "$MAIN_LOG"

    echo "✅ 所有服务已后台启动完成！"
    echo "👉 打开监控：./mcp_service.sh monitor"
}

# ---------------- 实时监控面板（单独功能） ----------------
monitor_panel() {
    echo "✅ 已进入实时监控面板（1秒刷新） | 按 Ctrl+C 退出"
    while true; do
        clear
        echo "==================== MCP 服务实时监控 ===================="
        echo "环境：$CONDA_ENV    刷新：$(date '+%H:%M:%S')"
        echo "=========================================================="

        # 检查 MCP 状态
        mcp_down=0
        for s in "${MCP_SERVICES[@]}"; do
            IFS=':' read -r script port log <<< "$s"
            name=$(basename "$script")
            if port_alive $port; then
                echo -e "✅ $port   $name    正常运行"
            else
                echo -e "❌ $port   $name    已断开"
                mcp_down=1
            fi
        done

        # 检查主服务
        echo "----------------------------------------------------------"
        if port_alive $MAIN_PORT; then
            echo -e "✅ 8000   主服务 multi_mcp_service    正常运行"
        else
            echo -e "❌ 8000   主服务 multi_mcp_service    已停止"
        fi

        echo "=========================================================="
        echo "命令说明：start=启动 | stop=停止 | monitor=监控 | Ctrl+C=退出"
        echo "=========================================================="

        # 自动重启主服务（MCP异常时）
        if [ $mcp_down -eq 1 ]; then
            echo -e "\n⚠️  MCP 服务异常，正在自动重启主服务..."
            pkill -f "python $MAIN_SERVICE" 2>/dev/null
            sleep 2
            start_service "$MAIN_SERVICE" $MAIN_PORT "$MAIN_LOG"
            echo "✅ 主服务已重启"
            sleep 1
        fi

        sleep 1
    done
}

# ---------------- 命令入口 ----------------
case "$1" in
    start)
        start_all
        ;;
    stop)
        stop_all
        ;;
    monitor)
        monitor_panel
        ;;
    restart)
        start_all
        ;;
    *)
        echo "用法："
        echo "  ./mcp_service.sh start      # 后台启动所有服务"
        echo "  ./mcp_service.sh stop       # 停止所有服务"
        echo "  ./mcp_service.sh restart    # 重启所有服务"
        echo "  ./mcp_service.sh monitor    # 单独打开实时监控面板（一直刷新）"
        ;;
esac