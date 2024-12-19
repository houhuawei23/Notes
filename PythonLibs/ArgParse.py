import argparse


def main():
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(
        description="一个综合示例，展示 argparse 的各种用法"
    )

    # 添加位置参数
    parser.add_argument("filename", help="输入文件名")

    # 添加可选参数
    parser.add_argument("-v", "--verbose", action="store_true", help="启用详细输出")
    parser.add_argument("-o", "--output", help="输出文件名")

    # 添加带默认值的参数
    parser.add_argument(
        "--timeout", type=int, default=30, help="超时时间，默认为 30 秒"
    )

    # 添加带类型转换的参数
    parser.add_argument("--port", type=int, help="端口号")

    parser.add_argument("--train", type=bool)
    # 添加互斥选项
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--start", action="store_true", help="启动服务")
    group.add_argument("--stop", action="store_true", help="停止服务")

    # 添加子命令
    subparsers = parser.add_subparsers(title="子命令", dest="command")

    # 子命令 'config'
    config_parser = subparsers.add_parser("config", help="配置相关命令")
    config_parser.add_argument("--set", help="设置配置项")
    config_parser.add_argument("--get", help="获取配置项")

    # 子命令 'status'
    status_parser = subparsers.add_parser("status", help="状态相关命令")
    status_parser.add_argument("--detail", action="store_true", help="显示详细状态")


    # 解析命令行参数
    args = parser.parse_args()

    # 输出解析结果
    print(f"文件名: {args.filename}")
    print(f"详细输出: {args.verbose}")
    print(f"输出文件名: {args.output}")
    print(f"超时时间: {args.timeout}")
    print(f"端口号: {args.port}")
    print(f"启动服务: {args.start}")
    print(f"停止服务: {args.stop}")
    print(f"子命令: {args.command}")
    print(f"train: {args.train}")
    if args.command == "config":
        print(f"设置配置项: {args.set}")
        print(f"获取配置项: {args.get}")
    elif args.command == "status":
        print(f"显示详细状态: {args.detail}")


if __name__ == "__main__":
    main()
