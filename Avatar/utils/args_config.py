import json
import os
import argparse
import yaml
args = None

def parse_hp_string(hp_string):
    result = {}
    for pair in hp_string.split(','):
        if not pair:
            continue
        key, value = pair.split('=')
        try:
            # 自动转换为 int / float / str
            ori_value = value
            value = float(value)
            if '.' not in str(ori_value):
                value = int(value)
        except ValueError:
            pass

        if value in ['true', 'True']:
            value = True
        if value in ['false', 'False']:
            value = False
        if '.' in key:
            keys = key.split('.')
            keys = keys
            current = result
            for key in keys[:-1]:
                if key not in current or not isinstance(current[key], dict):
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = value
        else:
            result[key.strip()] = value
    return result

def parse_args():
    global args
    parser = argparse.ArgumentParser(description="Run Chanjing-Avatar inference.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    
    # 定义 argparse 参数
    parser.add_argument("--exp_path", type=str, help="Path to save the model.")
    parser.add_argument("--input_file", type=str, help="Batch file using prompt@@media@@audio on each line.")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt for one inference job.")
    parser.add_argument("--image_path", type=str, default=None, help="Reference image for 14B image-to-video inference.")
    parser.add_argument("--audio_path", type=str, default=None, help="Driving audio; overrides the path in --input_file.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for inference results.")
    parser.add_argument("--result_prefix", type=str, default=None, help="Output filename prefix.")
    parser.add_argument("--validate_only", action="store_true", help="Validate config, model paths, and media without loading models.")
    parser.add_argument("--debug", action='store_true', default=None)
    parser.add_argument("--infer", action='store_true')
    parser.add_argument("-hp", "--hparams", type=str, default="")

    args = parser.parse_args()

    # 读取 YAML 配置（如果提供了 --config 参数）
    if args.config:
        with open(args.config, "r") as f:
            yaml_config = yaml.safe_load(f)
        
        yaml_config = _resolve_placeholders(yaml_config)
        # 遍历 YAML 配置，将其添加到 args（如果 argparse 里没有定义）
        for key, value in yaml_config.items():
            if not hasattr(args, key):  # argparse 没有的参数
                setattr(args, key, value)
            elif getattr(args, key) is None:  # argparse 有但值为空
                setattr(args, key, value)

    args.rank = int(os.getenv("RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))  # torchrun
    args.device = f'cuda:{args.local_rank}'
    args.num_nodes = int(os.getenv("NNODES", "1"))
    debug = args.debug
    if hasattr(args, 'reload_cfg') and args.reload_cfg:
        # 重新加载配置文件
        conf_path = os.path.join(args.exp_path, "config.json")
        if os.path.exists(conf_path):
            print('| Reloading config from:', conf_path)
            args = reload(args, conf_path)
    if len(args.hparams) > 0:
        hp_dict = parse_hp_string(args.hparams)
        for key, value in hp_dict.items():
            if not hasattr(args, key):
                setattr(args, key, value)
            else:
                if isinstance(value, dict):
                    ori_v = getattr(args, key)
                    ori_v.update(value)
                    setattr(args, key, ori_v)
                else:
                    setattr(args, key, value)
    args.debug = debug
    dict_args = convert_namespace_to_dict(args)
    if args.local_rank == 0:
        print(dict_args)
    return args

def reload(args, conf_path):
    """重新加载配置文件,不覆盖已有的参数"""
    with open(conf_path, "r") as f:
        yaml_config = _resolve_placeholders(yaml.safe_load(f))
    # 遍历 YAML 配置，将其添加到 args（如果 argparse 里没有定义）
    for key, value in yaml_config.items():
        if not hasattr(args, key):  # argparse 没有的参数
            setattr(args, key, value)
        elif getattr(args, key) is None:  # argparse 有但值为空
            setattr(args, key, value)
    return args

def _resolve_placeholders(cfg):
    """Resolve environment variables and ${key} references in a YAML mapping."""
    if not isinstance(cfg, dict):
        return cfg
    resolved = {
        key: os.path.expandvars(value) if isinstance(value, str) else value
        for key, value in cfg.items()
    }
    for _ in range(len(resolved)):
        changed = False
        for key, value in resolved.items():
            if not isinstance(value, str):
                continue
            new_value = value
            for ref_key, ref_value in resolved.items():
                if isinstance(ref_value, str):
                    new_value = new_value.replace(f"${{{ref_key}}}", ref_value)
            changed |= new_value != value
            resolved[key] = new_value
        if not changed:
            break
    return resolved

def convert_namespace_to_dict(namespace):
    """将 argparse.Namespace 转为字典，并处理不可序列化对象"""
    result = {}
    for key, value in vars(namespace).items():
        try:
            json.dumps(value)  # 检查是否可序列化
            result[key] = value
        except (TypeError, OverflowError):
            result[key] = str(value)  # 将不可序列化的对象转为字符串表示
    return result
