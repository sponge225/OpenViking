import os
import sys
import json
import yaml
import importlib
from argparse import ArgumentParser
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from src.core.logger import setup_logging
# ==========================================
# 1. Environment Initialization
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR

ov_config_path = os.path.join(SCRIPT_DIR, "ov.conf")
if os.path.exists(ov_config_path):
    os.environ["OPENVIKING_CONFIG_FILE"] = ov_config_path
    print(f"[Init] Auto-detected OpenViking config: {ov_config_path}")

try:
    from src.pipeline import BenchmarkPipeline
    from src.core.vector_store import VikingStoreWrapper, VikingStoreHTTPWrapper
    from src.core.vector_store_with_relations import VikingStoreWithRelations, VikingStoreHTTPWithRelations
    from src.core.llm_client import LLMClientWrapper
except SyntaxError as e:
    print(f"\n[Fatal Error] Syntax error while importing modules: {e}")
    sys.exit(1)
except ImportError as e:
    print(f"\n[Fatal Error] Cannot import modules: {e}")
    print(f"Current sys.path: {sys.path}\n")
    sys.exit(1)

# ==========================================
# 2. Helper Functions
# ==========================================

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def resolve_path(path_str, base_path):
    """
    Convert relative path to absolute path based on base_path.
    If path_str is already absolute, keep it unchanged.
    """
    if not path_str:
        return path_str
    if os.path.isabs(path_str):
        return path_str
    return os.path.normpath(os.path.join(base_path, path_str))

from src.vikingbot_runner import _generate_temp_ov_conf, _ensure_openviking_server, _load_server_url_and_key

# ==========================================
# 3. Main Program
# ==========================================

def main():
    parser = ArgumentParser(description="Run RAG Benchmark (Smart Path Handling)")
    default_config_path = os.path.join(SCRIPT_DIR, "config/config.yaml")
    
    parser.add_argument("--config", default=default_config_path, 
                        help=f"Path to config file. Default: {default_config_path}")
    
    parser.add_argument("--step", choices=["all", "import", "gen", "eval", "gen+eval", "del"], default="all", 
                        help="Execution step: 'import' (Ingest), 'gen' (Retrieve+LLM), 'eval' (Judge), 'gen+eval' (Gen then Eval), or 'all'")

    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint if available")
    
    args = parser.parse_args()

    # --- B. Load and Parse Config ---
    config_path = os.path.abspath(args.config)
    print(f"[Init] Loading configuration from: {config_path}")
    
    try:
        config = load_config(config_path)
    except FileNotFoundError as e:
        print(f"[Error] {e}")
        return

    # --- C. Path Resolution ---
    print(f"[Init] Resolving paths relative to Project Root: {PROJECT_ROOT}")
    dataset_name = config.get('dataset_name', 'UnknownDataset')
    retrieval_topk = config.get('execution', {}).get('retrieval_topk', 5)
    mode_for_paths = config.get('execution', {}).get('mode')
    if mode_for_paths == "ov_fallback_bot_relations":
        path_relations_topk = config.get('execution', {}).get('relations_topk', 0)
    else:
        path_relations_topk = config.get('vikingbot', {}).get('relations_topk', '')
    
    format_vars = {
        'dataset_name': dataset_name,
        'retrieval_topk': retrieval_topk,
        'search_limit': config.get('vikingbot', {}).get('search_limit', ''),
        'max_iterations': config.get('vikingbot', {}).get('max_iterations', ''),
        'relations_topk': path_relations_topk,
        'relations_similarity_threshold': config.get('vikingbot', {}).get('relations_similarity_threshold', ''),
    }
    
    path_keys = ['dataset_path', 'output_dir', 'vector_store', 'log_file', 'doc_output_dir']
    for key in path_keys:
        if key in config.get('paths', {}):
            original = config['paths'][key]
            rendered_path = original.format(**format_vars)
            resolved = resolve_path(rendered_path, PROJECT_ROOT)
            config['paths'][key] = resolved
            # print(f"  - {key}: {resolved}")

    # --- D. Initialize Components ---
    try:
        search_limit = config.get('vikingbot', {}).get('search_limit')
        llm_config = config.get('llm', None)
        server_port = config.get('execution', {}).get('server_port', None)
        if os.path.exists(ov_config_path):
            temp_conf_path = _generate_temp_ov_conf(
                ov_config_path,
                config['paths'].get('vector_store', ''),
                search_limit=search_limit,
                llm_config=llm_config,
                server_port=server_port,
            )
            os.environ["OPENVIKING_CONFIG_FILE"] = temp_conf_path
            print(f"[Init] Generated temporary ov.conf: {temp_conf_path}")

        logger = setup_logging(config['paths']['log_file'])
        logger.info(">>> Benchmark Session Started")
        
        # 1. Adapter (Dynamic Loading)
        adapter_cfg = config.get('adapter', {})
        module_path = adapter_cfg.get('module', 'src.adapters.locomo_adapter')
        class_name = adapter_cfg.get('class_name', 'LocomoAdapter')
        
        logger.info(f"Dynamically loading Adapter: {class_name} from {module_path}")
        logger.info(f"Loading dataset from: {config['paths']['dataset_path']}")
        
        try:
            mod = importlib.import_module(module_path)
            AdapterClass = getattr(mod, class_name)
            adapter = AdapterClass(raw_file_path=config['paths']['dataset_path'])
        except ImportError as e:
            logger.error(f"Could not import module '{module_path}'. Please check your config 'adapter.module'. Error: {e}")
            raise e
        except AttributeError as e:
            logger.error(f"Class '{class_name}' not found in module '{module_path}'. Please check your config 'adapter.class_name'. Error: {e}")
            raise e
        
        # 2. LLM Client (created before vector store, may be needed for query expansion)
        api_key = os.environ.get(
            config['llm'].get('api_key_env_var', ''),
            config['llm'].get('api_key')
        )
        api_key = os.path.expandvars(api_key) if api_key else api_key
        if not api_key or api_key.startswith("${"):
            logger.warning("No API Key found in config or environment variables!")
        llm_client = LLMClientWrapper(config=config['llm'], api_key=api_key)

        # 3. Vector Store
        mode = config.get('execution', {}).get('mode')
        if mode is None:
            if config.get('execution', {}).get('use_nanobot', False):
                mode = "nanobot"
            elif config.get('execution', {}).get('use_vikingbot', False):
                mode = "vikingbot"
            else:
                mode = "standard"

        if mode == "nanobot":
            vector_store = None
            logger.info("Nanobot mode: skipping VikingStoreWrapper initialization")
        elif mode in ("ov_fallback_bot", "ov_fallback_bot_relations"):
            vector_store_path = config['paths']['vector_store']
            search_limit = config.get('vikingbot', {}).get('search_limit')
            fallback_conf_path = _generate_temp_ov_conf(
                ov_config_path, vector_store_path,
                search_limit=search_limit,
                llm_config=config.get('llm'),
                server_port=config.get('execution', {}).get('server_port'),
            )
            _ensure_openviking_server(fallback_conf_path)
            server_url, api_key = _load_server_url_and_key(fallback_conf_path)
            if mode == "ov_fallback_bot_relations":
                embedder = None
                embedding_cfg = config.get('embedding', {})
                emb_api_key = embedding_cfg.get('api_key', '')
                emb_api_key = os.path.expandvars(emb_api_key) if emb_api_key else emb_api_key
                if emb_api_key and not emb_api_key.startswith("${"):
                    from src.core.embedder import VolcengineEmbedder
                    embedder = VolcengineEmbedder(
                        api_key=emb_api_key,
                        base_url=embedding_cfg.get('base_url', 'https://ark.cn-beijing.volces.com/api/v3'),
                        model=embedding_cfg.get('model', 'doubao-embedding-vision-250615'),
                    )
                link_strategy = config.get('execution', {}).get('link_strategy', 'llm_review')
                relations_topk = config.get('execution', {}).get('relations_topk', 0)
                relations_similarity_threshold = config.get('execution', {}).get('relations_similarity_threshold')
                vector_store = VikingStoreHTTPWithRelations(
                    server_url=server_url,
                    api_key=api_key,
                    store_path=vector_store_path,
                    embedder=embedder,
                    strategy=link_strategy,
                    relations_topk=relations_topk,
                    similarity_threshold=relations_similarity_threshold,
                )
                logger.info(
                    f"Fallback mode ({mode}): using HTTP wrapper with relations at {server_url} "
                    f"(relations_topk={relations_topk}, "
                    f"relations_similarity_threshold={relations_similarity_threshold})"
                )
            else:
                vector_store = VikingStoreHTTPWrapper(server_url=server_url, api_key=api_key)
                logger.info(f"Fallback mode ({mode}): using HTTP wrapper at {server_url}")
        else:
            use_relations = config.get('execution', {}).get('use_relations', False)
            if use_relations:
                relations_topk = config['execution'].get('relations_topk', 0)
                relations_similarity_threshold = config['execution'].get('relations_similarity_threshold')
                use_query_expansion = config['execution'].get('use_query_expansion', False)
                link_strategy = config['execution'].get('link_strategy', 'llm_review')

                embedder = None
                embedding_cfg = config.get('embedding', {})
                emb_api_key = embedding_cfg.get('api_key', '')
                emb_api_key = os.path.expandvars(emb_api_key) if emb_api_key else emb_api_key
                if emb_api_key and not emb_api_key.startswith("${"):
                    from src.core.embedder import VolcengineEmbedder
                    embedder = VolcengineEmbedder(
                        api_key=emb_api_key,
                        base_url=embedding_cfg.get('base_url', 'https://ark.cn-beijing.volces.com/api/v3'),
                        model=embedding_cfg.get('model', 'doubao-embedding-vision-250615'),
                    )
                    logger.info(f"Embedder initialized (model={embedding_cfg.get('model', 'doubao-embedding-vision-250615')})")
                else:
                    logger.warning("No embedding API key found, vector matching in relations will be disabled")

                vector_store = VikingStoreWithRelations(
                    store_path=config['paths']['vector_store'],
                    relations_topk=relations_topk,
                    use_query_expansion=use_query_expansion,
                    llm=llm_client if use_query_expansion else None,
                    embedder=embedder,
                    strategy=link_strategy,
                    similarity_threshold=relations_similarity_threshold,
                )
                logger.info(
                    "Using VikingStoreWithRelations "
                    f"(relations_topk={relations_topk}, query_expansion={use_query_expansion}, "
                    f"link_strategy={link_strategy}, "
                    f"relations_similarity_threshold={relations_similarity_threshold})"
                )
            else:
                vector_store = VikingStoreWrapper(store_path=config['paths']['vector_store'])

        # 4. Pipeline
        pipeline = BenchmarkPipeline(
            config=config,
            adapter=adapter,
            vector_db=vector_store,
            llm=llm_client,
            resume=args.resume,
        )

        # --- E. Execute Tasks ---
        if args.step in ["all", "import"]:
            logger.info("Stage: Import (Data Prepare + Ingest)")
            pipeline.run_import()
            
        if args.step in ["all", "gen", "gen+eval"]:
            logger.info("Stage: Generation (Retrieve + Generate)")
            pipeline.run_generation()
            
        if args.step in ["all", "eval", "gen+eval"]:
            logger.info("Stage: Evaluation (Judge -> Metrics)")
            pipeline.run_evaluation()

        if args.step in ["del"]:
            logger.info("Stage: Delete Vector Store")
            pipeline.run_deletion()
        
        logger.info("Benchmark finished successfully.")

    except KeyboardInterrupt:
        print("\n[Stop] Execution interrupted by user.")
    except Exception as e:
        if 'logger' in locals():
            logger.exception("Fatal error during execution")
        print(f"\n[Fatal Error] Program execution error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
