def test_imports():
    import importlib

    # 把你项目里的重要模块名填进去，能 import 就算通过
    modules = [
        "models",
        "utils",
        "dispatching",
        "train_lstm_transformer",  # 如果根目录有这个文件
    ]

    for m in modules:
        importlib.import_module(m)
