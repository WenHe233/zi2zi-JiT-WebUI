from __future__ import annotations


TRANSLATIONS = {
    "zh": {
        "app_title": "zi2zi-JiT 字体工作室",
        "projects": "项目",
        "model_assets": "模型与素材",
        "dataset": "数据集",
        "training": "微调与监控",
        "generation": "生成",
        "review": "字形审校",
        "export": "字体导出",
        "advanced": "高级与任务",
        "create_project": "新建项目",
        "project_name": "项目名称",
        "refresh": "刷新",
        "save": "保存",
        "queue": "加入队列",
        "cancel": "取消任务",
    },
    "en": {
        "app_title": "zi2zi-JiT Font Studio",
        "projects": "Projects",
        "model_assets": "Models & assets",
        "dataset": "Dataset",
        "training": "Fine-tune & monitor",
        "generation": "Generate",
        "review": "Glyph review",
        "export": "Font export",
        "advanced": "Advanced & jobs",
        "create_project": "Create project",
        "project_name": "Project name",
        "refresh": "Refresh",
        "save": "Save",
        "queue": "Queue job",
        "cancel": "Cancel job",
    },
}


def normalize_language(language: str | None) -> str:
    value = (language or "zh").lower()
    return "en" if value.startswith("en") else "zh"


def translator(language: str | None):
    messages = TRANSLATIONS[normalize_language(language)]
    return lambda key: messages.get(key, key)
